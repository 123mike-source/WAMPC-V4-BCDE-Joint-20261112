"""Eight-GPU Slurm evaluation using the existing route queue and scorer."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue
from evaluation.carla_eval_resources import checked_evaluation_path, checked_run_root, cpu_groups, memory_snapshot
from evaluation.tongji_slurm import (
    cuda_visible_uuids, gpu_inventory, process_gpu_contexts, verify_render_contexts,
    verify_slurm_job, vulkan_environment, vulkan_gpu_map,
)

EXPECTED_CHECKPOINT = 'ea3eea848318c86cef17b9bc59a5a8c7fa17eed345123a5d359967a5323eb027'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    if args.inference_steps != 25 or args.lanes != 2:
        raise ValueError('This prepared experiment uses 25 DDIM steps and two lanes per GPU')
    if args.run_root.exists():
        raise FileExistsError('Use a new run directory for a new evaluation')
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('The evaluated epoch-10 V3 checkpoint is required')
    paths = json.loads((ROOT / 'configs/paths.json').read_text())
    if paths['garage_root'] != '/share/home/u21012/data/bench2drive-PDMlite':
        raise ValueError('Dataset path has not been configured as requested')
    if not Path(paths['garage_root']).is_dir():
        raise FileNotFoundError(paths['garage_root'])
    runtime_source = json.loads((args.carla_root / 'runtime_source.json').read_text())
    args.run_root.mkdir(parents=True)
    sources = set()
    for directory in ('evaluation', 'method/full_diffusion', 'method/full_diffusion2',
                      'method/full_diffusionv3', 'third_party/carla_garage/team_code'):
        sources.update(path for path in (ROOT / directory).rglob('*.py') if '__pycache__' not in path.parts)
    sources.update(ROOT / path for path in (
        'configs/paths.json', 'method/full_diffusionv3/config.json',
        'leaderboard/data/bench2drive220.xml', 'leaderboard/leaderboard/leaderboard_evaluator.py',
        'evaluation/launch_tongji_v3_25.sh', 'evaluation/tongji_v3_in_job.sh',
        'evaluation/attach_tongji_v3_25.sh',
        'tools/prepare_tongji_v3_runtime.py'))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(sources)}
    save(args.run_root / 'source_sha256.json', hashes)
    identity = {'checkpoint_sha256': EXPECTED_CHECKPOINT, 'seed': args.seed,
                'inference_steps': args.inference_steps, 'backend': 'direct',
                'method': 'full_diffusionv3', 'source_sha256': hashes}
    routes = [dict(id=r.attrib['id'], town=r.attrib['town']) for r in
              ET.parse(ROOT / 'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    routes.sort(key=lambda route: (route['town'], int(route['id'])))
    if len(routes) != 220 or len({r['id'] for r in routes}) != 220:
        raise ValueError('Expected all 220 unique Bench2Drive routes')
    RouteQueue(args.run_root).initialize(routes, identity)
    save(args.run_root / 'prepared.json', {
        'unix': time.time(), 'identity': identity, 'route_count': len(routes),
        'requested_gpus': 8, 'requested_cpus': 56, 'requested_memory': 'all node memory (--mem=0)',
        'lanes_per_gpu': args.lanes, 'total_lanes': 8 * args.lanes,
        'maximum_concurrent_carla_initializations': 2,
        'writable_carla_state': 'Private Config/Saved/Intermediate/cache directories per GPU lane',
        'runtime': str(args.carla_root), 'runtime_source': runtime_source,
        'dataset_root': paths['garage_root'], 'checkpoint': str(args.checkpoint),
        'buffered_metrics': True, 'scoring': 'Original Bench2Drive scorer; driving failures retained',
        'gpu_readiness': 'Each GPU must pass a live RPC/map/RGB camera probe after allocation',
    })
    print(json.dumps({'prepared': str(args.run_root), 'routes': 220, 'steps': 25, 'lanes': 16}), flush=True)


def check_sources(args):
    hashes = json.loads((args.run_root / 'source_sha256.json').read_text())
    changed = [name for name, digest in hashes.items() if sha256(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f'Prepared source files changed: {changed}')


def lane_arguments(args):
    return SimpleNamespace(**{**vars(args), 'job_id': os.environ['SLURM_JOB_ID'],
        'method': 'full_diffusionv3', 'gpu': args.graphics_adapter, 'gpu_uuid': args.gpu_uuid,
        'backend': 'direct', 'slot_base': 1000 + args.logical_gpu * 100,
        'tm_base': 40000 + args.logical_gpu * 2048, 'cpus_per_gpu': 7,
        'server_mode': 'warm', 'startup_wait': 'rpc', 'routes_per_server': 32,
        'capture_first_policy_call': False})


def runtime_probe(args, group):
    """Generate readiness evidence from the allocated GPU, never a placeholder."""
    import carla
    from evaluation.run_full_diffusion2_carla import start_lane_server
    probe = group / 'runtime_probe'
    probe.mkdir(parents=True)
    settings_args = lane_arguments(args)
    settings_args.lane = 0
    owner = None
    camera = None
    vehicle = None
    world = None
    old_settings = None
    try:
        owner, _ = start_lane_server(settings_args, probe / 'servers', 0)
        client = carla.Client('localhost', 2000 + settings_args.slot * 10)
        client.set_timeout(180.0)
        if client.get_server_version() != '0.9.15':
            raise RuntimeError('Unexpected CARLA RPC version')
        town = 'Town13' if args.logical_gpu == 0 else 'Town01'
        world = client.load_world(town)
        if world.get_map().name.split('/')[-1] != town:
            raise RuntimeError('Requested preflight map did not load')
        old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.tile_stream_distance = 650.0
        settings.actor_active_distance = 650.0
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        blueprint = library.find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', 'hero')
        for transform in world.get_map().get_spawn_points()[:20]:
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError('Could not spawn a preflight ego vehicle')
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        blueprint = library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '160')
        blueprint.set_attribute('image_size_y', '90')
        camera = world.spawn_actor(blueprint, carla.Transform(carla.Location(z=2.3)), attach_to=vehicle)
        images = queue.Queue(maxsize=8)
        def received(image):
            try:
                images.put_nowait(image)
            except queue.Full:
                pass
        camera.listen(received)
        image = None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            world.tick(60.0)
            try:
                candidate = images.get(timeout=1.0)
                raw = bytes(candidate.raw_data)
                pixels = raw[0::4] + raw[1::4] + raw[2::4]
                if max(pixels) - min(pixels) > 4:
                    image = candidate
                    break
            except queue.Empty:
                continue
        if image is None:
            raise RuntimeError('No nonempty rendered RGB frame received')
        image_path = probe / 'render_probe.png'
        image.save_to_disk(str(image_path))
        gpu_contexts = process_gpu_contexts(owner.process.pid)
        save(probe / 'gpu_contexts.json', {'expected_uuid': args.gpu_uuid, 'contexts': gpu_contexts})
        rendering = verify_render_contexts(gpu_contexts, args.gpu_uuid)
        graphics_gpus = [row['uuid'] for row in gpu_contexts]
        record = {'ready': True, 'unix': time.time(), 'job_id': os.environ['SLURM_JOB_ID'],
                  'host': socket.gethostname(), 'gpu_uuid': args.gpu_uuid,
                  'graphics_adapter': owner.vulkan_adapter, 'server_pid': owner.process.pid,
                  'private_runtime_root': str(owner.runtime_root),
                  'version': client.get_server_version(), 'loaded_map': world.get_map().name,
                  'rgb_frame': image.frame, 'rgb_size': [image.width, image.height],
                  'rgb_png_sha256': sha256(image_path), 'graphics_process_gpus': graphics_gpus,
                  'gpu_contexts': gpu_contexts, 'render_gpu_uuid': rendering['uuid'],
                  'auxiliary_graphics_only_context_max_mib': 32,
                  'vulkan_environment': vulkan_environment(),
                  'scope': 'Observed live GPU rendering, RPC, map and sensor readiness in the Slurm job'}
        save(probe / 'ready.json', record)
        with (args.carla_root / 'readiness.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = args.carla_root / 'runtime_ready.json'
            previous = json.loads(path.read_text()) if path.exists() else {}
            records = previous.get('probes', [])
            records.append(record)
            save(path, {'version': '0.9.15', 'binary_build_id': '12cdff35da2c8067',
                        'generated_from_live_probes': True, 'probes': records})
        print('RUNTIME_READY ' + json.dumps(record), flush=True)
    finally:
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError:
                pass
        if vehicle is not None:
            try:
                vehicle.destroy()
            except RuntimeError:
                pass
        if world is not None and old_settings is not None:
            try:
                world.apply_settings(old_settings)
            except RuntimeError:
                pass
        if owner is not None:
            owner.close()


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=50)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def release_claims(args, worker_prefix, reason):
    route_queue = RouteQueue(args.run_root)
    state = route_queue.snapshot()
    for route, item in state['items'].items():
        if item['state'] == 'claimed' and item.get('owner', '').startswith(worker_prefix):
            route_queue.finish(route, item['owner'], {
                'route_id': route, 'classification': 'infrastructure_error',
                'supervisor_reason': reason, 'inference_steps': args.inference_steps,
                'unix': time.time()})


def group_run(args):
    verify_slurm_job(os.environ.get('SLURM_JOB_ID', ''), expected_gpus=8, expected_cpus=56)
    check_sources(args)
    if cuda_visible_uuids() != [args.gpu_uuid]:
        raise RuntimeError('Group process does not see exactly its allocated GPU')
    group = args.run_root / 'groups' / args.group_id
    group.mkdir(parents=True, exist_ok=False)
    state = RouteQueue(args.run_root).snapshot()
    save(group / 'manifest.json', {'identity': state['identity'], 'unix': time.time(),
        'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(), 'gpu_uuid': args.gpu_uuid,
        'logical_gpu': args.logical_gpu, 'graphics_adapter': args.graphics_adapter,
        'lanes': args.lanes, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'scheduler': 'Slurm', 'server_mode': 'warm'})
    children = []
    streams = []
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    lock_root = args.run_root / 'gpu_locks'
    lock_root.mkdir(exist_ok=True)
    try:
        with (lock_root / f'{args.gpu_uuid}.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runtime_probe(args, group)
            for lane in range(args.lanes):
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                    '--role', 'lane', '--run-root', str(args.run_root), '--checkpoint', str(args.checkpoint),
                    '--carla-root', str(args.carla_root), '--inference-steps', str(args.inference_steps),
                    '--logical-gpu', str(args.logical_gpu), '--graphics-adapter', str(args.graphics_adapter),
                    '--gpu-uuid', args.gpu_uuid, '--group-id', args.group_id,
                    '--lanes', str(args.lanes), '--lane', str(lane), '--seed', str(args.seed)]
                stream = (group / f'lane{lane}.log').open('wb')
                streams.append(stream)
                children.append(subprocess.Popen(command, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True))
            while any(child.poll() is None for child in children):
                if (args.run_root / 'fatal.json').exists():
                    raise RuntimeError('Evaluation stopped after a recorded fatal error')
                failed = [child.returncode for child in children if child.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Route lane exited unsuccessfully: {failed}')
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError('A route lane failed')
            save(group / 'completed.json', {'unix': time.time(), 'returncode': 0})
    except BaseException as exc:
        save(group / 'failure.json', {'unix': time.time(), 'error': repr(exc)})
        raise
    finally:
        for child in children:
            stop_owned(child)
        release_claims(args, args.group_id + '/', 'group_controller_exited')
        for stream in streams:
            stream.close()


def cpu_assignment(allowed, gpu_rows):
    remaining = set(allowed)
    result = []
    for gpu in gpu_rows:
        domain, bus, slot = gpu['pci_bus_id'].lower().split(':')
        numa_file = Path(f'/sys/bus/pci/devices/{domain[-4:]}:{bus}:{slot}/numa_node')
        local = []
        if numa_file.is_file() and int(numa_file.read_text()) >= 0:
            cpulist = Path('/sys/devices/system/node') / f'node{int(numa_file.read_text())}' / 'cpulist'
            for span in cpulist.read_text().strip().split(','):
                limits = list(map(int, span.split('-')))
                local.extend(range(limits[0], limits[-1] + 1))
        selected = sorted(remaining.intersection(local))[:7]
        selected.extend(sorted(remaining.difference(selected))[:7-len(selected)])
        if len(selected) != 7:
            raise RuntimeError('Cannot assign seven disjoint CPUs to each GPU')
        result.append(selected)
        remaining.difference_update(selected)
    return result


def node_run(args):
    proof = verify_slurm_job(os.environ.get('SLURM_JOB_ID', ''), expected_gpus=8, expected_cpus=56)
    if Path(sys.prefix).resolve() != (ROOT / '.venvs/full-diffusion-carla').resolve():
        raise RuntimeError('Use the prepared project virtual environment')
    check_sources(args)
    allocation_affinity = sorted(os.sched_getaffinity(0))
    physical = cpu_groups(allocation_affinity)
    allowed = [core[level] for level in range(max(map(len, physical), default=0))
               for core in physical if len(core) > level][:56]
    if len(allowed) != 56:
        raise RuntimeError(f'Fewer than the 56 allocated CPUs are accessible: {allocation_affinity}')
    os.sched_setaffinity(0, allowed)
    visible = cuda_visible_uuids()
    if len(visible) != 8 or len(set(visible)) != 8:
        raise RuntimeError(f'Expected eight unique allocation-visible GPUs, got {visible}')
    inventory = {row['uuid']: row for row in gpu_inventory()}
    gpus = [inventory[value] for value in visible]
    if any('L40' not in gpu['name'] or gpu['memory_used_mib'] >= 1024 for gpu in gpus):
        raise RuntimeError('Allocated GPUs must be idle L40 devices before launch')
    vulkan_env = {**os.environ, 'NODEVICE_SELECT': '1'}
    devices = vulkan_gpu_map(vulkan_env)
    if set(devices) != set(visible):
        raise RuntimeError('Vulkan device UUIDs differ from the eight allocated CUDA GPUs')
    device_record = {'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
                     'unix': time.time(), 'devices': devices, 'nodevice_select': '1'}
    save(args.run_root / 'vulkan_devices.json', device_record)
    os.environ['WMMPC_VULKAN_GPU_MAP'] = json.dumps(device_record)
    cpus = cpu_assignment(allowed, gpus)
    save(args.run_root / 'slurm_allocation.json', {'proof': proof, 'gpus': gpus, 'cpu_groups': cpus,
         'visible_uuids': visible, 'unix': time.time(), 'pid': os.getpid(), 'memory': memory_snapshot(),
         'step_id': os.environ.get('SLURM_STEP_ID'),
         'allocation_affinity': allocation_affinity})
    streams = []
    children = []
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    began = time.time()
    last_report = 0
    try:
        for index, gpu in enumerate(gpus):
            group_id = f'{os.environ["SLURM_JOB_ID"]}_gpu{index}'
            command = ['taskset', '-c', ','.join(map(str, cpus[index])), sys.executable, '-u',
                str(Path(__file__).resolve()), '--role', 'group', '--run-root', str(args.run_root),
                '--checkpoint', str(args.checkpoint), '--carla-root', str(args.carla_root),
                '--inference-steps', str(args.inference_steps), '--lanes', str(args.lanes),
                '--logical-gpu', str(index), '--graphics-adapter', str(gpu['index']),
                '--gpu-uuid', gpu['uuid'], '--group-id', group_id, '--seed', str(args.seed)]
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=gpu['uuid'])
            stream = (args.run_root / f'gpu{index}.controller.log').open('wb')
            streams.append(stream)
            children.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'GPU controller failed: {failed}')
            now = time.time()
            snapshot = RouteQueue(args.run_root).snapshot()
            status = {'unix': now, 'elapsed_seconds': now-began, 'job_id': os.environ['SLURM_JOB_ID'],
                      'states': dict(Counter(item['state'] for item in snapshot['items'].values())),
                      'gpu': gpu_inventory(), 'memory': memory_snapshot()}
            with (args.run_root / 'resources.jsonl').open('a') as handle:
                handle.write(json.dumps(status) + '\n')
            save(args.run_root / 'status.json', status)
            print('V3_PROGRESS ' + json.dumps(status), flush=True)
            if now-last_report >= 60:
                from evaluation.summarize_parallel_carla import summarize
                save(args.run_root / 'summary.json', summarize(args.run_root))
                last_report = now
            time.sleep(15)
        if any(child.returncode for child in children):
            raise RuntimeError('A GPU controller exited unsuccessfully')
        from evaluation.summarize_parallel_carla import summarize
        report = summarize(args.run_root, extended=True, merge=True)
        save(args.run_root / 'summary.json', report)
        if not report['complete_220_routes']:
            raise RuntimeError('Evaluation ended with incomplete routes; inspect the queue and reports')
        save(args.run_root / 'completed.json', {'unix': time.time(), 'job_id': os.environ['SLURM_JOB_ID'],
             'inference_steps': args.inference_steps, 'complete_220_routes': True,
             'metrics': report['metrics_full_benchmark']})
        print('V3_COMPLETE ' + json.dumps(report['metrics_full_benchmark']), flush=True)
    except BaseException as exc:
        save(args.run_root / 'fatal.json', {'unix': time.time(), 'error': repr(exc)})
        raise
    finally:
        for child in children:
            stop_owned(child)
        from evaluation.run_full_diffusion_carla import stop_server
        for owner in (args.run_root / 'groups').rglob('server_process.json'):
            stop_server(owner.parent, os.environ['SLURM_JOB_ID'])
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('prepare', 'node', 'group', 'lane'), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path, required=True)
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--logical-gpu', type=int, default=0)
    parser.add_argument('--graphics-adapter', type=int, default=0)
    parser.add_argument('--gpu-uuid', default='')
    parser.add_argument('--group-id', default='')
    parser.add_argument('--lane', type=int, default=0)
    args = parser.parse_args()
    args.run_root = checked_run_root(args.run_root)
    args.checkpoint = checked_evaluation_path(args.checkpoint)
    args.carla_root = checked_evaluation_path(args.carla_root)
    if args.inference_steps != 25 or args.lanes != 2:
        raise ValueError('This run requires 25 inference steps and two lanes per GPU')
    os.environ.update(WMMPC_DIFFUSION_INFERENCE_STEPS=str(args.inference_steps), WMMPC_BUFFER_METRICS='1')
    vulkan = vulkan_environment()
    if not vulkan:
        os.environ.pop('VK_ICD_FILENAMES', None)
    os.environ.update(vulkan)
    if args.role == 'prepare':
        prepare(args)
    elif args.role == 'node':
        node_run(args)
    elif args.role == 'group':
        group_run(args)
    else:
        verify_slurm_job(os.environ.get('SLURM_JOB_ID', ''))
        from evaluation.run_full_diffusion2_carla import lane
        lane(lane_arguments(args))


if __name__ == '__main__':
    main()
