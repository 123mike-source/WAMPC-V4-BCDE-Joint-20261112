"""Slurm allocation checks and Linux resource inspection for Tongji CARLA runs."""
import csv
import ctypes
import ctypes.util
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import time
import uuid

_proof_cache = {}


def parse_slurm_fields(text):
    return dict(token.split('=', 1) for token in text.split() if '=' in token)


def has_slurm_ancestor(proc_root=Path('/proc')):
    pid = os.getpid()
    for _ in range(64):
        try:
            base = proc_root / str(pid)
            if (base / 'comm').read_text().strip().startswith('slurmstepd'):
                return True
            status = (base / 'status').read_text()
            parent = int(re.search(r'^PPid:\s*(\d+)', status, re.M)[1])
        except (OSError, TypeError):
            return False
        if parent <= 1 or parent == pid:
            return False
        pid = parent
    return False


def verify_slurm_job(job_id, expected_gpus=None, expected_cpus=None):
    """Require an owned RUNNING job, its assigned host and a Slurm process tree."""
    job_id = str(job_id)
    if not job_id.isdigit() or os.environ.get('SLURM_JOB_ID') != job_id:
        raise ValueError('Run this command inside the designated Slurm allocation')
    key = (os.getpid(), job_id)
    cached = _proof_cache.get(key)
    if cached and time.monotonic() - cached[0] < 20:
        proof = cached[1]
    else:
        result = subprocess.run(['scontrol', 'show', 'job', '-o', job_id],
                                text=True, capture_output=True, timeout=20, check=True)
        proof = parse_slurm_fields(result.stdout)
        if proof.get('JobState') != 'RUNNING':
            raise ValueError(f'Slurm job is not RUNNING: {proof.get("JobState")}')
        owner = re.fullmatch(r'([^()]+)\((\d+)\)', proof.get('UserId', ''))
        if not owner or int(owner[2]) != os.getuid() or owner[1] != pwd.getpwuid(os.getuid()).pw_name:
            raise ValueError('Slurm job is not owned by the current user')
        hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', proof['NodeList']],
                                        text=True, timeout=15).splitlines()
        if socket.gethostname().split('.')[0] not in [h.split('.')[0] for h in hosts]:
            raise ValueError('Current host is outside the Slurm allocation')
        cgroup = Path('/proc/self/cgroup').read_text()
        match = re.search(rf'(?<!\d){re.escape(job_id)}(?!\d)', cgroup)
        if re.search(r'job[_/-]\d+', cgroup) and not match:
            raise ValueError('Current cgroup belongs to another Slurm job')
        if not match and not has_slurm_ancestor():
            raise ValueError('Cannot verify that this process was launched by Slurm')
        proof = {**proof, 'verified_host': socket.gethostname(), 'verified_uid': os.getuid(),
                 'cgroup_contains_job': bool(match), 'checked_unix': time.time()}
        _proof_cache[key] = (time.monotonic(), proof)
    tres = parse_tres(proof.get('AllocTRES') or proof.get('TRES', ''))
    gpu_count = int(tres.get('gres/gpu', proof.get('NumGPUs', '0')))
    if expected_gpus is not None and gpu_count != expected_gpus:
        raise ValueError(f'Expected {expected_gpus} allocated GPUs, got {gpu_count}')
    if expected_cpus is not None and int(proof.get('NumCPUs', '0')) != expected_cpus:
        raise ValueError(f'Expected {expected_cpus} allocated CPUs, got {proof.get("NumCPUs")}')
    return proof


def parse_tres(value):
    return dict(item.split('=', 1) for item in value.split(',') if '=' in item)


def cgroup_memory_snapshot(cgroup_file=Path('/proc/self/cgroup'), root=Path('/sys/fs/cgroup')):
    """Read actual v1/v2 limits; report 'max' when no finite limit is present."""
    entries = [line.split(':', 2) for line in cgroup_file.read_text().splitlines()]
    v1 = next((e for e in entries if 'memory' in e[1].split(',')), None)
    v2 = next((e for e in entries if e[1] == ''), None)
    if v2 and v1 and not (root / v2[2].lstrip('/') / 'memory.current').exists():
        v2 = None
    entry = v2 or v1
    if entry is None:
        return {'job_memory.max': 'max', 'job_memory.current': '0',
                'job_memory.working_set': '0', 'cgroup': None, 'memory_controller': 'unavailable'}
    base = root if v2 else root / 'memory'
    leaf = base / entry[2].lstrip('/')
    limit_name = 'memory.max' if v2 else 'memory.limit_in_bytes'
    usage_name = 'memory.current' if v2 else 'memory.usage_in_bytes'
    candidates = []
    current = leaf
    while current.is_relative_to(base):
        path = current / limit_name
        if path.is_file():
            value = path.read_text().strip()
            if value != 'max' and 0 < int(value) < 2**60:
                candidates.append((int(value), current))
        if current == base:
            break
        current = current.parent
    limit, group = min(candidates, key=lambda pair: pair[0]) if candidates else (None, leaf)
    usage = group / usage_name
    used = int(usage.read_text()) if usage.is_file() else 0
    stats_path = group / 'memory.stat'
    stats = dict((key, int(value)) for key, value in
                 (line.split() for line in stats_path.read_text().splitlines())) if stats_path.is_file() else {}
    inactive = (stats.get('inactive_file', 0) if v2 else
                stats.get('total_inactive_file', stats.get('inactive_file', 0)))
    anon = stats.get('anon', stats.get('total_rss', stats.get('rss', 0)))
    file_bytes = stats.get('file', stats.get('total_cache', stats.get('cache', 0)))
    events = group / ('memory.events' if v2 else 'memory.failcnt')
    return {'job_memory.max': str(limit) if limit is not None else 'max',
            'job_memory.current': str(used), 'job_memory.working_set': str(max(0, used-inactive)),
            'job_memory.noncache': str(anon + sum(stats.get(k, 0) for k in ('kernel', 'shmem', 'sock'))),
            'job_memory.anon': anon, 'job_memory.file': file_bytes,
            'job_memory.inactive_file': inactive,
            'job_memory.events': events.read_text().strip() if events.is_file() else '',
            'cgroup': str(group), 'memory_controller': 'v2' if v2 else 'v1'}


def vulkan_environment():
    existing = os.environ.get('VK_ICD_FILENAMES')
    if existing and all(Path(p).is_file() for p in existing.split(':')):
        return {'VK_ICD_FILENAMES': existing}
    candidates = [Path('/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json'),
                  Path('/usr/share/vulkan/icd.d/nvidia_icd.json'),
                  Path('/etc/vulkan/icd.d/nvidia_icd.json')]
    for directory in ('/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d'):
        candidates.extend(sorted(Path(directory).glob('*nvidia*.json')))
    for candidate in candidates:
        if candidate.is_file():
            return {'VK_ICD_FILENAMES': str(candidate)}
    return {}


def vulkan_gpu_map(environment=None):
    """Query once before starting CARLA; concurrent driver probes can serialize."""
    output = subprocess.check_output(['vulkaninfo', '--summary'], text=True,
                                     stderr=subprocess.PIPE, timeout=120, env=environment)
    devices = {}
    for block in re.split(r'(?m)^GPU(?=\d+:)', output)[1:]:
        index = re.match(r'(\d+):', block)
        device_uuid = re.search(r'deviceUUID\s*=\s*([0-9a-fA-F-]{36})', block)
        if index and device_uuid:
            key = 'GPU-' + device_uuid[1].lower()
            if key in devices:
                raise RuntimeError('Duplicate UUID in Vulkan enumeration')
            devices[key] = int(index[1])
    if not devices or len(set(devices.values())) != len(devices):
        raise RuntimeError('Vulkan enumeration has no unambiguous UUID mapping')
    return devices


def vulkan_gpu_index(gpu_uuid, environment=None):
    """Use this allocation's live map, avoiding repeated driver probes under load."""
    import json
    environment = os.environ if environment is None else environment
    cached = environment.get('WMMPC_VULKAN_GPU_MAP')
    if cached:
        record = json.loads(cached)
        if (record['job_id'] != environment.get('SLURM_JOB_ID')
                or record['host'] != socket.gethostname()):
            raise RuntimeError('Vulkan device map belongs to a different allocation or host')
        devices = record['devices']
    else:
        devices = vulkan_gpu_map(environment)
    normalized = 'GPU-' + gpu_uuid.removeprefix('GPU-').lower()
    if normalized not in devices or not isinstance(devices[normalized], int) or devices[normalized] < 0:
        raise RuntimeError(f'Cannot map {gpu_uuid} to a Vulkan physical device')
    return devices[normalized]


def cuda_visible_uuids():
    """Query CUDA's actual visible ordinals without creating model contexts."""
    driver = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
    def checked(code):
        if code:
            raise RuntimeError(f'CUDA driver query failed with code {code}')
    checked(driver.cuInit(0))
    count = ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    result = []
    for index in range(count.value):
        device = ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device), index))
        value = (ctypes.c_ubyte * 16)()
        checked(driver.cuDeviceGetUuid(ctypes.byref(value), device))
        result.append('GPU-' + str(uuid.UUID(bytes=bytes(value))))
    return result


def gpu_inventory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return [{'index': int(row[0].strip()), 'uuid': row[1].strip(), 'name': row[2].strip(),
             'pci_bus_id': row[3].strip(), 'memory_total_mib': int(row[4]),
             'memory_used_mib': int(row[5]), 'utilization_percent': int(row[6])}
            for row in csv.reader(output.splitlines())]


def process_graphics_gpus(pid):
    return [row['uuid'] for row in process_gpu_contexts(pid)]


def process_gpu_contexts(pid):
    import xml.etree.ElementTree as ET
    output = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=20)
    contexts = []
    for index, gpu in enumerate(ET.fromstring(output).findall('gpu')):
        for process in gpu.findall('processes/process_info'):
            if int(process.findtext('pid', '-1')) != pid:
                continue
            used = process.findtext('used_memory', '')
            memory = re.fullmatch(r'(\d+)\s*MiB', used)
            contexts.append({'uuid': gpu.findtext('uuid'), 'nvidia_smi_index': index,
                             'type': process.findtext('type', ''), 'pid': pid,
                             'memory_used_mib': int(memory[1]) if memory else None})
    return contexts


def verify_render_contexts(contexts, expected_uuid):
    """Require real graphics load on the requested GPU; retain tiny driver contexts.

    Driver 550.54.14 creates a 6 MiB graphics-only context on GPU0 even when the
    actual CARLA Vulkan renderer and CUDA context use a different selected GPU.
    A frame and the selected Vulkan index are checked separately by runtime_probe.
    """
    target = [row for row in contexts if row['uuid'] == expected_uuid]
    if (len(target) != 1 or 'G' not in target[0]['type']
            or target[0]['memory_used_mib'] is None or target[0]['memory_used_mib'] < 128):
        raise RuntimeError(f'No substantial CARLA graphics context on {expected_uuid}: {contexts}')
    for row in contexts:
        if row['uuid'] == expected_uuid:
            continue
        if row['type'] != 'G' or row['memory_used_mib'] is None or row['memory_used_mib'] > 32:
            raise RuntimeError(f'Unexpected CARLA work on a different GPU: {contexts}')
    return target[0]
