"""Prepare index, fill/verify frozen cache, then train in an existing Slurm step."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .prepare import ROOT, write_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('An existing Slurm allocation is required')
    run = args.run_dir.resolve()
    run.relative_to(ROOT / '.runtime')
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_file = run / 'config.json'
    config = json.loads(config_file.read_text())
    launch = json.loads((run / 'launch.json').read_text())
    if str(os.environ['SLURM_JOB_ID']) != str(launch['job']):
        raise ValueError('The Slurm allocation differs from the requested job')
    import torch
    if torch.cuda.device_count() != 8 or any('L40' not in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError('This run requires the existing eight L40 GPUs')
    index, cache = ROOT / config['index'], ROOT / config['cache']
    output = Path(launch['output']).resolve()
    output.relative_to(ROOT / 'ckps/irwm_bev_t_large')
    state = {'job': launch['job'], 'host': socket.gethostname(), 'pid': os.getpid(),
             'run_dir': str(run), 'output': str(output), 'started_unix': time.time(),
             'config_sha256': sha(config_file), 'steps': []}

    def phase(name, command):
        state.update(phase=name, status='running', updated_unix=time.time())
        write_json(run / 'pipeline.json', state)
        print(json.dumps({'event': 'phase_start', 'phase': name, 'command': command}), flush=True)
        with (run / f'{name}.log').open('a') as log:
            log.write('\n' + json.dumps({'command': command, 'started_unix': time.time()}) + '\n')
            log.flush()
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['steps'].append({'phase': name, 'completed_unix': time.time()})
        write_json(run / 'pipeline.json', state)

    python = sys.executable
    module = 'method.irwm_bev_t_large.'
    try:
        phase('prepare', [python, '-u', '-m', module + 'prepare', '--data-root', launch['data_root'],
                          '--output', str(index), '--workers', '24'])
        common = ['--index', str(index), '--cache', str(cache), '--output', str(run / 'cache')]
        phase('cache_scan', [python, '-u', '-m', module + 'cache_fill', *common, '--scan'])
        scan = json.loads((run / 'cache/scan.json').read_text())
        if scan['missing_frames']:
            phase('cache_fill', [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                                 '-m', module + 'cache_fill', *common, '--workers', '4'])
        phase('cache_verify', [python, '-u', '-m', module + 'cache_fill', *common, '--verify'])
        command = [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '-m', module + 'train', '--config', str(config_file), '--output', str(output)]
        if (output / 'latest.pt').exists():
            command.append('--resume')
        phase('train', command)
        state.update(status='complete' if (output / 'completed.json').exists() else 'stopped_with_checkpoint',
                     updated_unix=time.time())
        write_json(run / 'pipeline.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error), updated_unix=time.time())
        write_json(run / 'pipeline.json', state)
        raise


if __name__ == '__main__':
    main()
