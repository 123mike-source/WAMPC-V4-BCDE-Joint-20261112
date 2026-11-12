"""Separate writable Unreal state and bound concurrent driver initialization."""
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import time

from evaluation.carla_eval_resources import checked_evaluation_path, evaluation_runtime_root

MUTABLE = {'Saved', 'Intermediate', 'DerivedDataCache', 'Logs'}
STARTUP_CONCURRENCY = 2


def private_lane_runtime(source, target):
    source = checked_evaluation_path(source)
    target = checked_evaluation_path(target)
    record_path = target / 'runtime_reference.json'
    if target.exists():
        if json.loads(record_path.read_text())['source'] != str(source):
            raise ValueError('Lane runtime points to a different CARLA installation')
        return target
    target.mkdir(parents=True)
    binary_storage = None
    for item in source.iterdir():
        out = target / item.name
        if item.name in ('runtime_source.json', 'runtime_ready.json', 'readiness.lock'):
            continue
        if item.name in ('Engine', 'CarlaUE4'):
            out.mkdir()
            for child in item.iterdir():
                destination = out / child.name
                if child.name == 'Config':
                    shutil.copytree(child, destination)
                elif child.name in MUTABLE:
                    destination.mkdir()
                elif item.name == 'CarlaUE4' and child.name == 'Binaries':
                    destination.mkdir()
                    for platform in child.iterdir():
                        platform_out = destination / platform.name
                        if platform.name != 'Linux':
                            platform_out.symlink_to(platform.resolve())
                            continue
                        platform_out.mkdir()
                        for binary in platform.iterdir():
                            binary_out = platform_out / binary.name
                            if binary.name != 'CarlaUE4-Linux-Shipping':
                                binary_out.symlink_to(binary.resolve())
                                continue
                            try:
                                os.link(binary, binary_out)
                                binary_storage = 'hardlink to the private verified executable'
                            except OSError as error:
                                if error.errno not in (errno.EXDEV, errno.EPERM):
                                    raise
                                shutil.copy2(binary, binary_out)
                                binary_storage = 'copy of the private verified executable'
                else:
                    destination.symlink_to(child.resolve())
            for name in MUTABLE:
                (out / name).mkdir(exist_ok=True)
        elif item.name in MUTABLE:
            out.mkdir()
        else:
            out.symlink_to(item.resolve())
    record_path.write_text(json.dumps({'source': str(source), 'runtime': str(target),
        'binary_storage': binary_storage, 'private_writable_directories': sorted(MUTABLE),
        'unix': time.time()}, indent=2) + '\n')
    return target


@contextmanager
def startup_slot(job_id, directory):
    if not str(job_id).isdigit():
        raise ValueError('A numeric allocation ID is required')
    locks = evaluation_runtime_root() / 'carla_startup_locks' / socket.gethostname() / str(job_id)
    locks.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    lease = None
    while lease is None:
        for index in range(STARTUP_CONCURRENCY):
            handle = (locks / f'slot{index}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lease = handle
                break
            except BlockingIOError:
                handle.close()
        if lease is None:
            if time.monotonic() - began > 1200:
                raise TimeoutError('Timed out waiting for a CARLA driver initialization slot')
            time.sleep(.2)
    try:
        (directory / 'startup_slot.json').write_text(json.dumps({
            'job_id': str(job_id), 'slot': index, 'pid': os.getpid(),
            'maximum_concurrent_initializations': STARTUP_CONCURRENCY,
            'wait_seconds': time.monotonic() - began, 'unix': time.time()}, indent=2) + '\n')
        yield
    finally:
        lease.close()
