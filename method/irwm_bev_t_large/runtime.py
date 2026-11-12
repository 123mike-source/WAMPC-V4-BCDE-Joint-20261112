"""Keep worker sockets and temporary files on the compute node, away from NFS."""
import os
from pathlib import Path
import tempfile

_temporary = None


def configure_worker_tmp():
    global _temporary
    if _temporary is None:
        base = Path('/tmp') if os.name == 'posix' else Path(tempfile.gettempdir())
        _temporary = tempfile.TemporaryDirectory(prefix='wampc-irwm-', dir=base)
        os.environ['TMPDIR'] = _temporary.name
        tempfile.tempdir = _temporary.name
    return Path(_temporary.name)
