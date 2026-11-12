"""Verify published file bytes and the training-start code identities."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Also require the original server paths configuration')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/publication_manifest.json').read_text())
    skipped = []
    for entry in manifest['files']:
        if entry['path'] == 'configs/paths.json' and not args.full:
            skipped.append(entry['path'])
            continue
        path = ROOT / entry['path']
        if not path.is_file() or path.is_symlink() or sha(path) != entry['sha256']:
            raise ValueError('Snapshot file differs: ' + entry['path'])
    run = ROOT / 'runs/tongji_bcde_joint_seed901_20261112'
    startup = json.loads((run / 'startup.json').read_text())
    for name, expected in startup['code_sha256'].items():
        if sha(ROOT / 'method/full_diffusionv4' / name) != expected:
            raise ValueError('Training-start source differs: ' + name)
    config = json.loads((run / 'config.json').read_text())
    if config != json.loads((ROOT / 'method/full_diffusionv4/config_joint.json').read_text()):
        raise ValueError('Formal recipe differs from archived run configuration')
    completed = json.loads((run / 'completed.json').read_text())
    validations = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    if completed['updates'] != 2180 or completed['epoch'] != 10 or len(validations) != 10:
        raise ValueError('Training completion records differ')
    print(json.dumps({'status': 'PASS', 'files_verified': len(manifest['files']) - len(skipped),
                      'training_start_code_hashes': len(startup['code_sha256']),
                      'local_path_overrides_skipped': skipped,
                      'carla_garage_submodule': manifest['submodules']['third_party/carla_garage']}, indent=2))


if __name__ == '__main__':
    main()
