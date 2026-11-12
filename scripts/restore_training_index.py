"""Restore the archived source index and generate local path configuration."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/dataset_index/source'
INDEX = ROOT / 'data/irwm_bev_t/pdm_seed901_dev44_dt05_tongji_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def immutable_copy(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing index differs; use a fresh data directory: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path, required=True)
    args = parser.parse_args()
    data_root, carla_root = args.data_root.resolve(), args.carla_root.resolve()
    if not data_root.is_dir() or not carla_root.is_dir():
        raise ValueError('Both --data-root and --carla-root must be existing directories')
    provenance = json.loads((ROOT / 'provenance/source_snapshot.json').read_text())['source_files']
    manifest = None
    for source in sorted(SOURCE.iterdir()):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f'Unexpected source entry: {source}')
        data = source.read_bytes()
        relative = source.relative_to(ROOT).as_posix()
        if digest(data) != provenance[relative]['sha256']:
            raise ValueError(f'Archived source checksum differs: {relative}')
        if source.name == 'manifest.json':
            manifest = json.loads(data)
            manifest['data_root'] = str(data_root)
            data = (json.dumps(manifest, indent=2) + '\n').encode()
        immutable_copy(INDEX / source.name, data)

    hashes = json.loads((ROOT / 'artifacts/dataset_index/source_metadata_sha256.json').read_text())
    archive = ROOT / 'artifacts/dataset_index/source_metadata.zip'
    if digest(archive.read_bytes()) != provenance[archive.relative_to(ROOT).as_posix()]['sha256']:
        raise ValueError('Archived route metadata checksum differs')
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        expected = {'metadata/' + name for name in hashes}
        if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
            raise ValueError('Route metadata archive entries differ')
        if len(entries) != manifest['routes']:
            raise ValueError('Route metadata count differs from source manifest')
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if not re.fullmatch(r'metadata/[0-9]{5}\.npz', entry.filename):
                raise ValueError('Unexpected route metadata path')
            data = z.read(entry)
            if digest(data) != hashes[path.name]:
                raise ValueError('Route metadata checksum differs: ' + entry.filename)
            immutable_copy(INDEX.joinpath(*path.parts), data)

    config = json.loads((ROOT / 'runs/tongji_bcde_joint_seed901_20261112/config.json').read_text())
    config['data_root'] = str(data_root)
    config_path = ROOT / 'configs/local_joint.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    paths_file = ROOT / 'configs/paths.json'
    paths = json.loads(paths_file.read_text())
    paths.update(carla_root=str(carla_root), garage_root=str(data_root), results_root=str(ROOT / 'results'))
    paths_file.write_text(json.dumps(paths, indent=2) + '\n', encoding='utf-8')
    report = {'source_index': str(INDEX), 'metadata_routes': len(hashes),
              'data_root': str(data_root), 'config': str(config_path),
              'prepared_labels_and_feature_caches': 'must be generated separately',
              'original_artifacts_modified': False}
    (INDEX / 'restoration.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
