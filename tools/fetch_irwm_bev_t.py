"""Fetch a user-specified trained irwm_bev_t checkpoint and record provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='HTTPS URL or accessible checkpoint file')
    parser.add_argument('--output', type=Path, default=ROOT / 'ckps/irwm_bev_t/release/best.pt')
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace an existing weight file: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(f'Inspect the prior incomplete download first: {temporary}')
    if args.source.startswith('https://'):
        request = urllib.request.Request(args.source, headers={'User-Agent': 'WAMPC-irwm-weight-fetch'})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open('xb') as output:
            shutil.copyfileobj(source, output, length=8 << 20)
    else:
        source = Path(args.source)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open('rb') as source_file, temporary.open('xb') as output:
            shutil.copyfileobj(source_file, output, length=8 << 20)
    digest = hashlib.sha256()
    with temporary.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if args.sha256 and value != args.sha256:
        raise ValueError(f'Download checksum mismatch; incomplete file retained at {temporary}')
    import torch
    from method.full_diffusionv4.world import timed_model_class
    state = torch.load(temporary, map_location='cpu', weights_only=False)
    if state.get('synthetic_interface_test_only'):
        raise ValueError('Synthetic interface-test weights are not a trained IRWM checkpoint')
    model_class = timed_model_class(state['config'].get('method'), state['config'].get('architecture'))
    model = model_class(**state['config']['model'])
    model.load_state_dict(state['model'], strict=True)
    if state['dataset'].get('query_times_seconds') != [.5, 1., 1.5, 2., 2.5, 3.]:
        raise ValueError('Downloaded world model uses an incompatible future grid')
    if not state['dataset'].get('teacher_sha256'):
        raise ValueError('World checkpoint is missing frozen-teacher provenance')
    temporary.replace(args.output)
    manifest = {'source': args.source, 'path': str(args.output.resolve()), 'bytes': args.output.stat().st_size,
                'sha256': value, 'teacher_sha256': state['dataset']['teacher_sha256'],
                'progress': state.get('progress'), 'metrics': state.get('metrics'),
                'fetched_utc': datetime.now(timezone.utc).isoformat(),
                'method': state['config']['method'], 'architecture': state['config'].get('architecture')}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
