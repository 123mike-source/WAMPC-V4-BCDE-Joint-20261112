"""Measure six-point normalization on the retained training split only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.full_diffusionv4.data import prepare_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'method/full_diffusionv4/config.json')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path, help='Relocated raw dataset; defaults to config data_root')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = prepare_normalization(args.index or ROOT / config['index_directory'],
                                   ROOT / config['backbone_directory'], config['labels'],
                                   args.output or ROOT / config['normalization_file'],
                                   args.data_root or config['data_root'])
    print(json.dumps({'method': result['method'], 'train_frames': result['train_frames'],
                      'valid_slot_counts': result['valid_slot_counts']}))


if __name__ == '__main__':
    main()
