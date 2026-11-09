"""Generate a deterministic 16,384-record binary INRIA-format smoke fixture."""
import argparse
from pathlib import Path
import struct
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error('Output exists')
args.output.parent.mkdir(parents=True, exist_ok=True)
properties = ['x', 'y', 'z', 'nx', 'ny', 'nz']
properties += ['f_dc_' + str(i) for i in range(3)]
properties += ['f_rest_' + str(i) for i in range(45)]
properties += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
header = 'ply\nformat binary_little_endian 1.0\nelement vertex 16384\n'
header += ''.join('property float ' + p + '\n' for p in properties) + 'end_header\n'
rng = random.Random(42)
with args.output.open('wb') as handle:
    handle.write(header.encode('ascii'))
    for i in range(16384):
        row = [(i % 128) / 32 - 2, (i // 128) / 32 - 2, rng.uniform(-0.5, 0.5), 0, 0, 0]
        row += [rng.uniform(-1, 1) for _ in range(3)]
        row += [rng.uniform(-0.2, 0.2) for _ in range(45)]
        row += [rng.uniform(-1, 3), -4, -4, -4, 1, 0, 0, 0]
        handle.write(struct.pack('<62f', *row))
print(args.output)
