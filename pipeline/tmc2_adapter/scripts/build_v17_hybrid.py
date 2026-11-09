#!/usr/bin/env python3
"""Build a V-PCC geometry + Morton-ordered SOG semantic payload.

The geometry transform is taken from the V-PCC manifest.  This supports both
the original uniform grid and V18's SOG-compatible signed-log1p grid.
"""
from __future__ import annotations
import argparse, json, shutil, zlib
from pathlib import Path
import numpy as np

def layout(path):
    with path.open('rb') as f:
        h=bytearray()
        while not h.endswith(b'end_header\n'): h.extend(f.read(1))
    lines=h.decode().splitlines(); n=0; props=[]; inside=False
    for line in lines:
        p=line.split()
        if p[:2]==['element','vertex']: n=int(p[2]); inside=True
        elif p[:1]==['element']: inside=False
        elif inside and p[:2]==['property','float']: props.append(p[2])
    return len(h),n,len(props)*4

def morton(q,bits):
    out=np.zeros(len(q),np.uint64)
    for b in range(bits):
        for a in range(3): out|=((q[:,a]>>b)&1).astype(np.uint64) << (3*b+a)
    return out

def xyz(path):
    off,n,stride=layout(path); dt=np.dtype({'names':['xyz'],'formats':[('<f4',(3,))],'offsets':[0],'itemsize':stride})
    return np.memmap(path,dtype=dt,mode='r',offset=off,shape=(n,))['xyz']

def quantize(x, transform):
    if transform.get('signed_log_coordinates', False):
        domain=np.sign(x)*np.log1p(np.abs(x))
        offset=float(transform['coordinate_domain_min'])
    else:
        domain=x-np.asarray(transform['source_min'],dtype=np.float64)
        offset=0.0
    return np.rint((domain-offset)*float(transform['scale_to_quantized'])).clip(0,65535).astype(np.uint16)

def local_signed_log_low_bits(x, full_quantized, transform):
    """Compand only the 16 reconstruction levels inside a uniform VPCC cell.

    The 12-bit high coordinate remains byte-for-byte compatible with V12's
    V-PCC stream.  Only the 4-bit refinement uses signed-log spacing.
    """
    scale=float(transform['scale_to_quantized'])
    minimum=np.asarray(transform['source_min'],dtype=np.float64)
    high=(full_quantized.astype(np.uint32)>>4).astype(np.float64)
    lower=minimum+high*16.0/scale
    upper=minimum+(high*16.0+15.0)/scale
    lower_domain=np.sign(lower)*np.log1p(np.abs(lower))
    upper_domain=np.sign(upper)*np.log1p(np.abs(upper))
    value_domain=np.sign(x)*np.log1p(np.abs(x))
    return np.rint((value_domain-lower_domain)/(upper_domain-lower_domain)*15.0).clip(0,15).astype(np.uint8)

def pack_low_coordinates(low, bits):
    """Pack XYZ residual coordinates LSB-first without assuming four bits.

    Four-bit residuals retain V17's original nibble layout.  V18's compact
    10+6 trial needs 18 bits per Gaussian, so its three residual components
    must be packed as a continuous bitstream rather than stored in u32s.
    """
    if bits <= 0 or bits > 10:
        raise ValueError(f'unsupported residual width: {bits}')
    rows_per_chunk = 500_000
    alignment = 8 // np.gcd(8, 3 * bits)
    out = bytearray()
    shifts = np.arange(bits, dtype=np.uint16)
    for start in range(0, len(low), rows_per_chunk):
        stop = min(len(low), start + rows_per_chunk)
        if stop < len(low):
            stop -= (stop - start) % alignment
        if stop == start:
            stop = min(len(low), start + alignment)
        values = low[start:stop].reshape(-1).astype(np.uint16, copy=False)
        packed_bits = ((values[:, None] >> shifts) & 1).reshape(-1)
        out.extend(np.packbits(packed_bits, bitorder='little').tobytes())
    return bytes(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('v12',type=Path); p.add_argument('v16',type=Path); p.add_argument('out',type=Path); p.add_argument('--order',type=Path,required=True); p.add_argument('--vpcc-geometry-bitdepth',type=int,default=12,help='coarse V-PCC coordinate width; defaults to V17\'s 12'); p.add_argument('--local-signed-log-low-bits',action='store_true'); a=p.parse_args()
    if a.out.exists(): raise SystemExit(f'refusing to overwrite {a.out}')
    m=json.loads((a.v12/'manifest.json').read_text()); t=m['transform']
    source=Path('/mnt/c/Users/email/3DGS-VPCC-encoding')/m['source_ply']
    source_xyz=xyz(source); allq=quantize(source_xyz.astype(np.float64),t)
    high_bits=a.vpcc_geometry_bitdepth
    low_bits=16-high_bits
    if not 1 <= low_bits <= 10:
        raise SystemExit(f'unsupported high/low split: {high_bits}+{low_bits}')
    codes=np.unique(morton(allq>>low_bits,high_bits))
    order=np.load(a.order).reshape(-1)
    q=allq[order]
    high_q=q>>low_bits
    high_codes=morton(high_q,high_bits)
    ids=np.searchsorted(codes,high_codes).astype(np.uint32)
    if not np.all(codes[ids]==high_codes): raise SystemExit('anchor association mismatch')
    d=np.empty(len(ids),np.int64); d[0]=ids[0]; d[1:]=ids[1:].astype(np.int64)-ids[:-1].astype(np.int64); zz=((d<<1)^(d>>63)).astype('<u4')
    if a.local_signed_log_low_bits and low_bits != 4:
        raise SystemExit('--local-signed-log-low-bits is defined only for the 12+4 uniform-anchor experiment')
    low_quantized=(local_signed_log_low_bits(source_xyz.astype(np.float64),allq,t)[order] if a.local_signed_log_low_bits else (q&((1<<low_bits)-1)).astype(np.uint16))
    low=pack_low_coordinates(low_quantized,low_bits)
    a.out.mkdir(); sem=a.out/'semantic_sog'; sem.mkdir()
    for f in (a.v16/'sog_web_morton').iterdir():
        if f.name not in {'means_bytes_0.avif','means_bytes_1.avif'}: shutil.copy2(f,sem/f.name)
    shutil.copy2(a.v12/'stream.bin',a.out/'stream.bin')
    (a.out/'association.deflate').write_bytes(zlib.compress(zz.tobytes(),9)); (a.out/'low_bits.deflate').write_bytes(zlib.compress(low,9))
    mode=('v17_uniform_vpcc_local_signed_log_lowbits_morton_sog_hybrid' if a.local_signed_log_low_bits else ('v18_vpcc_signed_log_morton_sog_hybrid' if t.get('signed_log_coordinates',False) else 'v17_vpcc_morton_sog_hybrid'))
    files=[x for x in a.out.rglob('*') if x.is_file()]; report={'mode':mode,'records':len(q),'vpcc_anchors':len(codes),'vpcc_geometry_bitdepth':high_bits,'low_coordinate_bits':low_bits,'vpcc_stream':'stream.bin','semantic_dir':'semantic_sog','omitted_sog_means':['means_bytes_0.avif','means_bytes_1.avif'],'association':'association.deflate','low_bits':'low_bits.deflate','low_bits_transform':('signed_log1p_within_uniform_12bit_anchor_cell' if a.local_signed_log_low_bits else f'quantized_{low_bits}bit_residual'),'transform':t,'payload_bytes':sum(x.stat().st_size for x in files)}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
