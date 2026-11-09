//! This will be a morton order sorter
//! The difficulty will be quantizing to an integer for morton ordering while maintaining enough precision to exploit redundancy due to spatial proximity.
use gaussian_types::Gaussian;

//ChatGPT recommended function:
// a_q = clamp( floor( ((a - a_min) / (a_max - a_min)) * (2^b - 1) + 0.5 ), 0, 2^b - 1 )
//
// where:
//   a      = original floating-point coordinate (x, y, or z)
//   a_min  = minimum value of that axis in the dataset
//   a_max  = maximum value of that axis in the dataset
//   b      = number of bits per axis (e.g. 21)
//   a_q    = quantized integer coordinate in [0, 2^b - 1]
//
//notes:
// - this requires knowing the min and max of each value
//    -> we can track min/max during the parsing process without adding much computation
// - this is uniform distribution across min max
//    -> we should explore better quantization approaches to get more accurate predictions
fn quantize(value: f32, min : f32, max : f32, bits: u32) -> u32 {
    if max <= min {return 0;}
    let n = ((1u64 << bits) - 1) as f32;
    let t = ((value - min) / (max - min)).clamp(0.0, 1.0 );
    ((t * n).round() as u64) as u32
}

fn spread(mut x: u64) -> u64 {
    x &= 0x1F_FFFF;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    x
}

fn morton3d_64(x: u32, y: u32, z: u32) -> u64 {
    let xx = spread(x as u64);
    let yy = spread(y as u64) << 1;
    let zz = spread(z as u64) << 2;
    xx | yy | zz
}

pub fn generate_morton_code(
    gaussian: &Gaussian,
    mins: &[f32; 3],
    maxes: &[f32; 3],
) -> u64 {
    let b = 21;
    let xi = quantize(gaussian.xyz[0], mins[0], maxes[0], b);
    let yi = quantize(gaussian.xyz[1], mins[1], maxes[1], b);
    let zi = quantize(gaussian.xyz[2], mins[2], maxes[2], b);
    morton3d_64(xi, yi, zi)
}