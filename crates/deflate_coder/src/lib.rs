// src/main.rs
use libflate::deflate::{Decoder, EncodeOptions, Encoder};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Payload {
    data: Vec<f32>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct U16Payload {
    data: Vec<u16>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct U8Payload {
    data: Vec<u8>,
}

pub fn compress_f32_vec(input: Vec<f32>) -> anyhow::Result<Vec<u8>> {
    // 1) Serialize to bytes (endian-stable)
    let payload = Payload { data: input };
    let serialized = bincode::serialize(&payload)?; // Vec<u8>

    // 2) LZ77-based DEFLATE compression
    let mut encoder = Encoder::new(Vec::new());
    encoder.write_all(&serialized)?;
    let compressed = encoder.finish().into_result()?; // Vec<u8>
    Ok(compressed)
}

pub fn decompress_i32_vec(compressed: &[u8]) -> anyhow::Result<Vec<f32>> {
    // 3) DEFLATE decompression
    let mut decoder = Decoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    // 4) Deserialize back to Vec<i32>
    let restored: Payload = bincode::deserialize(&decompressed)?;
    Ok(restored.data)
}

pub fn compress_u16_vec(input: Vec<u16>) -> anyhow::Result<Vec<u8>> {
    compress_u16_vec_with_level(input, 1)
}

pub fn compress_u16_vec_with_level(input: Vec<u16>, level: u32) -> anyhow::Result<Vec<u8>> {
    let payload = U16Payload { data: input };
    let serialized = bincode::serialize(&payload)?;

    let options = match level {
        0 => EncodeOptions::new().no_compression(),
        1 => EncodeOptions::new().fixed_huffman_codes().block_size(256 * 1024),
        _ => EncodeOptions::new(),
    };
    let mut encoder = Encoder::with_options(Vec::new(), options);
    encoder.write_all(&serialized)?;
    let compressed = encoder.finish().into_result()?;
    Ok(compressed)
}

pub fn decompress_u16_vec(compressed: &[u8]) -> anyhow::Result<Vec<u16>> {
    let mut decoder = Decoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    let restored: U16Payload = bincode::deserialize(&decompressed)?;
    Ok(restored.data)
}

pub fn compress_u8_vec(input: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    compress_u8_vec_with_level(input, 1)
}

pub fn compress_u8_vec_with_level(input: Vec<u8>, level: u32) -> anyhow::Result<Vec<u8>> {
    let payload = U8Payload { data: input };
    let serialized = bincode::serialize(&payload)?;

    let options = match level {
        0 => EncodeOptions::new().no_compression(),
        1 => EncodeOptions::new().fixed_huffman_codes().block_size(256 * 1024),
        _ => EncodeOptions::new(),
    };
    let mut encoder = Encoder::with_options(Vec::new(), options);
    encoder.write_all(&serialized)?;
    let compressed = encoder.finish().into_result()?;
    Ok(compressed)
}

pub fn decompress_u8_vec(compressed: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut decoder = Decoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    let restored: U8Payload = bincode::deserialize(&decompressed)?;
    Ok(restored.data)
}

// fn main() -> anyhow::Result<()> {
//     let original = vec![1, -2, 3, 3, 3, 4, 4, 4, 4, 5];
//     let compressed = compress_i32_vec(original.clone())?;
//     let roundtrip = decompress_i32_vec(&compressed)?;
//
//     println!("original:  {:?}", original);
//     println!("compressed len: {}", compressed.len());
//     println!("roundtrip: {:?}", roundtrip);
//     Ok(())
// }
