use deflate_coder::{compress_u16_vec_with_level, decompress_u16_vec};
use gaussian_types::Gaussian;
use serde::{Deserialize, Serialize};

use crate::QuantParams;

pub const OPACITY_PAYLOAD_FILE: &str = "opacity_payload.bin";
pub const SCALE_PAYLOAD_FILE: &str = "scale_payload.bin";
pub const ROTATION_PAYLOAD_FILE: &str = "rotation_payload.bin";
pub const NORMALS_PAYLOAD_FILE: &str = "normals_payload.bin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpacityStrategy {
    RawF32,
    QuantizedDeflate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleStrategy {
    RawF32,
    LogDomainQuantizedDeflate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationStrategy {
    RawF32,
    DropLargest3QuantizedDeflate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalsStrategy {
    RawF32,
    OctahedralQuantizedDeflate,
    Omitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpacityCodecConfig {
    pub strategy: OpacityStrategy,
    pub quant_bits: u8,
    pub deflate_level: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleCodecConfig {
    pub strategy: ScaleStrategy,
    pub quant_bits: u8,
    pub deflate_level: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RotationCodecConfig {
    pub strategy: RotationStrategy,
    pub quant_bits: u8,
    pub deflate_level: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalsCodecConfig {
    pub strategy: NormalsStrategy,
    pub quant_bits: u8,
    pub deflate_level: u32,
    pub keep_if_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamPayloadLayout {
    pub name: String,
    pub coded_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarStats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub stddev: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpacityCodecMetadata {
    pub strategy: OpacityStrategy,
    pub quant_bits: Option<u8>,
    pub quant: Option<QuantParams>,
    pub raw_stream_bytes: u64,
    pub payload_streams: Vec<StreamPayloadLayout>,
    pub source_stats: ScalarStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleCodecMetadata {
    pub strategy: ScaleStrategy,
    pub quant_bits: Option<u8>,
    pub quant: Vec<QuantParams>,
    pub raw_stream_bytes: Vec<u64>,
    pub payload_streams: Vec<StreamPayloadLayout>,
    pub source_stats: Vec<ScalarStats>,
    pub assumes_input_is_log_domain: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RotationCodecMetadata {
    pub strategy: RotationStrategy,
    pub quant_bits: Option<u8>,
    pub quant: Vec<QuantParams>,
    pub raw_stream_bytes: Vec<u64>,
    pub payload_streams: Vec<StreamPayloadLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalsCodecMetadata {
    pub strategy: NormalsStrategy,
    pub quant_bits: Option<u8>,
    pub quant: Vec<QuantParams>,
    pub raw_stream_bytes: Vec<u64>,
    pub payload_streams: Vec<StreamPayloadLayout>,
    pub source_stats: Vec<ScalarStats>,
    pub kept_for_downstream: bool,
    pub valid_count: usize,
    pub invalid_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuxiliaryCodecMetadata {
    pub opacity: OpacityCodecMetadata,
    pub scale: ScaleCodecMetadata,
    pub rotation: RotationCodecMetadata,
    pub normals: NormalsCodecMetadata,
}

impl Default for OpacityCodecConfig {
    fn default() -> Self {
        Self {
            strategy: OpacityStrategy::QuantizedDeflate,
            quant_bits: 12,
            deflate_level: 1,
        }
    }
}

impl Default for ScaleCodecConfig {
    fn default() -> Self {
        Self {
            strategy: ScaleStrategy::LogDomainQuantizedDeflate,
            quant_bits: 14,
            deflate_level: 1,
        }
    }
}

impl Default for RotationCodecConfig {
    fn default() -> Self {
        Self {
            strategy: RotationStrategy::DropLargest3QuantizedDeflate,
            quant_bits: 15,
            deflate_level: 1,
        }
    }
}

impl Default for NormalsCodecConfig {
    fn default() -> Self {
        Self {
            strategy: NormalsStrategy::OctahedralQuantizedDeflate,
            quant_bits: 14,
            deflate_level: 1,
            keep_if_present: true,
        }
    }
}

pub fn encode_opacity(
    gaussians: &[Gaussian],
    config: OpacityCodecConfig,
) -> Result<(Vec<f32>, Vec<u8>, OpacityCodecMetadata), Box<dyn std::error::Error>> {
    let values = gaussians.iter().map(|g| g.opacity).collect::<Vec<_>>();
    let raw_stream_bytes = (values.len() * std::mem::size_of::<f32>()) as u64;
    let source_stats = scalar_stats(&values);
    match config.strategy {
        OpacityStrategy::RawF32 => {
            let payload = f32_payload(&values);
            Ok((
                values,
                payload.clone(),
                OpacityCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: None,
                    quant: None,
                    raw_stream_bytes,
                    payload_streams: vec![StreamPayloadLayout {
                        name: "opacity".to_string(),
                        coded_bytes: payload.len() as u64,
                    }],
                    source_stats,
                },
            ))
        }
        OpacityStrategy::QuantizedDeflate => {
            let quant = quant_params(&values);
            let quantized = quantize_stream(&values, quant, config.quant_bits);
            let payload = compress_u16_vec_with_level(quantized, config.deflate_level)?;
            let decoded =
                decode_quantized_stream(&payload, values.len(), quant, config.quant_bits)?;
            Ok((
                decoded,
                payload.clone(),
                OpacityCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant: Some(quant),
                    raw_stream_bytes,
                    payload_streams: vec![StreamPayloadLayout {
                        name: "opacity".to_string(),
                        coded_bytes: payload.len() as u64,
                    }],
                    source_stats,
                },
            ))
        }
    }
}

pub fn decode_opacity(
    payload: &[u8],
    gaussian_len: usize,
    metadata: &OpacityCodecMetadata,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match metadata.strategy {
        OpacityStrategy::RawF32 => decode_f32_payload(payload, gaussian_len),
        OpacityStrategy::QuantizedDeflate => decode_quantized_stream(
            payload,
            gaussian_len,
            metadata.quant.ok_or("missing opacity quant params")?,
            metadata.quant_bits.ok_or("missing opacity quant bits")?,
        ),
    }
}

pub fn encode_scale(
    gaussians: &[Gaussian],
    config: ScaleCodecConfig,
) -> Result<([Vec<f32>; 3], Vec<u8>, ScaleCodecMetadata), Box<dyn std::error::Error>> {
    let streams = [
        gaussians.iter().map(|g| g.scale[0]).collect::<Vec<_>>(),
        gaussians.iter().map(|g| g.scale[1]).collect::<Vec<_>>(),
        gaussians.iter().map(|g| g.scale[2]).collect::<Vec<_>>(),
    ];
    let raw_stream_bytes = streams
        .iter()
        .map(|stream| (stream.len() * std::mem::size_of::<f32>()) as u64)
        .collect::<Vec<_>>();
    let source_stats = streams
        .iter()
        .map(|stream| scalar_stats(stream))
        .collect::<Vec<_>>();
    match config.strategy {
        ScaleStrategy::RawF32 => {
            let payload = pack_raw_f32_streams(&[
                ("scale_x", &streams[0]),
                ("scale_y", &streams[1]),
                ("scale_z", &streams[2]),
            ]);
            Ok((
                streams.clone(),
                payload.clone(),
                ScaleCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: None,
                    quant: Vec::new(),
                    raw_stream_bytes,
                    payload_streams: vec![
                        StreamPayloadLayout {
                            name: "scale_x".to_string(),
                            coded_bytes: streams[0].len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "scale_y".to_string(),
                            coded_bytes: streams[1].len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "scale_z".to_string(),
                            coded_bytes: streams[2].len() as u64 * 4,
                        },
                    ],
                    source_stats,
                    assumes_input_is_log_domain: true,
                },
            ))
        }
        ScaleStrategy::LogDomainQuantizedDeflate => {
            let mut payload = Vec::new();
            let mut quant = Vec::with_capacity(3);
            let mut payload_streams = Vec::with_capacity(3);
            let mut decoded = [Vec::new(), Vec::new(), Vec::new()];
            for (idx, name) in ["scale_x", "scale_y", "scale_z"].into_iter().enumerate() {
                let params = quant_params(&streams[idx]);
                let quantized = quantize_stream(&streams[idx], params, config.quant_bits);
                let compressed = compress_u16_vec_with_level(quantized, config.deflate_level)?;
                decoded[idx] = decode_quantized_stream(
                    &compressed,
                    streams[idx].len(),
                    params,
                    config.quant_bits,
                )?;
                payload.extend_from_slice(&compressed);
                quant.push(params);
                payload_streams.push(StreamPayloadLayout {
                    name: name.to_string(),
                    coded_bytes: compressed.len() as u64,
                });
            }
            Ok((
                decoded,
                payload,
                ScaleCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant,
                    raw_stream_bytes,
                    payload_streams,
                    source_stats,
                    assumes_input_is_log_domain: true,
                },
            ))
        }
    }
}

pub fn decode_scale(
    payload: &[u8],
    gaussian_len: usize,
    metadata: &ScaleCodecMetadata,
) -> Result<[Vec<f32>; 3], Box<dyn std::error::Error>> {
    match metadata.strategy {
        ScaleStrategy::RawF32 => decode_raw_f32_streams3(payload, gaussian_len),
        ScaleStrategy::LogDomainQuantizedDeflate => {
            let quant_bits = metadata.quant_bits.ok_or("missing scale quant bits")?;
            let chunks = split_payload(payload, &metadata.payload_streams)?;
            Ok([
                decode_quantized_stream(&chunks[0], gaussian_len, metadata.quant[0], quant_bits)?,
                decode_quantized_stream(&chunks[1], gaussian_len, metadata.quant[1], quant_bits)?,
                decode_quantized_stream(&chunks[2], gaussian_len, metadata.quant[2], quant_bits)?,
            ])
        }
    }
}

pub fn encode_rotation(
    gaussians: &[Gaussian],
    config: RotationCodecConfig,
) -> Result<([Vec<f32>; 4], Vec<u8>, RotationCodecMetadata), Box<dyn std::error::Error>> {
    let raw_stream_bytes = vec![(gaussians.len() * std::mem::size_of::<f32>()) as u64; 4];
    match config.strategy {
        RotationStrategy::RawF32 => {
            let streams = [
                gaussians.iter().map(|g| g.rot[0]).collect::<Vec<_>>(),
                gaussians.iter().map(|g| g.rot[1]).collect::<Vec<_>>(),
                gaussians.iter().map(|g| g.rot[2]).collect::<Vec<_>>(),
                gaussians.iter().map(|g| g.rot[3]).collect::<Vec<_>>(),
            ];
            let payload = pack_raw_f32_streams(&[
                ("rot_x", &streams[0]),
                ("rot_y", &streams[1]),
                ("rot_z", &streams[2]),
                ("rot_w", &streams[3]),
            ]);
            Ok((
                streams.clone(),
                payload,
                RotationCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: None,
                    quant: Vec::new(),
                    raw_stream_bytes,
                    payload_streams: vec![
                        StreamPayloadLayout {
                            name: "rot_x".to_string(),
                            coded_bytes: streams[0].len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "rot_y".to_string(),
                            coded_bytes: streams[1].len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "rot_z".to_string(),
                            coded_bytes: streams[2].len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "rot_w".to_string(),
                            coded_bytes: streams[3].len() as u64 * 4,
                        },
                    ],
                },
            ))
        }
        RotationStrategy::DropLargest3QuantizedDeflate => {
            let mut dropped_index = Vec::with_capacity(gaussians.len());
            let mut kept = [
                Vec::with_capacity(gaussians.len()),
                Vec::with_capacity(gaussians.len()),
                Vec::with_capacity(gaussians.len()),
            ];
            for gaussian in gaussians {
                let canonical = canonicalize_quaternion(gaussian.rot);
                let drop_idx = dominant_component_index(canonical);
                dropped_index.push(drop_idx as u16);
                let mut kept_idx = 0usize;
                for component_idx in 0..4 {
                    if component_idx == drop_idx {
                        continue;
                    }
                    kept[kept_idx].push(canonical[component_idx]);
                    kept_idx += 1;
                }
            }

            let index_payload =
                compress_u16_vec_with_level(dropped_index.clone(), config.deflate_level)?;
            let mut payload = index_payload.clone();
            let mut payload_streams = vec![StreamPayloadLayout {
                name: "rot_drop_index".to_string(),
                coded_bytes: index_payload.len() as u64,
            }];
            let mut quant = Vec::with_capacity(3);
            for stream_idx in 0..3 {
                let params = quant_params(&kept[stream_idx]);
                let quantized = quantize_stream(&kept[stream_idx], params, config.quant_bits);
                let compressed = compress_u16_vec_with_level(quantized, config.deflate_level)?;
                payload.extend_from_slice(&compressed);
                payload_streams.push(StreamPayloadLayout {
                    name: format!("rot_keep_{stream_idx}"),
                    coded_bytes: compressed.len() as u64,
                });
                quant.push(params);
            }
            let decoded = decode_rotation(
                &payload,
                gaussians.len(),
                &RotationCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant: quant.clone(),
                    raw_stream_bytes: raw_stream_bytes.clone(),
                    payload_streams: payload_streams.clone(),
                },
            )?;
            Ok((
                decoded,
                payload,
                RotationCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant,
                    raw_stream_bytes,
                    payload_streams,
                },
            ))
        }
    }
}

pub fn decode_rotation(
    payload: &[u8],
    gaussian_len: usize,
    metadata: &RotationCodecMetadata,
) -> Result<[Vec<f32>; 4], Box<dyn std::error::Error>> {
    match metadata.strategy {
        RotationStrategy::RawF32 => decode_raw_f32_streams(payload, gaussian_len, 4),
        RotationStrategy::DropLargest3QuantizedDeflate => {
            let quant_bits = metadata.quant_bits.ok_or("missing rotation quant bits")?;
            let chunks = split_payload(payload, &metadata.payload_streams)?;
            let dropped_index = decompress_u16_vec(&chunks[0])?;
            if dropped_index.len() != gaussian_len {
                return Err(format!(
                    "rotation dropped-index length mismatch: expected {gaussian_len}, got {}",
                    dropped_index.len()
                )
                .into());
            }
            let kept0 =
                decode_quantized_stream(&chunks[1], gaussian_len, metadata.quant[0], quant_bits)?;
            let kept1 =
                decode_quantized_stream(&chunks[2], gaussian_len, metadata.quant[1], quant_bits)?;
            let kept2 =
                decode_quantized_stream(&chunks[3], gaussian_len, metadata.quant[2], quant_bits)?;
            let mut streams = [
                Vec::with_capacity(gaussian_len),
                Vec::with_capacity(gaussian_len),
                Vec::with_capacity(gaussian_len),
                Vec::with_capacity(gaussian_len),
            ];
            for gaussian_idx in 0..gaussian_len {
                let drop_idx = dropped_index[gaussian_idx] as usize;
                if drop_idx >= 4 {
                    return Err(format!("invalid rotation drop index: {drop_idx}").into());
                }
                let kept_values = [
                    kept0[gaussian_idx],
                    kept1[gaussian_idx],
                    kept2[gaussian_idx],
                ];
                let mut quat = [0.0f32; 4];
                let mut kept_idx = 0usize;
                let mut sum_sq = 0.0f32;
                for component_idx in 0..4 {
                    if component_idx == drop_idx {
                        continue;
                    }
                    quat[component_idx] = kept_values[kept_idx];
                    sum_sq += kept_values[kept_idx] * kept_values[kept_idx];
                    kept_idx += 1;
                }
                quat[drop_idx] = (1.0 - sum_sq).max(0.0).sqrt();
                quat = normalize_quaternion(quat);
                for component_idx in 0..4 {
                    streams[component_idx].push(quat[component_idx]);
                }
            }
            Ok(streams)
        }
    }
}

pub fn encode_normals(
    gaussians: &[Gaussian],
    config: NormalsCodecConfig,
) -> Result<(Option<[Vec<f32>; 3]>, Vec<u8>, NormalsCodecMetadata), Box<dyn std::error::Error>> {
    let normals_present = gaussians.iter().any(|g| g.normals.is_some());
    if !normals_present || !config.keep_if_present {
        return Ok((
            None,
            Vec::new(),
            NormalsCodecMetadata {
                strategy: NormalsStrategy::Omitted,
                quant_bits: None,
                quant: Vec::new(),
                raw_stream_bytes: vec![0, 0, 0],
                payload_streams: Vec::new(),
                source_stats: Vec::new(),
                kept_for_downstream: false,
                valid_count: 0,
                invalid_count: gaussians.len(),
            },
        ));
    }

    let xyz = [
        gaussians
            .iter()
            .map(|g| g.normals.unwrap_or([0.0; 3])[0])
            .collect::<Vec<_>>(),
        gaussians
            .iter()
            .map(|g| g.normals.unwrap_or([0.0; 3])[1])
            .collect::<Vec<_>>(),
        gaussians
            .iter()
            .map(|g| g.normals.unwrap_or([0.0; 3])[2])
            .collect::<Vec<_>>(),
    ];
    let raw_stream_bytes = xyz
        .iter()
        .map(|stream| (stream.len() * std::mem::size_of::<f32>()) as u64)
        .collect::<Vec<_>>();
    let source_stats = xyz
        .iter()
        .map(|stream| scalar_stats(stream))
        .collect::<Vec<_>>();
    match config.strategy {
        NormalsStrategy::RawF32 => {
            let payload = pack_raw_f32_streams(&[
                ("normals_x", &xyz[0]),
                ("normals_y", &xyz[1]),
                ("normals_z", &xyz[2]),
            ]);
            Ok((
                Some(xyz),
                payload,
                NormalsCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: None,
                    quant: Vec::new(),
                    raw_stream_bytes,
                    payload_streams: vec![
                        StreamPayloadLayout {
                            name: "normals_x".to_string(),
                            coded_bytes: gaussians.len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "normals_y".to_string(),
                            coded_bytes: gaussians.len() as u64 * 4,
                        },
                        StreamPayloadLayout {
                            name: "normals_z".to_string(),
                            coded_bytes: gaussians.len() as u64 * 4,
                        },
                    ],
                    source_stats,
                    kept_for_downstream: true,
                    valid_count: gaussians.len(),
                    invalid_count: 0,
                },
            ))
        }
        NormalsStrategy::OctahedralQuantizedDeflate => {
            let mut validity = Vec::with_capacity(gaussians.len());
            let mut oct_u = Vec::with_capacity(gaussians.len());
            let mut oct_v = Vec::with_capacity(gaussians.len());
            for gaussian in gaussians {
                let source = gaussian.normals.unwrap_or([0.0, 0.0, 0.0]);
                let norm = vec3_norm(source);
                if norm <= 1e-12 {
                    validity.push(0u16);
                    oct_u.push(0.0);
                    oct_v.push(0.0);
                    continue;
                }
                validity.push(1u16);
                let [u, v] = oct_encode(normalize_vec3(source));
                oct_u.push(u);
                oct_v.push(v);
            }
            let valid_count = validity.iter().filter(|&&v| v != 0).count();
            let invalid_count = validity.len() - valid_count;
            let quant_u = QuantParams {
                min: -1.0,
                max: 1.0,
            };
            let quant_v = QuantParams {
                min: -1.0,
                max: 1.0,
            };
            let compressed_validity = compress_u16_vec_with_level(validity, config.deflate_level)?;
            let compressed_u = compress_u16_vec_with_level(
                quantize_stream(&oct_u, quant_u, config.quant_bits),
                config.deflate_level,
            )?;
            let compressed_v = compress_u16_vec_with_level(
                quantize_stream(&oct_v, quant_v, config.quant_bits),
                config.deflate_level,
            )?;
            let mut payload = compressed_validity.clone();
            payload.extend_from_slice(&compressed_u);
            payload.extend_from_slice(&compressed_v);
            let decoded = decode_normals(
                &payload,
                gaussians.len(),
                &NormalsCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant: vec![quant_u, quant_v],
                    raw_stream_bytes: raw_stream_bytes.clone(),
                    payload_streams: vec![
                        StreamPayloadLayout {
                            name: "normals_validity".to_string(),
                            coded_bytes: compressed_validity.len() as u64,
                        },
                        StreamPayloadLayout {
                            name: "normals_oct_u".to_string(),
                            coded_bytes: compressed_u.len() as u64,
                        },
                        StreamPayloadLayout {
                            name: "normals_oct_v".to_string(),
                            coded_bytes: compressed_v.len() as u64,
                        },
                    ],
                    source_stats: source_stats.clone(),
                    kept_for_downstream: true,
                    valid_count,
                    invalid_count,
                },
            )?;
            Ok((
                decoded,
                payload,
                NormalsCodecMetadata {
                    strategy: config.strategy,
                    quant_bits: Some(config.quant_bits),
                    quant: vec![quant_u, quant_v],
                    raw_stream_bytes,
                    payload_streams: vec![
                        StreamPayloadLayout {
                            name: "normals_validity".to_string(),
                            coded_bytes: compressed_validity.len() as u64,
                        },
                        StreamPayloadLayout {
                            name: "normals_oct_u".to_string(),
                            coded_bytes: compressed_u.len() as u64,
                        },
                        StreamPayloadLayout {
                            name: "normals_oct_v".to_string(),
                            coded_bytes: compressed_v.len() as u64,
                        },
                    ],
                    source_stats,
                    kept_for_downstream: true,
                    valid_count,
                    invalid_count,
                },
            ))
        }
        NormalsStrategy::Omitted => Ok((
            None,
            Vec::new(),
            NormalsCodecMetadata {
                strategy: NormalsStrategy::Omitted,
                quant_bits: None,
                quant: Vec::new(),
                raw_stream_bytes,
                payload_streams: Vec::new(),
                source_stats,
                kept_for_downstream: false,
                valid_count: 0,
                invalid_count: gaussians.len(),
            },
        )),
    }
}

pub fn decode_normals(
    payload: &[u8],
    gaussian_len: usize,
    metadata: &NormalsCodecMetadata,
) -> Result<Option<[Vec<f32>; 3]>, Box<dyn std::error::Error>> {
    match metadata.strategy {
        NormalsStrategy::Omitted => Ok(None),
        NormalsStrategy::RawF32 => Ok(Some(decode_raw_f32_streams3(payload, gaussian_len)?)),
        NormalsStrategy::OctahedralQuantizedDeflate => {
            let quant_bits = metadata.quant_bits.ok_or("missing normals quant bits")?;
            let chunks = split_payload(payload, &metadata.payload_streams)?;
            let validity = decompress_u16_vec(&chunks[0])?;
            if validity.len() != gaussian_len {
                return Err(format!(
                    "normals validity length mismatch: expected {gaussian_len}, got {}",
                    validity.len()
                )
                .into());
            }
            let u =
                decode_quantized_stream(&chunks[1], gaussian_len, metadata.quant[0], quant_bits)?;
            let v =
                decode_quantized_stream(&chunks[2], gaussian_len, metadata.quant[1], quant_bits)?;
            let mut xyz = [
                Vec::with_capacity(gaussian_len),
                Vec::with_capacity(gaussian_len),
                Vec::with_capacity(gaussian_len),
            ];
            for idx in 0..gaussian_len {
                let normal = if validity[idx] == 0 {
                    [0.0, 0.0, 0.0]
                } else {
                    oct_decode([u[idx], v[idx]])
                };
                xyz[0].push(normal[0]);
                xyz[1].push(normal[1]);
                xyz[2].push(normal[2]);
            }
            Ok(Some(xyz))
        }
    }
}

fn split_payload(
    payload: &[u8],
    layout: &[StreamPayloadLayout],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(layout.len());
    for stream in layout {
        let size = stream.coded_bytes as usize;
        let end = offset + size;
        if end > payload.len() {
            return Err(format!(
                "payload too short for stream {}: need {}, have {}",
                stream.name,
                end,
                payload.len()
            )
            .into());
        }
        chunks.push(payload[offset..end].to_vec());
        offset = end;
    }
    if offset != payload.len() {
        return Err(format!(
            "payload length mismatch after stream split: consumed {}, total {}",
            offset,
            payload.len()
        )
        .into());
    }
    Ok(chunks)
}

fn pack_raw_f32_streams(streams: &[(&str, &Vec<f32>)]) -> Vec<u8> {
    let total_values = streams
        .iter()
        .map(|(_, stream)| stream.len())
        .sum::<usize>();
    let mut payload = Vec::with_capacity(total_values * 4);
    for (_, stream) in streams {
        payload.extend_from_slice(&f32_payload(stream));
    }
    payload
}

fn decode_raw_f32_streams(
    payload: &[u8],
    gaussian_len: usize,
    stream_count: usize,
) -> Result<[Vec<f32>; 4], Box<dyn std::error::Error>> {
    let bytes_per_stream = gaussian_len * std::mem::size_of::<f32>();
    let expected = bytes_per_stream * stream_count;
    if payload.len() != expected {
        return Err(format!(
            "raw f32 payload length mismatch: expected {expected}, got {}",
            payload.len()
        )
        .into());
    }
    let mut out = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for stream_idx in 0..stream_count {
        let start = stream_idx * bytes_per_stream;
        let end = start + bytes_per_stream;
        out[stream_idx] = decode_f32_payload(&payload[start..end], gaussian_len)?;
    }
    Ok(out)
}

fn decode_raw_f32_streams3(
    payload: &[u8],
    gaussian_len: usize,
) -> Result<[Vec<f32>; 3], Box<dyn std::error::Error>> {
    let [x, y, z, _] = decode_raw_f32_streams(payload, gaussian_len, 3)?;
    Ok([x, y, z])
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(values.len() * 4);
    for &value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload
}

fn decode_f32_payload(
    payload: &[u8],
    expected_len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if payload.len() != expected_len * 4 {
        return Err(format!(
            "raw f32 payload length mismatch: expected {}, got {}",
            expected_len * 4,
            payload.len()
        )
        .into());
    }
    Ok(payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn quant_params(values: &[f32]) -> QuantParams {
    let Some(&first) = values.first() else {
        return QuantParams { min: 0.0, max: 0.0 };
    };
    let mut min = first;
    let mut max = first;
    for &value in values.iter().skip(1) {
        min = min.min(value);
        max = max.max(value);
    }
    QuantParams { min, max }
}

fn quantize_stream(values: &[f32], params: QuantParams, quant_bits: u8) -> Vec<u16> {
    let max_code = ((1u32 << quant_bits.min(16)) - 1) as f32;
    if params.max <= params.min {
        return vec![0u16; values.len()];
    }
    values
        .iter()
        .map(|&value| {
            let normalized = ((value - params.min) / (params.max - params.min)).clamp(0.0, 1.0);
            (normalized * max_code).round() as u16
        })
        .collect()
}

fn decode_quantized_stream(
    payload: &[u8],
    expected_len: usize,
    params: QuantParams,
    quant_bits: u8,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let quantized = decompress_u16_vec(payload)?;
    if quantized.len() != expected_len {
        return Err(format!(
            "quantized stream length mismatch: expected {expected_len}, got {}",
            quantized.len()
        )
        .into());
    }
    let max_code = ((1u32 << quant_bits.min(16)) - 1) as f32;
    if params.max <= params.min || max_code <= 0.0 {
        return Ok(vec![params.min; expected_len]);
    }
    Ok(quantized
        .into_iter()
        .map(|value| params.min + (value as f32 / max_code) * (params.max - params.min))
        .collect())
}

fn scalar_stats(values: &[f32]) -> ScalarStats {
    let Some(&first) = values.first() else {
        return ScalarStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            stddev: 0.0,
        };
    };
    let mut min = first;
    let mut max = first;
    let mut sum = 0.0f64;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
    }
    let mean = sum / values.len() as f64;
    let mut sum_sq = 0.0f64;
    for &value in values {
        let centered = value as f64 - mean;
        sum_sq += centered * centered;
    }
    ScalarStats {
        min,
        max,
        mean,
        stddev: (sum_sq / values.len() as f64).sqrt(),
    }
}

fn canonicalize_quaternion(quat: [f32; 4]) -> [f32; 4] {
    let normalized = normalize_quaternion(quat);
    let idx = dominant_component_index(normalized);
    if normalized[idx] < 0.0 {
        [
            -normalized[0],
            -normalized[1],
            -normalized[2],
            -normalized[3],
        ]
    } else {
        normalized
    }
}

fn dominant_component_index(quat: [f32; 4]) -> usize {
    let mut best_idx = 0usize;
    let mut best_abs = quat[0].abs();
    for idx in 1..4 {
        let abs = quat[idx].abs();
        if abs > best_abs {
            best_abs = abs;
            best_idx = idx;
        }
    }
    best_idx
}

fn normalize_quaternion(quat: [f32; 4]) -> [f32; 4] {
    let norm =
        (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();
    if norm <= 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [
            quat[0] / norm,
            quat[1] / norm,
            quat[2] / norm,
            quat[3] / norm,
        ]
    }
}

fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let norm = vec3_norm(v);
    if norm <= 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

fn oct_encode(normal: [f32; 3]) -> [f32; 2] {
    let inv_l1 = 1.0 / (normal[0].abs() + normal[1].abs() + normal[2].abs()).max(1e-12);
    let mut x = normal[0] * inv_l1;
    let mut y = normal[1] * inv_l1;
    if normal[2] < 0.0 {
        let old_x = x;
        x = (1.0 - y.abs()) * sign_not_zero(old_x);
        y = (1.0 - old_x.abs()) * sign_not_zero(y);
    }
    [x, y]
}

fn oct_decode(oct: [f32; 2]) -> [f32; 3] {
    let mut normal = [oct[0], oct[1], 1.0 - oct[0].abs() - oct[1].abs()];
    if normal[2] < 0.0 {
        let old_x = normal[0];
        normal[0] = (1.0 - normal[1].abs()) * sign_not_zero(old_x);
        normal[1] = (1.0 - old_x.abs()) * sign_not_zero(normal[1]);
    }
    normalize_vec3(normal)
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sign_not_zero(value: f32) -> f32 {
    if value < 0.0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaussian_types::Gaussian;

    #[test]
    fn oct_roundtrip_preserves_unit_normals() {
        let samples = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.57735026, 0.57735026, 0.57735026],
            [-0.30151135, 0.904534, -0.30151135],
        ];
        for sample in samples {
            let decoded = oct_decode(oct_encode(sample));
            let dot = sample[0] * decoded[0] + sample[1] * decoded[1] + sample[2] * decoded[2];
            assert!(
                dot > 0.999,
                "dot too low: {dot} sample={sample:?} decoded={decoded:?}"
            );
        }
    }

    #[test]
    fn normals_codec_preserves_zero_normals() {
        let gaussians = vec![
            Gaussian {
                xyz: [0.0; 3],
                normals: Some([0.0, 0.0, 0.0]),
                sh_dc: [0.0; 3],
                sh_rest: Vec::new(),
                opacity: 0.0,
                scale: [0.0; 3],
                rot: [0.0, 0.0, 0.0, 1.0],
            },
            Gaussian {
                xyz: [0.0; 3],
                normals: Some([0.0, 0.0, 1.0]),
                sh_dc: [0.0; 3],
                sh_rest: Vec::new(),
                opacity: 0.0,
                scale: [0.0; 3],
                rot: [0.0, 0.0, 0.0, 1.0],
            },
        ];
        let (decoded, payload, metadata) =
            encode_normals(&gaussians, NormalsCodecConfig::default()).unwrap();
        let roundtrip = decode_normals(&payload, gaussians.len(), &metadata).unwrap();
        let decoded = decoded.unwrap();
        let roundtrip = roundtrip.unwrap();
        assert_eq!(decoded[2][0], 0.0);
        assert_eq!(roundtrip[2][0], 0.0);
        assert_eq!(metadata.invalid_count, 1);
        assert_eq!(metadata.valid_count, 1);
    }
}
