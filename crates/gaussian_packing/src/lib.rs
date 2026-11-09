mod experimental_streams;

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use deflate_coder::{compress_u8_vec_with_level, decompress_u8_vec};
use gaussian_parser::Scene;
use gaussian_sorter::generate_morton_code;
use gaussian_types::Gaussian;
use serde::{Deserialize, Serialize};

pub use experimental_streams::{
    AuxiliaryCodecMetadata, NORMALS_PAYLOAD_FILE, NormalsCodecConfig, NormalsCodecMetadata,
    NormalsStrategy, OPACITY_PAYLOAD_FILE, OpacityCodecConfig, OpacityCodecMetadata,
    OpacityStrategy, ROTATION_PAYLOAD_FILE, RotationCodecConfig, RotationCodecMetadata,
    RotationStrategy, SCALE_PAYLOAD_FILE, ScaleCodecConfig, ScaleCodecMetadata, ScaleStrategy,
    StreamPayloadLayout,
};

const META_FILE: &str = "meta.json";
const OCCUPANCY_FILE: &str = "occupancy.bin";
const GEOM_X_FILE: &str = "geom_x.bin";
const GEOM_Y_FILE: &str = "geom_y.bin";
const GEOM_Z_FILE: &str = "geom_z.bin";
const GEOM_HI_VIDEO_FILE: &str = "geom_hi.mkv";
const GEOM_LO_VIDEO_FILE: &str = "geom_lo.mkv";
const SH_DC_R_FILE: &str = "sh_dc_r.bin";
const SH_DC_G_FILE: &str = "sh_dc_g.bin";
const SH_DC_B_FILE: &str = "sh_dc_b.bin";
const SH_DC_HI_VIDEO_FILE: &str = "sh_dc_hi.mkv";
const SH_DC_LO_VIDEO_FILE: &str = "sh_dc_lo.mkv";
const SH_DC_R_HI_DEFLATE_FILE: &str = "sh_dc_r_hi.deflate";
const SH_DC_R_LO_DEFLATE_FILE: &str = "sh_dc_r_lo.deflate";
const SH_DC_G_HI_DEFLATE_FILE: &str = "sh_dc_g_hi.deflate";
const SH_DC_G_LO_DEFLATE_FILE: &str = "sh_dc_g_lo.deflate";
const SH_DC_B_HI_DEFLATE_FILE: &str = "sh_dc_b_hi.deflate";
const SH_DC_B_LO_DEFLATE_FILE: &str = "sh_dc_b_lo.deflate";
const SH_REST_HI_VIDEO_FILE: &str = "sh_rest_hi.mkv";
const SH_REST_LO_VIDEO_FILE: &str = "sh_rest_lo.mkv";
const STORAGE_RAW: &str = "raw";
const SH_REST_STORAGE_CODED_VIDEO_ATLAS: &str = "coded_video_atlas";
const STORAGE_BYTE_SPLIT_DEFLATE: &str = "byte_split_deflate";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    Lossless,
    VeryGood,
    Ok,
}

impl QualityPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lossless => "lossless",
            Self::VeryGood => "very_good",
            Self::Ok => "ok",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShRestVideoAtlasCodecConfig {
    pub quality_preset: QualityPreset,
    pub codec: &'static str,
    pub pixel_format: &'static str,
    pub ffmpeg_preset: Option<&'static str>,
    pub crf: Option<u8>,
    pub lossless: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupStorageMode {
    Raw,
    CodedVideoAtlas,
    ByteSplitDeflate,
}

impl GroupStorageMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => STORAGE_RAW,
            Self::CodedVideoAtlas => SH_REST_STORAGE_CODED_VIDEO_ATLAS,
            Self::ByteSplitDeflate => STORAGE_BYTE_SPLIT_DEFLATE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedRasterStorageConfig {
    pub geometry: GroupStorageMode,
    pub sh_dc: GroupStorageMode,
    pub sh_rest: GroupStorageMode,
    pub opacity: OpacityCodecConfig,
    pub scale: ScaleCodecConfig,
    pub rotation: RotationCodecConfig,
    pub normals: NormalsCodecConfig,
    pub geometry_codec: ShRestVideoAtlasCodecConfig,
    pub sh_dc_codec: ShRestVideoAtlasCodecConfig,
    pub sh_rest_codec: ShRestVideoAtlasCodecConfig,
}

impl PackedRasterStorageConfig {
    pub fn vpcc_video_atlas(preset: QualityPreset) -> Self {
        let (geometry_codec, sh_dc_codec, sh_rest_codec) = match preset {
            QualityPreset::Lossless => (
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
            ),
            QualityPreset::VeryGood => (
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
                ShRestVideoAtlasCodecConfig {
                    quality_preset: preset,
                    codec: "libx264rgb",
                    pixel_format: "rgb24",
                    ffmpeg_preset: Some("medium"),
                    crf: Some(6),
                    lossless: false,
                },
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::VeryGood),
            ),
            QualityPreset::Ok => (
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
                ShRestVideoAtlasCodecConfig {
                    quality_preset: preset,
                    codec: "libx264rgb",
                    pixel_format: "rgb24",
                    ffmpeg_preset: Some("medium"),
                    crf: Some(12),
                    lossless: false,
                },
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Ok),
            ),
        };

        Self {
            geometry: GroupStorageMode::CodedVideoAtlas,
            sh_dc: GroupStorageMode::ByteSplitDeflate,
            sh_rest: GroupStorageMode::CodedVideoAtlas,
            opacity: OpacityCodecConfig::default(),
            scale: ScaleCodecConfig::default(),
            rotation: RotationCodecConfig::default(),
            normals: NormalsCodecConfig::default(),
            geometry_codec,
            sh_dc_codec,
            sh_rest_codec,
        }
    }
}

impl ShRestVideoAtlasCodecConfig {
    pub fn from_quality_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::Lossless => Self {
                quality_preset: preset,
                codec: "ffv1",
                pixel_format: "rgb24",
                ffmpeg_preset: None,
                crf: None,
                lossless: true,
            },
            QualityPreset::VeryGood => Self {
                quality_preset: preset,
                codec: "libx264rgb",
                pixel_format: "rgb24",
                ffmpeg_preset: Some("medium"),
                crf: Some(12),
                lossless: false,
            },
            QualityPreset::Ok => Self {
                quality_preset: preset,
                codec: "libx264rgb",
                pixel_format: "rgb24",
                ffmpeg_preset: Some("veryfast"),
                crf: Some(24),
                lossless: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantParams {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackedRasterScene {
    pub num_gaussians: u32,
    pub width: u32,
    pub height: u32,
    pub sh_rest_len: u32,
    pub normals_present: bool,
    pub occupancy: Vec<u8>,
    pub geom_x: Vec<u16>,
    pub geom_y: Vec<u16>,
    pub geom_z: Vec<u16>,
    pub sh_dc_r: Vec<u16>,
    pub sh_dc_g: Vec<u16>,
    pub sh_dc_b: Vec<u16>,
    pub normals_x: Vec<f32>,
    pub normals_y: Vec<f32>,
    pub normals_z: Vec<f32>,
    pub sh_rest_quant: Vec<QuantParams>,
    pub sh_rest_payload: Vec<u8>,
    pub opacity: Vec<f32>,
    pub scale_x: Vec<f32>,
    pub scale_y: Vec<f32>,
    pub scale_z: Vec<f32>,
    pub rot_x: Vec<f32>,
    pub rot_y: Vec<f32>,
    pub rot_z: Vec<f32>,
    pub rot_w: Vec<f32>,
    pub geom_quant: [QuantParams; 3],
    pub sh_dc_quant: [QuantParams; 3],
    pub auxiliary_codec_metadata: Option<AuxiliaryCodecMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodedGroupWriteTiming {
    pub group_name: String,
    pub ffmpeg_encode: Duration,
    pub temp_file_io: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerWriteTimings {
    pub container_assembly: Duration,
    pub coded_groups: Vec<CodedGroupWriteTiming>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PackedRasterMetadata {
    num_gaussians: u32,
    width: u32,
    height: u32,
    sh_rest_len: u32,
    normals_present: bool,
    geometry_storage: String,
    sh_dc_storage: String,
    sh_rest_storage: String,
    opacity_codec: OpacityCodecMetadata,
    scale_codec: ScaleCodecMetadata,
    rotation_codec: RotationCodecMetadata,
    normals_codec: NormalsCodecMetadata,
    sh_rest_quant: Vec<QuantParams>,
    geom_quant: [QuantParams; 3],
    sh_dc_quant: [QuantParams; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDeltaStats {
    pub count: usize,
    pub mean_abs: f64,
    pub std_abs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborDeltaStats {
    pub xyz: [ChannelDeltaStats; 3],
    pub sh_dc: [ChannelDeltaStats; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationStats {
    pub count: usize,
    pub pearson_r: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlaneStats {
    pub name: &'static str,
    pub min: u32,
    pub max: u32,
    pub entropy_bits_per_symbol: f64,
    pub neighbor_delta: ChannelDeltaStats,
    pub horizontal_correlation: CorrelationStats,
    pub vertical_correlation: CorrelationStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ByteSplitPlane {
    pub name: &'static str,
    pub hi: Vec<u8>,
    pub lo: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ByteSplitPlaneStats {
    pub name: &'static str,
    pub base_u16: PlaneStats,
    pub hi: PlaneStats,
    pub lo: PlaneStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiPngPaths {
    pub geom_x_hi: PathBuf,
    pub geom_y_hi: PathBuf,
    pub geom_z_hi: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiRgbPngPath {
    pub geom_hi_rgb: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShDcHiRgbPngPath {
    pub sh_dc_hi_rgb: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiPngExperimentResult {
    pub image_sizes: [u64; 3],
    pub total_coded_bytes: u64,
    pub xyz_rmse: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShRestPayloadStats {
    pub coded_bytes: usize,
    pub raw_bytes: usize,
    pub storage: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShRestTierSpec {
    pub label: String,
    pub start_stream: usize,
    pub end_stream: usize,
    pub codec_config: ShRestVideoAtlasCodecConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedContainerByteSummary {
    pub geometry_bytes: u64,
    pub sh_dc_bytes: u64,
    pub sh_rest_bytes: u64,
    pub opacity_bytes: u64,
    pub scale_bytes: u64,
    pub rotation_bytes: u64,
    pub normals_bytes: u64,
    pub occupancy_bytes: u64,
    pub metadata_bytes: u64,
    pub total_coded_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShRestPackTimings {
    pub quantization: Duration,
    pub buffer_construction: Duration,
    pub atlas_construction: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackRasterTimings {
    pub geometry: Duration,
    pub sh_dc: Duration,
    pub sh_rest: Duration,
    pub sh_rest_breakdown: ShRestPackTimings,
    pub remaining_streams: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundtripDiagnostics {
    pub xyz_rmse: [f64; 3],
    pub sh_dc_rmse: [f64; 3],
    pub original_neighbor_deltas: NeighborDeltaStats,
    pub reconstructed_neighbor_deltas: NeighborDeltaStats,
    pub plane_stats: Vec<PlaneStats>,
    pub byte_split_plane_stats: Vec<ByteSplitPlaneStats>,
}

pub fn pack_scene_to_raster(scene: &Scene, width: u32) -> PackedRasterScene {
    assert!(width > 0, "atlas width must be greater than zero");

    let mut sorted = scene.gaussians.clone();
    sorted.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));
    pack_sorted_gaussians_to_raster(&sorted, width)
}

pub fn pack_sorted_gaussians_to_raster(sorted: &[Gaussian], width: u32) -> PackedRasterScene {
    pack_sorted_gaussians_to_raster_with_timing(sorted, width).0
}

pub fn pack_sorted_gaussians_to_raster_with_timing(
    sorted: &[Gaussian],
    width: u32,
) -> (PackedRasterScene, PackRasterTimings) {
    assert!(width > 0, "atlas width must be greater than zero");

    let num_gaussians = sorted.len() as u32;
    let sh_rest_len = sorted.first().map(|g| g.sh_rest.len() as u32).unwrap_or(0);
    let normals_present = sorted.iter().any(|g| g.normals.is_some());
    let sh_rest_start = Instant::now();
    let (sh_rest_quant, sh_rest_payload, sh_rest_breakdown) =
        encode_sh_rest_payload(sorted, sh_rest_len as usize).expect("sh_rest encoding failed");
    let sh_rest = sh_rest_start.elapsed();
    let height = if num_gaussians == 0 {
        0
    } else {
        num_gaussians.div_ceil(width)
    };
    let plane_len = (width as usize) * (height as usize);

    let (geom_quant, sh_dc_quant) = compute_quant_params(sorted);
    let mut packed = PackedRasterScene {
        num_gaussians,
        width,
        height,
        sh_rest_len,
        normals_present,
        occupancy: vec![0; plane_len],
        geom_x: vec![0; plane_len],
        geom_y: vec![0; plane_len],
        geom_z: vec![0; plane_len],
        sh_dc_r: vec![0; plane_len],
        sh_dc_g: vec![0; plane_len],
        sh_dc_b: vec![0; plane_len],
        normals_x: Vec::with_capacity(num_gaussians as usize),
        normals_y: Vec::with_capacity(num_gaussians as usize),
        normals_z: Vec::with_capacity(num_gaussians as usize),
        sh_rest_quant,
        sh_rest_payload,
        opacity: Vec::with_capacity(num_gaussians as usize),
        scale_x: Vec::with_capacity(num_gaussians as usize),
        scale_y: Vec::with_capacity(num_gaussians as usize),
        scale_z: Vec::with_capacity(num_gaussians as usize),
        rot_x: Vec::with_capacity(num_gaussians as usize),
        rot_y: Vec::with_capacity(num_gaussians as usize),
        rot_z: Vec::with_capacity(num_gaussians as usize),
        rot_w: Vec::with_capacity(num_gaussians as usize),
        geom_quant,
        sh_dc_quant,
        auxiliary_codec_metadata: None,
    };

    let geometry_start = Instant::now();
    for (idx, gaussian) in sorted.iter().enumerate() {
        packed.occupancy[idx] = 1;
        packed.geom_x[idx] = quantize_to_u16(gaussian.xyz[0], packed.geom_quant[0]);
        packed.geom_y[idx] = quantize_to_u16(gaussian.xyz[1], packed.geom_quant[1]);
        packed.geom_z[idx] = quantize_to_u16(gaussian.xyz[2], packed.geom_quant[2]);
    }
    let geometry = geometry_start.elapsed();

    let sh_dc_start = Instant::now();
    for (idx, gaussian) in sorted.iter().enumerate() {
        packed.sh_dc_r[idx] = quantize_to_u16(gaussian.sh_dc[0], packed.sh_dc_quant[0]);
        packed.sh_dc_g[idx] = quantize_to_u16(gaussian.sh_dc[1], packed.sh_dc_quant[1]);
        packed.sh_dc_b[idx] = quantize_to_u16(gaussian.sh_dc[2], packed.sh_dc_quant[2]);
    }
    let sh_dc = sh_dc_start.elapsed();

    let remaining_streams_start = Instant::now();
    for gaussian in sorted {
        let normals = gaussian.normals.unwrap_or([0.0; 3]);
        packed.normals_x.push(normals[0]);
        packed.normals_y.push(normals[1]);
        packed.normals_z.push(normals[2]);
        packed.opacity.push(gaussian.opacity);
        packed.scale_x.push(gaussian.scale[0]);
        packed.scale_y.push(gaussian.scale[1]);
        packed.scale_z.push(gaussian.scale[2]);
        packed.rot_x.push(gaussian.rot[0]);
        packed.rot_y.push(gaussian.rot[1]);
        packed.rot_z.push(gaussian.rot[2]);
        packed.rot_w.push(gaussian.rot[3]);
    }
    let remaining_streams = remaining_streams_start.elapsed();

    (
        packed,
        PackRasterTimings {
            geometry,
            sh_dc,
            sh_rest,
            sh_rest_breakdown,
            remaining_streams,
        },
    )
}

pub fn unpack_raster_to_scene(packed: &PackedRasterScene) -> Scene {
    let mut gaussians = Vec::with_capacity(packed.num_gaussians as usize);
    let sh_rest_len = packed.sh_rest_len as usize;
    let mut gaussian_idx = 0usize;
    let decoded_sh_rest = decode_sh_rest_payload(
        &packed.sh_rest_payload,
        &packed.sh_rest_quant,
        packed.num_gaussians as usize,
    )
    .expect("sh_rest decoding failed");

    for idx in 0..packed.occupancy.len() {
        if packed.occupancy[idx] == 0 {
            continue;
        }

        let xyz = [
            dequantize_from_u16(packed.geom_x[idx], packed.geom_quant[0]),
            dequantize_from_u16(packed.geom_y[idx], packed.geom_quant[1]),
            dequantize_from_u16(packed.geom_z[idx], packed.geom_quant[2]),
        ];
        let sh_dc = [
            dequantize_from_u16(packed.sh_dc_r[idx], packed.sh_dc_quant[0]),
            dequantize_from_u16(packed.sh_dc_g[idx], packed.sh_dc_quant[1]),
            dequantize_from_u16(packed.sh_dc_b[idx], packed.sh_dc_quant[2]),
        ];
        let normals = if packed.normals_present {
            Some([
                packed.normals_x[gaussian_idx],
                packed.normals_y[gaussian_idx],
                packed.normals_z[gaussian_idx],
            ])
        } else {
            None
        };

        gaussians.push(Gaussian {
            xyz,
            normals,
            sh_dc,
            sh_rest: decoded_sh_rest[gaussian_idx * sh_rest_len..(gaussian_idx + 1) * sh_rest_len]
                .to_vec(),
            opacity: packed.opacity[gaussian_idx],
            scale: [
                packed.scale_x[gaussian_idx],
                packed.scale_y[gaussian_idx],
                packed.scale_z[gaussian_idx],
            ],
            rot: [
                packed.rot_x[gaussian_idx],
                packed.rot_y[gaussian_idx],
                packed.rot_z[gaussian_idx],
                packed.rot_w[gaussian_idx],
            ],
        });
        gaussian_idx += 1;
    }

    gaussians.truncate(packed.num_gaussians as usize);
    let (mins, maxes) = compute_xyz_bounds(&gaussians);
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "packed_raster_v1".to_string());
    meta.insert("vertex_count".to_string(), gaussians.len().to_string());

    Scene {
        gaussians,
        meta,
        mins,
        maxes,
    }
}

pub fn write_packed_raster_scene_dir(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_packed_raster_scene_dir_with_config(
        packed,
        output_dir,
        PackedRasterStorageConfig {
            geometry: GroupStorageMode::Raw,
            sh_dc: GroupStorageMode::Raw,
            sh_rest: GroupStorageMode::CodedVideoAtlas,
            opacity: OpacityCodecConfig {
                strategy: OpacityStrategy::RawF32,
                ..OpacityCodecConfig::default()
            },
            scale: ScaleCodecConfig {
                strategy: ScaleStrategy::RawF32,
                ..ScaleCodecConfig::default()
            },
            rotation: RotationCodecConfig {
                strategy: RotationStrategy::RawF32,
                ..RotationCodecConfig::default()
            },
            normals: NormalsCodecConfig {
                strategy: NormalsStrategy::RawF32,
                ..NormalsCodecConfig::default()
            },
            geometry_codec: ShRestVideoAtlasCodecConfig::from_quality_preset(
                QualityPreset::Lossless,
            ),
            sh_dc_codec: ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless),
            sh_rest_codec: ShRestVideoAtlasCodecConfig::from_quality_preset(
                QualityPreset::Lossless,
            ),
        },
    )
}

pub fn write_packed_raster_scene_dir_with_config(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
    storage_config: PackedRasterStorageConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    write_packed_raster_scene_dir_with_report(packed, output_dir, storage_config).map(|_| ())
}

pub fn write_packed_raster_scene_dir_with_report(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
    storage_config: PackedRasterStorageConfig,
) -> Result<ContainerWriteTimings, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let container_assembly_start = Instant::now();
    let gaussians = gaussians_from_aux_streams(packed);
    let (_, opacity_payload, opacity_codec) =
        experimental_streams::encode_opacity(&gaussians, storage_config.opacity)?;
    let (_, scale_payload, scale_codec) =
        experimental_streams::encode_scale(&gaussians, storage_config.scale)?;
    let (_, rotation_payload, rotation_codec) =
        experimental_streams::encode_rotation(&gaussians, storage_config.rotation)?;
    let (_, normals_payload, normals_codec) =
        experimental_streams::encode_normals(&gaussians, storage_config.normals)?;

    let metadata = PackedRasterMetadata {
        num_gaussians: packed.num_gaussians,
        width: packed.width,
        height: packed.height,
        sh_rest_len: packed.sh_rest_len,
        normals_present: normals_codec.kept_for_downstream,
        geometry_storage: storage_config.geometry.as_str().to_string(),
        sh_dc_storage: storage_config.sh_dc.as_str().to_string(),
        sh_rest_storage: storage_config.sh_rest.as_str().to_string(),
        opacity_codec,
        scale_codec,
        rotation_codec,
        normals_codec,
        sh_rest_quant: packed.sh_rest_quant.clone(),
        geom_quant: packed.geom_quant,
        sh_dc_quant: packed.sh_dc_quant,
    };

    let meta_path = output_dir.join(META_FILE);
    let meta_writer = BufWriter::new(File::create(meta_path)?);
    serde_json::to_writer_pretty(meta_writer, &metadata)?;

    write_u8_plane(output_dir.join(OCCUPANCY_FILE), &packed.occupancy)?;
    let mut coded_groups = Vec::new();
    if let Some(timing) = write_u16_group(
        output_dir,
        "geom",
        [&packed.geom_x, &packed.geom_y, &packed.geom_z],
        storage_config.geometry,
        storage_config.geometry_codec,
        [GEOM_X_FILE, GEOM_Y_FILE, GEOM_Z_FILE],
        [GEOM_HI_VIDEO_FILE, GEOM_LO_VIDEO_FILE],
        [
            "geom_x_hi.deflate",
            "geom_x_lo.deflate",
            "geom_y_hi.deflate",
            "geom_y_lo.deflate",
            "geom_z_hi.deflate",
            "geom_z_lo.deflate",
        ],
        packed.width,
        packed.height,
        packed.num_gaussians as usize,
    )? {
        coded_groups.push(timing);
    }
    if let Some(timing) = write_u16_group(
        output_dir,
        "sh_dc",
        [&packed.sh_dc_r, &packed.sh_dc_g, &packed.sh_dc_b],
        storage_config.sh_dc,
        storage_config.sh_dc_codec,
        [SH_DC_R_FILE, SH_DC_G_FILE, SH_DC_B_FILE],
        [SH_DC_HI_VIDEO_FILE, SH_DC_LO_VIDEO_FILE],
        [
            SH_DC_R_HI_DEFLATE_FILE,
            SH_DC_R_LO_DEFLATE_FILE,
            SH_DC_G_HI_DEFLATE_FILE,
            SH_DC_G_LO_DEFLATE_FILE,
            SH_DC_B_HI_DEFLATE_FILE,
            SH_DC_B_LO_DEFLATE_FILE,
        ],
        packed.width,
        packed.height,
        packed.num_gaussians as usize,
    )? {
        coded_groups.push(timing);
    }
    match storage_config.sh_rest {
        GroupStorageMode::Raw => {
            return Err("raw sh_rest storage is not supported".into());
        }
        GroupStorageMode::ByteSplitDeflate => {
            return Err("byte_split_deflate sh_rest storage is not supported".into());
        }
        GroupStorageMode::CodedVideoAtlas => {
            coded_groups.push(write_sh_rest_video_atlas(
                packed,
                output_dir,
                storage_config.sh_rest_codec,
            )?);
        }
    }
    write_u8_plane(output_dir.join(OPACITY_PAYLOAD_FILE), &opacity_payload)?;
    write_u8_plane(output_dir.join(SCALE_PAYLOAD_FILE), &scale_payload)?;
    write_u8_plane(output_dir.join(ROTATION_PAYLOAD_FILE), &rotation_payload)?;
    write_u8_plane(output_dir.join(NORMALS_PAYLOAD_FILE), &normals_payload)?;

    Ok(ContainerWriteTimings {
        container_assembly: container_assembly_start.elapsed(),
        coded_groups,
    })
}

pub fn read_packed_raster_scene_dir(
    input_dir: impl AsRef<Path>,
) -> Result<PackedRasterScene, Box<dyn std::error::Error>> {
    let input_dir = input_dir.as_ref();
    let metadata: PackedRasterMetadata =
        serde_json::from_reader(BufReader::new(File::open(input_dir.join(META_FILE))?))?;

    let plane_len = (metadata.width as usize) * (metadata.height as usize);

    let occupancy = read_u8_plane(input_dir.join(OCCUPANCY_FILE), plane_len)?;
    let geometry_storage = parse_storage_mode(&metadata.geometry_storage)?;
    let sh_dc_storage = parse_storage_mode(&metadata.sh_dc_storage)?;
    let [geom_x, geom_y, geom_z] = read_u16_group(
        input_dir,
        geometry_storage,
        [GEOM_X_FILE, GEOM_Y_FILE, GEOM_Z_FILE],
        [GEOM_HI_VIDEO_FILE, GEOM_LO_VIDEO_FILE],
        [
            "geom_x_hi.deflate",
            "geom_x_lo.deflate",
            "geom_y_hi.deflate",
            "geom_y_lo.deflate",
            "geom_z_hi.deflate",
            "geom_z_lo.deflate",
        ],
        metadata.width,
        metadata.height,
        plane_len,
        "geom",
    )?;
    let [sh_dc_r, sh_dc_g, sh_dc_b] = read_u16_group(
        input_dir,
        sh_dc_storage,
        [SH_DC_R_FILE, SH_DC_G_FILE, SH_DC_B_FILE],
        [SH_DC_HI_VIDEO_FILE, SH_DC_LO_VIDEO_FILE],
        [
            SH_DC_R_HI_DEFLATE_FILE,
            SH_DC_R_LO_DEFLATE_FILE,
            SH_DC_G_HI_DEFLATE_FILE,
            SH_DC_G_LO_DEFLATE_FILE,
            SH_DC_B_HI_DEFLATE_FILE,
            SH_DC_B_LO_DEFLATE_FILE,
        ],
        metadata.width,
        metadata.height,
        plane_len,
        "sh_dc",
    )?;
    let gaussian_len = metadata.num_gaussians as usize;
    if parse_storage_mode(&metadata.sh_rest_storage)? != GroupStorageMode::CodedVideoAtlas {
        return Err(format!(
            "unsupported sh_rest storage mode: {}",
            metadata.sh_rest_storage
        )
        .into());
    }
    let sh_rest = read_sh_rest_video_atlas(
        input_dir,
        metadata.width,
        metadata.height,
        metadata.num_gaussians as usize,
        metadata.sh_rest_len as usize,
    )?;
    let opacity_payload = read_u8_blob(input_dir.join(OPACITY_PAYLOAD_FILE))?;
    let scale_payload = read_u8_blob(input_dir.join(SCALE_PAYLOAD_FILE))?;
    let rotation_payload = read_u8_blob(input_dir.join(ROTATION_PAYLOAD_FILE))?;
    let normals_payload = read_u8_blob(input_dir.join(NORMALS_PAYLOAD_FILE))?;
    let opacity = experimental_streams::decode_opacity(
        &opacity_payload,
        gaussian_len,
        &metadata.opacity_codec,
    )?;
    let [scale_x, scale_y, scale_z] =
        experimental_streams::decode_scale(&scale_payload, gaussian_len, &metadata.scale_codec)?;
    let [rot_x, rot_y, rot_z, rot_w] = experimental_streams::decode_rotation(
        &rotation_payload,
        gaussian_len,
        &metadata.rotation_codec,
    )?;
    let normals = experimental_streams::decode_normals(
        &normals_payload,
        gaussian_len,
        &metadata.normals_codec,
    )?;
    let (normals_x, normals_y, normals_z) = match normals {
        Some([x, y, z]) => (x, y, z),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };

    Ok(PackedRasterScene {
        num_gaussians: metadata.num_gaussians,
        width: metadata.width,
        height: metadata.height,
        sh_rest_len: metadata.sh_rest_len,
        normals_present: metadata.normals_present,
        occupancy,
        geom_x,
        geom_y,
        geom_z,
        sh_dc_r,
        sh_dc_g,
        sh_dc_b,
        normals_x,
        normals_y,
        normals_z,
        sh_rest_quant: metadata.sh_rest_quant,
        sh_rest_payload: sh_rest,
        opacity,
        scale_x,
        scale_y,
        scale_z,
        rot_x,
        rot_y,
        rot_z,
        rot_w,
        geom_quant: metadata.geom_quant,
        sh_dc_quant: metadata.sh_dc_quant,
        auxiliary_codec_metadata: Some(AuxiliaryCodecMetadata {
            opacity: metadata.opacity_codec,
            scale: metadata.scale_codec,
            rotation: metadata.rotation_codec,
            normals: metadata.normals_codec,
        }),
    })
}

fn gaussians_from_aux_streams(packed: &PackedRasterScene) -> Vec<Gaussian> {
    let gaussian_len = packed.num_gaussians as usize;
    let normals_present = packed.normals_present;
    let mut gaussians = Vec::with_capacity(gaussian_len);
    for idx in 0..gaussian_len {
        gaussians.push(Gaussian {
            xyz: [0.0; 3],
            normals: normals_present.then_some([
                packed.normals_x[idx],
                packed.normals_y[idx],
                packed.normals_z[idx],
            ]),
            sh_dc: [0.0; 3],
            sh_rest: Vec::new(),
            opacity: packed.opacity[idx],
            scale: [
                packed.scale_x[idx],
                packed.scale_y[idx],
                packed.scale_z[idx],
            ],
            rot: [
                packed.rot_x[idx],
                packed.rot_y[idx],
                packed.rot_z[idx],
                packed.rot_w[idx],
            ],
        });
    }
    gaussians
}

pub fn write_scene_to_ply(
    path: impl AsRef<Path>,
    scene: &Scene,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_gaussians = scene.gaussians.len();
    let sh_rest_len = scene
        .gaussians
        .first()
        .map(|g| g.sh_rest.len())
        .unwrap_or(0);
    let normals_present = scene.gaussians.iter().any(|g| g.normals.is_some());

    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "element vertex {}", num_gaussians)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    if normals_present {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }
    writeln!(writer, "property float f_dc_0")?;
    writeln!(writer, "property float f_dc_1")?;
    writeln!(writer, "property float f_dc_2")?;
    for i in 0..sh_rest_len {
        writeln!(writer, "property float f_rest_{}", i)?;
    }
    writeln!(writer, "property float opacity")?;
    writeln!(writer, "property float scale_0")?;
    writeln!(writer, "property float scale_1")?;
    writeln!(writer, "property float scale_2")?;
    writeln!(writer, "property float rot_0")?;
    writeln!(writer, "property float rot_1")?;
    writeln!(writer, "property float rot_2")?;
    writeln!(writer, "property float rot_3")?;
    writeln!(writer, "end_header")?;

    for gaussian in &scene.gaussians {
        writer.write_all(&gaussian.xyz[0].to_le_bytes())?;
        writer.write_all(&gaussian.xyz[1].to_le_bytes())?;
        writer.write_all(&gaussian.xyz[2].to_le_bytes())?;
        if normals_present {
            let normals = gaussian.normals.unwrap_or([0.0; 3]);
            writer.write_all(&normals[0].to_le_bytes())?;
            writer.write_all(&normals[1].to_le_bytes())?;
            writer.write_all(&normals[2].to_le_bytes())?;
        }
        writer.write_all(&gaussian.sh_dc[0].to_le_bytes())?;
        writer.write_all(&gaussian.sh_dc[1].to_le_bytes())?;
        writer.write_all(&gaussian.sh_dc[2].to_le_bytes())?;
        for value in &gaussian.sh_rest {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&gaussian.opacity.to_le_bytes())?;
        writer.write_all(&gaussian.scale[0].to_le_bytes())?;
        writer.write_all(&gaussian.scale[1].to_le_bytes())?;
        writer.write_all(&gaussian.scale[2].to_le_bytes())?;
        writer.write_all(&gaussian.rot[0].to_le_bytes())?;
        writer.write_all(&gaussian.rot[1].to_le_bytes())?;
        writer.write_all(&gaussian.rot[2].to_le_bytes())?;
        writer.write_all(&gaussian.rot[3].to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

pub fn packed_raster_raw_payload_bytes(packed: &PackedRasterScene) -> usize {
    packed.occupancy.len()
        + (packed.geom_x.len() * std::mem::size_of::<u16>())
        + (packed.geom_y.len() * std::mem::size_of::<u16>())
        + (packed.geom_z.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_r.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_g.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_b.len() * std::mem::size_of::<u16>())
        + (packed.normals_x.len() * std::mem::size_of::<f32>())
        + (packed.normals_y.len() * std::mem::size_of::<f32>())
        + (packed.normals_z.len() * std::mem::size_of::<f32>())
        + sh_rest_payload_stats(packed).raw_bytes
        + (packed.opacity.len() * std::mem::size_of::<f32>())
        + (packed.scale_x.len() * std::mem::size_of::<f32>())
        + (packed.scale_y.len() * std::mem::size_of::<f32>())
        + (packed.scale_z.len() * std::mem::size_of::<f32>())
        + (packed.rot_x.len() * std::mem::size_of::<f32>())
        + (packed.rot_y.len() * std::mem::size_of::<f32>())
        + (packed.rot_z.len() * std::mem::size_of::<f32>())
        + (packed.rot_w.len() * std::mem::size_of::<f32>())
}

pub fn sh_rest_payload_stats(packed: &PackedRasterScene) -> ShRestPayloadStats {
    ShRestPayloadStats {
        coded_bytes: packed.sh_rest_payload.len(),
        raw_bytes: packed.num_gaussians as usize
            * packed.sh_rest_len as usize
            * std::mem::size_of::<f32>(),
        storage: SH_REST_STORAGE_CODED_VIDEO_ATLAS,
    }
}

pub fn decode_sh_rest_payload_values(
    payload: &[u8],
    quant_params: &[QuantParams],
    gaussian_len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    decode_sh_rest_payload(payload, quant_params, gaussian_len)
}

pub fn encode_sh_rest_payload_from_values(
    values: &[f32],
    quant_params: &[QuantParams],
    gaussian_len: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let sh_rest_len = quant_params.len();
    let expected_len = gaussian_len * sh_rest_len;
    if values.len() != expected_len {
        return Err(format!(
            "sh_rest values length mismatch: expected {expected_len}, got {}",
            values.len()
        )
        .into());
    }

    let mut payload = vec![0u8; expected_len * std::mem::size_of::<u16>()];
    for stream_idx in 0..sh_rest_len {
        let params = quant_params[stream_idx];
        for gaussian_idx in 0..gaussian_len {
            let value = values[gaussian_idx * sh_rest_len + stream_idx];
            let quantized = quantize_to_u16(value, params).to_le_bytes();
            let offset = (stream_idx * gaussian_len + gaussian_idx) * std::mem::size_of::<u16>();
            payload[offset] = quantized[0];
            payload[offset + 1] = quantized[1];
        }
    }
    Ok(payload)
}

pub fn quantize_sh_rest_scalar(value: f32, params: QuantParams) -> u16 {
    quantize_to_u16(value, params)
}

pub fn measure_packed_raster_scene_dir(
    input_dir: impl AsRef<Path>,
) -> std::io::Result<PackedContainerByteSummary> {
    let input_dir = input_dir.as_ref();
    let metadata: PackedRasterMetadata =
        serde_json::from_reader(BufReader::new(File::open(input_dir.join(META_FILE))?))?;

    let geometry_bytes = measure_u16_group_bytes(
        input_dir,
        parse_storage_mode(&metadata.geometry_storage)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?,
        [GEOM_X_FILE, GEOM_Y_FILE, GEOM_Z_FILE],
        [GEOM_HI_VIDEO_FILE, GEOM_LO_VIDEO_FILE],
        [
            "geom_x_hi.deflate",
            "geom_x_lo.deflate",
            "geom_y_hi.deflate",
            "geom_y_lo.deflate",
            "geom_z_hi.deflate",
            "geom_z_lo.deflate",
        ],
    )?;
    let sh_dc_bytes = measure_u16_group_bytes(
        input_dir,
        parse_storage_mode(&metadata.sh_dc_storage)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?,
        [SH_DC_R_FILE, SH_DC_G_FILE, SH_DC_B_FILE],
        [SH_DC_HI_VIDEO_FILE, SH_DC_LO_VIDEO_FILE],
        [
            SH_DC_R_HI_DEFLATE_FILE,
            SH_DC_R_LO_DEFLATE_FILE,
            SH_DC_G_HI_DEFLATE_FILE,
            SH_DC_G_LO_DEFLATE_FILE,
            SH_DC_B_HI_DEFLATE_FILE,
            SH_DC_B_LO_DEFLATE_FILE,
        ],
    )?;
    let sh_rest_bytes = file_len(input_dir.join(SH_REST_HI_VIDEO_FILE))?
        + file_len(input_dir.join(SH_REST_LO_VIDEO_FILE))?;
    let opacity_bytes = file_len(input_dir.join(OPACITY_PAYLOAD_FILE))?;
    let scale_bytes = file_len(input_dir.join(SCALE_PAYLOAD_FILE))?;
    let rotation_bytes = file_len(input_dir.join(ROTATION_PAYLOAD_FILE))?;
    let normals_bytes = file_len(input_dir.join(NORMALS_PAYLOAD_FILE))?;
    let occupancy_bytes = file_len(input_dir.join(OCCUPANCY_FILE))?;
    let metadata_bytes = file_len(input_dir.join(META_FILE))?;
    let total_coded_bytes = geometry_bytes
        + sh_dc_bytes
        + sh_rest_bytes
        + opacity_bytes
        + scale_bytes
        + rotation_bytes
        + normals_bytes
        + occupancy_bytes
        + metadata_bytes;

    Ok(PackedContainerByteSummary {
        geometry_bytes,
        sh_dc_bytes,
        sh_rest_bytes,
        opacity_bytes,
        scale_bytes,
        rotation_bytes,
        normals_bytes,
        occupancy_bytes,
        metadata_bytes,
        total_coded_bytes,
    })
}

fn file_len(path: impl AsRef<Path>) -> std::io::Result<u64> {
    Ok(fs::metadata(path.as_ref())?.len())
}

fn parse_storage_mode(value: &str) -> Result<GroupStorageMode, Box<dyn std::error::Error>> {
    match value {
        STORAGE_RAW => Ok(GroupStorageMode::Raw),
        SH_REST_STORAGE_CODED_VIDEO_ATLAS => Ok(GroupStorageMode::CodedVideoAtlas),
        STORAGE_BYTE_SPLIT_DEFLATE => Ok(GroupStorageMode::ByteSplitDeflate),
        _ => Err(format!("unsupported storage mode: {value}").into()),
    }
}

fn measure_u16_group_bytes(
    input_dir: &Path,
    storage_mode: GroupStorageMode,
    raw_files: [&str; 3],
    video_files: [&str; 2],
    byte_split_deflate_files: [&str; 6],
) -> std::io::Result<u64> {
    match storage_mode {
        GroupStorageMode::Raw => Ok(file_len(input_dir.join(raw_files[0]))?
            + file_len(input_dir.join(raw_files[1]))?
            + file_len(input_dir.join(raw_files[2]))?),
        GroupStorageMode::CodedVideoAtlas => {
            Ok(file_len(input_dir.join(video_files[0]))?
                + file_len(input_dir.join(video_files[1]))?)
        }
        GroupStorageMode::ByteSplitDeflate => Ok(byte_split_deflate_files
            .into_iter()
            .map(|path| file_len(input_dir.join(path)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum()),
    }
}

fn write_u16_group(
    output_dir: &Path,
    temp_prefix: &str,
    planes: [&[u16]; 3],
    storage_mode: GroupStorageMode,
    codec_config: ShRestVideoAtlasCodecConfig,
    raw_files: [&str; 3],
    video_files: [&str; 2],
    byte_split_deflate_files: [&str; 6],
    width: u32,
    height: u32,
    active_len: usize,
) -> Result<Option<CodedGroupWriteTiming>, Box<dyn std::error::Error>> {
    match storage_mode {
        GroupStorageMode::Raw => {
            write_u16_plane(output_dir.join(raw_files[0]), planes[0])?;
            write_u16_plane(output_dir.join(raw_files[1]), planes[1])?;
            write_u16_plane(output_dir.join(raw_files[2]), planes[2])?;
            Ok(None)
        }
        GroupStorageMode::CodedVideoAtlas => Ok(Some(write_u16_video_atlas(
            planes,
            width,
            height,
            active_len,
            output_dir,
            video_files,
            temp_prefix,
            codec_config,
        )?)),
        GroupStorageMode::ByteSplitDeflate => {
            write_u16_byte_split_deflate(output_dir, planes, byte_split_deflate_files)?;
            Ok(None)
        }
    }
}

fn read_u16_group(
    input_dir: &Path,
    storage_mode: GroupStorageMode,
    raw_files: [&str; 3],
    video_files: [&str; 2],
    byte_split_deflate_files: [&str; 6],
    width: u32,
    height: u32,
    plane_len: usize,
    temp_prefix: &str,
) -> Result<[Vec<u16>; 3], Box<dyn std::error::Error>> {
    match storage_mode {
        GroupStorageMode::Raw => Ok([
            read_u16_plane(input_dir.join(raw_files[0]), plane_len)?,
            read_u16_plane(input_dir.join(raw_files[1]), plane_len)?,
            read_u16_plane(input_dir.join(raw_files[2]), plane_len)?,
        ]),
        GroupStorageMode::CodedVideoAtlas => read_u16_video_atlas(
            input_dir,
            width,
            height,
            plane_len,
            video_files,
            temp_prefix,
        ),
        GroupStorageMode::ByteSplitDeflate => {
            read_u16_byte_split_deflate(input_dir, plane_len, byte_split_deflate_files)
        }
    }
}

pub fn write_debug_preview_pngs(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    write_preview_u8_png(
        output_dir.join("occupancy.png"),
        packed.width,
        packed.height,
        &packed.occupancy,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_x.png"),
        packed.width,
        packed.height,
        &packed.geom_x,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_y.png"),
        packed.width,
        packed.height,
        &packed.geom_y,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_z.png"),
        packed.width,
        packed.height,
        &packed.geom_z,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_r.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_r,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_g.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_g,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_b.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_b,
        &packed.occupancy,
    )?;

    let byte_split = split_packed_u16_planes(packed);
    for plane in &byte_split {
        write_preview_u8_png(
            output_dir.join(format!("{}_hi.png", plane.name)),
            packed.width,
            packed.height,
            &plane.hi,
            &packed.occupancy,
        )?;
        write_preview_u8_png(
            output_dir.join(format!("{}_lo.png", plane.name)),
            packed.width,
            packed.height,
            &plane.lo,
            &packed.occupancy,
        )?;
    }

    Ok(())
}

pub fn write_byte_split_planes_dir(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let byte_split = split_packed_u16_planes(packed);
    for plane in &byte_split {
        write_u8_plane(output_dir.join(format!("{}_hi.bin", plane.name)), &plane.hi)?;
        write_u8_plane(output_dir.join(format!("{}_lo.bin", plane.name)), &plane.lo)?;
    }
    Ok(())
}

pub fn write_geom_hi_pngs(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<GeomHiPngPaths, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let geom_x_hi = output_dir.join("geom_x_hi.png");
    let geom_y_hi = output_dir.join("geom_y_hi.png");
    let geom_z_hi = output_dir.join("geom_z_hi.png");

    let (geom_x_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_z);

    write_lossless_u8_png(&geom_x_hi, packed.width, packed.height, &geom_x_hi_bytes)?;
    write_lossless_u8_png(&geom_y_hi, packed.width, packed.height, &geom_y_hi_bytes)?;
    write_lossless_u8_png(&geom_z_hi, packed.width, packed.height, &geom_z_hi_bytes)?;

    Ok(GeomHiPngPaths {
        geom_x_hi,
        geom_y_hi,
        geom_z_hi,
    })
}

pub fn read_u8_png(
    path: impl AsRef<Path>,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path.as_ref())?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if info.width != expected_width || info.height != expected_height {
        return Err(format!(
            "png dimensions mismatch: expected {}x{}, got {}x{}",
            expected_width, expected_height, info.width, info.height
        )
        .into());
    }
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err("expected 8-bit grayscale png".into());
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf)?;
    Ok(buf[..frame.buffer_size()].to_vec())
}

pub fn write_geom_hi_rgb_png(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<GeomHiRgbPngPath, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let geom_hi_rgb = output_dir.join("geom_hi_rgb.png");

    let (geom_x_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_z);

    let mut rgb = Vec::with_capacity(geom_x_hi_bytes.len() * 3);
    for i in 0..geom_x_hi_bytes.len() {
        rgb.push(geom_x_hi_bytes[i]);
        rgb.push(geom_y_hi_bytes[i]);
        rgb.push(geom_z_hi_bytes[i]);
    }

    write_rgb_png(&geom_hi_rgb, packed.width, packed.height, &rgb)?;
    Ok(GeomHiRgbPngPath { geom_hi_rgb })
}

pub fn write_sh_dc_hi_rgb_png(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<ShDcHiRgbPngPath, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let sh_dc_hi_rgb = output_dir.join("sh_dc_hi_rgb.png");

    let (sh_dc_r_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (sh_dc_g_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (sh_dc_b_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    write_rgb_png_from_planes(
        &sh_dc_hi_rgb,
        packed.width,
        packed.height,
        &sh_dc_r_hi_bytes,
        &sh_dc_g_hi_bytes,
        &sh_dc_b_hi_bytes,
    )?;

    Ok(ShDcHiRgbPngPath { sh_dc_hi_rgb })
}

pub fn sh_rest_streams_from_packed(packed: &PackedRasterScene) -> Vec<Vec<u16>> {
    let gaussian_len = packed.num_gaussians as usize;
    (0..packed.sh_rest_len as usize)
        .map(|stream_idx| {
            let mut plane = vec![0u16; (packed.width as usize) * (packed.height as usize)];
            for gaussian_idx in 0..gaussian_len {
                plane[gaussian_idx] = sh_rest_value_at(
                    &packed.sh_rest_payload,
                    stream_idx,
                    gaussian_idx,
                    gaussian_len,
                );
            }
            plane
        })
        .collect()
}

pub fn packed_scene_with_sh_rest_streams(
    packed: &PackedRasterScene,
    streams: &[Vec<u16>],
) -> Result<PackedRasterScene, Box<dyn std::error::Error>> {
    let gaussian_len = packed.num_gaussians as usize;
    let sh_rest_len = packed.sh_rest_len as usize;
    if streams.len() != sh_rest_len {
        return Err(format!(
            "sh_rest stream count mismatch: expected {sh_rest_len}, got {}",
            streams.len()
        )
        .into());
    }
    let plane_len = (packed.width as usize) * (packed.height as usize);
    let mut payload = vec![0u8; gaussian_len * sh_rest_len * std::mem::size_of::<u16>()];
    for (stream_idx, stream) in streams.iter().enumerate() {
        if stream.len() != plane_len {
            return Err(format!(
                "sh_rest plane length mismatch for stream {stream_idx}: expected {plane_len}, got {}",
                stream.len()
            )
            .into());
        }
        for gaussian_idx in 0..gaussian_len {
            let bytes = stream[gaussian_idx].to_le_bytes();
            let offset = (stream_idx * gaussian_len + gaussian_idx) * std::mem::size_of::<u16>();
            payload[offset] = bytes[0];
            payload[offset + 1] = bytes[1];
        }
    }

    let mut rebuilt = packed.clone();
    rebuilt.sh_rest_payload = payload;
    Ok(rebuilt)
}

pub fn read_rgb_png(
    path: impl AsRef<Path>,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path.as_ref())?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if info.width != expected_width || info.height != expected_height {
        return Err(format!(
            "png dimensions mismatch: expected {}x{}, got {}x{}",
            expected_width, expected_height, info.width, info.height
        )
        .into());
    }
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
        return Err("expected 8-bit rgb png".into());
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf)?;
    Ok(buf[..frame.buffer_size()].to_vec())
}

pub fn load_geom_hi_pngs(
    png_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    Ok([
        read_u8_png(png_dir.join("geom_x_hi.png"), width, height)?,
        read_u8_png(png_dir.join("geom_y_hi.png"), width, height)?,
        read_u8_png(png_dir.join("geom_z_hi.png"), width, height)?,
    ])
}

pub fn load_geom_hi_rgb_png(
    png_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    read_rgb_png_to_planes(png_dir.as_ref().join("geom_hi_rgb.png"), width, height)
}

pub fn read_rgb_png_to_planes(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    let rgb = read_rgb_png(path, width, height)?;
    let pixel_count = (width as usize) * (height as usize);
    if rgb.len() != pixel_count * 3 {
        return Err(format!(
            "rgb png byte length mismatch: expected {}, got {}",
            pixel_count * 3,
            rgb.len()
        )
        .into());
    }

    let mut x = Vec::with_capacity(pixel_count);
    let mut y = Vec::with_capacity(pixel_count);
    let mut z = Vec::with_capacity(pixel_count);
    for chunk in rgb.chunks_exact(3) {
        x.push(chunk[0]);
        y.push(chunk[1]);
        z.push(chunk[2]);
    }
    Ok([x, y, z])
}

pub fn write_rgb_png_from_planes(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    r: &[u8],
    g: &[u8],
    b: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(r.len(), g.len(), "rgb plane lengths must match");
    assert_eq!(r.len(), b.len(), "rgb plane lengths must match");

    let mut rgb = Vec::with_capacity(r.len() * 3);
    for i in 0..r.len() {
        rgb.push(r[i]);
        rgb.push(g[i]);
        rgb.push(b[i]);
    }

    write_rgb_png(path, width, height, &rgb)
}

pub fn recombine_bytes_to_u16(hi: &[u8], lo: &[u8]) -> Vec<u16> {
    assert_eq!(hi.len(), lo.len(), "hi/lo planes must have equal length");
    hi.iter()
        .zip(lo.iter())
        .map(|(&hi_byte, &lo_byte)| u16::from_le_bytes([lo_byte, hi_byte]))
        .collect()
}

fn write_sh_rest_video_atlas(
    packed: &PackedRasterScene,
    output_dir: &Path,
    codec_config: ShRestVideoAtlasCodecConfig,
) -> Result<CodedGroupWriteTiming, Box<dyn std::error::Error>> {
    if packed.sh_rest_len == 0 {
        std::fs::write(output_dir.join(SH_REST_HI_VIDEO_FILE), [])?;
        std::fs::write(output_dir.join(SH_REST_LO_VIDEO_FILE), [])?;
        return Ok(CodedGroupWriteTiming {
            group_name: "sh_rest".to_string(),
            ffmpeg_encode: Duration::ZERO,
            temp_file_io: Duration::ZERO,
        });
    }

    let streams = (0..packed.sh_rest_len as usize)
        .map(|stream_idx| {
            let gaussian_len = packed.num_gaussians as usize;
            let mut plane = vec![0u16; (packed.width as usize) * (packed.height as usize)];
            for gaussian_idx in 0..gaussian_len {
                plane[gaussian_idx] = sh_rest_value_at(
                    &packed.sh_rest_payload,
                    stream_idx,
                    gaussian_idx,
                    gaussian_len,
                );
            }
            plane
        })
        .collect::<Vec<_>>();
    write_multi_u16_video_atlas(
        &streams,
        packed.width,
        packed.height,
        packed.num_gaussians as usize,
        output_dir,
        [SH_REST_HI_VIDEO_FILE, SH_REST_LO_VIDEO_FILE],
        "sh_rest",
        codec_config,
    )
}

pub fn write_sh_rest_video_atlas_streams(
    streams: &[Vec<u16>],
    width: u32,
    height: u32,
    active_len: usize,
    output_dir: impl AsRef<Path>,
    codec_config: ShRestVideoAtlasCodecConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    write_multi_u16_video_atlas(
        streams,
        width,
        height,
        active_len,
        output_dir,
        [SH_REST_HI_VIDEO_FILE, SH_REST_LO_VIDEO_FILE],
        "sh_rest_streams",
        codec_config,
    )
    .map(|_| ())
}

pub fn read_sh_rest_video_atlas_streams(
    input_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
    stream_count: usize,
) -> Result<Vec<Vec<u16>>, Box<dyn std::error::Error>> {
    read_multi_u16_video_atlas(
        input_dir.as_ref(),
        width,
        height,
        stream_count,
        [SH_REST_HI_VIDEO_FILE, SH_REST_LO_VIDEO_FILE],
        "vpcc_sh_rest_streams",
    )
}

pub fn write_sh_rest_tiered_video_atlas_streams(
    streams: &[Vec<u16>],
    width: u32,
    height: u32,
    active_len: usize,
    output_dir: impl AsRef<Path>,
    tiers: &[ShRestTierSpec],
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    for (tier_idx, tier) in tiers.iter().enumerate() {
        if tier.start_stream >= tier.end_stream || tier.end_stream > streams.len() {
            return Err(format!(
                "invalid sh_rest tier {} range [{}..{}) for {} streams",
                tier.label,
                tier.start_stream,
                tier.end_stream,
                streams.len()
            )
            .into());
        }
        let hi_name = format!("sh_rest_tier_{tier_idx}_hi.mkv");
        let lo_name = format!("sh_rest_tier_{tier_idx}_lo.mkv");
        let temp_prefix = format!("vpcc_sh_rest_tier_{tier_idx}");
        write_multi_u16_video_atlas(
            &streams[tier.start_stream..tier.end_stream],
            width,
            height,
            active_len,
            output_dir,
            [&hi_name, &lo_name],
            &temp_prefix,
            tier.codec_config,
        )?;
    }
    Ok(())
}

pub fn read_sh_rest_tiered_video_atlas_streams(
    input_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
    tiers: &[ShRestTierSpec],
) -> Result<Vec<Vec<u16>>, Box<dyn std::error::Error>> {
    let input_dir = input_dir.as_ref();
    let mut streams = Vec::new();
    for (tier_idx, tier) in tiers.iter().enumerate() {
        let hi_name = format!("sh_rest_tier_{tier_idx}_hi.mkv");
        let lo_name = format!("sh_rest_tier_{tier_idx}_lo.mkv");
        let temp_prefix = format!("vpcc_sh_rest_tier_{tier_idx}");
        let tier_streams = read_multi_u16_video_atlas(
            input_dir,
            width,
            height,
            tier.end_stream - tier.start_stream,
            [&hi_name, &lo_name],
            &temp_prefix,
        )?;
        streams.extend(tier_streams);
    }
    Ok(streams)
}

fn sh_rest_encode_args(
    input_pattern: &str,
    output_path: &str,
    codec_config: ShRestVideoAtlasCodecConfig,
) -> Vec<String> {
    let mut args = vec![
        "-y".to_string(),
        "-framerate".to_string(),
        "1".to_string(),
        "-i".to_string(),
        input_pattern.to_string(),
        "-c:v".to_string(),
        codec_config.codec.to_string(),
        "-pix_fmt".to_string(),
        codec_config.pixel_format.to_string(),
    ];
    if let Some(ffmpeg_preset) = codec_config.ffmpeg_preset {
        args.push("-preset".to_string());
        args.push(ffmpeg_preset.to_string());
    }
    if let Some(crf) = codec_config.crf {
        args.push("-crf".to_string());
        args.push(crf.to_string());
    }
    args.push(output_path.to_string());
    args
}

fn read_sh_rest_video_atlas(
    input_dir: &Path,
    width: u32,
    height: u32,
    gaussian_len: usize,
    sh_rest_len: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if sh_rest_len == 0 {
        return Ok(Vec::new());
    }
    let planes = read_multi_u16_video_atlas(
        input_dir,
        width,
        height,
        sh_rest_len,
        [SH_REST_HI_VIDEO_FILE, SH_REST_LO_VIDEO_FILE],
        "vpcc_sh_rest",
    )?;
    let mut payload = vec![0u8; gaussian_len * sh_rest_len * std::mem::size_of::<u16>()];
    for (stream_idx, plane) in planes.iter().enumerate() {
        for gaussian_idx in 0..gaussian_len {
            let bytes = plane[gaussian_idx].to_le_bytes();
            let offset = (stream_idx * gaussian_len + gaussian_idx) * std::mem::size_of::<u16>();
            payload[offset] = bytes[0];
            payload[offset + 1] = bytes[1];
        }
    }
    Ok(payload)
}

fn write_u16_video_atlas(
    planes: [&[u16]; 3],
    width: u32,
    height: u32,
    active_len: usize,
    output_dir: &Path,
    video_files: [&str; 2],
    temp_prefix: &str,
    codec_config: ShRestVideoAtlasCodecConfig,
) -> Result<CodedGroupWriteTiming, Box<dyn std::error::Error>> {
    let owned = planes
        .iter()
        .map(|plane| plane.to_vec())
        .collect::<Vec<_>>();
    write_multi_u16_video_atlas(
        &owned,
        width,
        height,
        active_len,
        output_dir,
        video_files,
        temp_prefix,
        codec_config,
    )
}

fn write_multi_u16_video_atlas(
    streams: &[Vec<u16>],
    width: u32,
    height: u32,
    active_len: usize,
    output_dir: &Path,
    video_files: [&str; 2],
    temp_prefix: &str,
    codec_config: ShRestVideoAtlasCodecConfig,
) -> Result<CodedGroupWriteTiming, Box<dyn std::error::Error>> {
    let frame_count = streams.len().div_ceil(3);
    let plane_len = (width as usize) * (height as usize);
    let hi_dir = unique_temp_dir(&format!("{temp_prefix}_hi_encode"));
    let lo_dir = unique_temp_dir(&format!("{temp_prefix}_lo_encode"));
    let temp_file_io_start = Instant::now();
    fs::create_dir_all(&hi_dir)?;
    fs::create_dir_all(&lo_dir)?;

    let mut hi_planes = [
        vec![0u8; plane_len],
        vec![0u8; plane_len],
        vec![0u8; plane_len],
    ];
    let mut lo_planes = [
        vec![0u8; plane_len],
        vec![0u8; plane_len],
        vec![0u8; plane_len],
    ];
    for frame_idx in 0..frame_count {
        for channel in 0..3 {
            hi_planes[channel].fill(0);
            lo_planes[channel].fill(0);
        }
        for channel in 0..3 {
            let stream_idx = frame_idx * 3 + channel;
            if stream_idx >= streams.len() {
                continue;
            }
            for idx in 0..active_len {
                let [lo_byte, hi_byte] = streams[stream_idx][idx].to_le_bytes();
                hi_planes[channel][idx] = hi_byte;
                lo_planes[channel][idx] = lo_byte;
            }
        }
        let frame_name = format!("frame_{:03}.png", frame_idx + 1);
        write_rgb_png_from_planes(
            hi_dir.join(&frame_name),
            width,
            height,
            &hi_planes[0],
            &hi_planes[1],
            &hi_planes[2],
        )?;
        write_rgb_png_from_planes(
            lo_dir.join(&frame_name),
            width,
            height,
            &lo_planes[0],
            &lo_planes[1],
            &lo_planes[2],
        )?;
    }

    let mut ffmpeg_encode = Duration::ZERO;
    let encode_args = sh_rest_encode_args(
        &hi_dir.join("frame_%03d.png").display().to_string(),
        &output_dir.join(video_files[0]).display().to_string(),
        codec_config,
    );
    let ffmpeg_start = Instant::now();
    run_ffmpeg(&encode_args)?;
    ffmpeg_encode += ffmpeg_start.elapsed();
    let encode_args = sh_rest_encode_args(
        &lo_dir.join("frame_%03d.png").display().to_string(),
        &output_dir.join(video_files[1]).display().to_string(),
        codec_config,
    );
    let ffmpeg_start = Instant::now();
    run_ffmpeg(&encode_args)?;
    ffmpeg_encode += ffmpeg_start.elapsed();
    fs::remove_dir_all(hi_dir)?;
    fs::remove_dir_all(lo_dir)?;
    Ok(CodedGroupWriteTiming {
        group_name: temp_prefix.to_string(),
        ffmpeg_encode,
        temp_file_io: temp_file_io_start.elapsed().saturating_sub(ffmpeg_encode),
    })
}

fn read_u16_video_atlas(
    input_dir: &Path,
    width: u32,
    height: u32,
    plane_len: usize,
    video_files: [&str; 2],
    temp_prefix: &str,
) -> Result<[Vec<u16>; 3], Box<dyn std::error::Error>> {
    let streams =
        read_multi_u16_video_atlas(input_dir, width, height, 3, video_files, temp_prefix)?;
    Ok([
        streams[0][..plane_len].to_vec(),
        streams[1][..plane_len].to_vec(),
        streams[2][..plane_len].to_vec(),
    ])
}

fn read_multi_u16_video_atlas(
    input_dir: &Path,
    width: u32,
    height: u32,
    stream_count: usize,
    video_files: [&str; 2],
    temp_prefix: &str,
) -> Result<Vec<Vec<u16>>, Box<dyn std::error::Error>> {
    let frame_count = stream_count.div_ceil(3);
    let plane_len = (width as usize) * (height as usize);
    let hi_dir = unique_temp_dir(&format!("{temp_prefix}_hi_decode"));
    let lo_dir = unique_temp_dir(&format!("{temp_prefix}_lo_decode"));
    fs::create_dir_all(&hi_dir)?;
    fs::create_dir_all(&lo_dir)?;
    let decode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        input_dir.join(video_files[0]).display().to_string(),
        "-start_number".to_string(),
        "1".to_string(),
        hi_dir.join("frame_%03d.png").display().to_string(),
    ];
    run_ffmpeg(&decode_args)?;
    let decode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        input_dir.join(video_files[1]).display().to_string(),
        "-start_number".to_string(),
        "1".to_string(),
        lo_dir.join("frame_%03d.png").display().to_string(),
    ];
    run_ffmpeg(&decode_args)?;
    let mut streams = vec![vec![0u16; plane_len]; stream_count];
    for frame_idx in 0..frame_count {
        let frame_name = format!("frame_{:03}.png", frame_idx + 1);
        let hi_planes = read_rgb_png_to_planes(hi_dir.join(&frame_name), width, height)?;
        let lo_planes = read_rgb_png_to_planes(lo_dir.join(&frame_name), width, height)?;
        for channel in 0..3 {
            let stream_idx = frame_idx * 3 + channel;
            if stream_idx >= stream_count {
                continue;
            }
            for idx in 0..plane_len {
                streams[stream_idx][idx] =
                    u16::from_le_bytes([lo_planes[channel][idx], hi_planes[channel][idx]]);
            }
        }
    }
    fs::remove_dir_all(hi_dir)?;
    fs::remove_dir_all(lo_dir)?;
    Ok(streams)
}

fn sh_rest_value_at(
    payload: &[u8],
    stream_idx: usize,
    gaussian_idx: usize,
    gaussian_len: usize,
) -> u16 {
    let offset = (stream_idx * gaussian_len + gaussian_idx) * std::mem::size_of::<u16>();
    u16::from_le_bytes([payload[offset], payload[offset + 1]])
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{}_{}_{}",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn run_ffmpeg(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("ffmpeg").args(args).output().map_err(|err| {
        format!("failed to launch ffmpeg; ensure it is installed and on PATH: {err}")
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg command failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into())
    }
}

pub fn rebuild_packed_scene_with_decoded_geom_hi(
    packed: &PackedRasterScene,
    decoded_geom_hi: [&[u8]; 3],
) -> PackedRasterScene {
    let (_, geom_x_lo) = split_u16_plane_to_bytes(&packed.geom_x);
    let (_, geom_y_lo) = split_u16_plane_to_bytes(&packed.geom_y);
    let (_, geom_z_lo) = split_u16_plane_to_bytes(&packed.geom_z);

    let mut rebuilt = packed.clone();
    rebuilt.geom_x = recombine_bytes_to_u16(decoded_geom_hi[0], &geom_x_lo);
    rebuilt.geom_y = recombine_bytes_to_u16(decoded_geom_hi[1], &geom_y_lo);
    rebuilt.geom_z = recombine_bytes_to_u16(decoded_geom_hi[2], &geom_z_lo);
    rebuilt
}

pub fn rebuild_packed_scene_with_decoded_sh_dc_hi(
    packed: &PackedRasterScene,
    decoded_sh_dc_hi: [&[u8]; 3],
) -> PackedRasterScene {
    let (_, sh_dc_r_lo) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (_, sh_dc_g_lo) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (_, sh_dc_b_lo) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    let mut rebuilt = packed.clone();
    rebuilt.sh_dc_r = recombine_bytes_to_u16(decoded_sh_dc_hi[0], &sh_dc_r_lo);
    rebuilt.sh_dc_g = recombine_bytes_to_u16(decoded_sh_dc_hi[1], &sh_dc_g_lo);
    rebuilt.sh_dc_b = recombine_bytes_to_u16(decoded_sh_dc_hi[2], &sh_dc_b_lo);
    rebuilt
}

pub fn evaluate_geom_hi_png_roundtrip(
    original_sorted: &[Gaussian],
    packed: &PackedRasterScene,
    png_dir: impl AsRef<Path>,
) -> Result<GeomHiPngExperimentResult, Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    let decoded_hi = load_geom_hi_pngs(png_dir, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let xyz_rmse = [
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[0]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[1]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[2]),
    ];

    let x_size = fs::metadata(png_dir.join("geom_x_hi.png"))?.len();
    let y_size = fs::metadata(png_dir.join("geom_y_hi.png"))?.len();
    let z_size = fs::metadata(png_dir.join("geom_z_hi.png"))?.len();
    let image_sizes = [x_size, y_size, z_size];
    let total_coded_bytes = image_sizes.iter().sum();

    Ok(GeomHiPngExperimentResult {
        image_sizes,
        total_coded_bytes,
        xyz_rmse,
    })
}

pub fn evaluate_geom_hi_rgb_png_roundtrip(
    original_sorted: &[Gaussian],
    packed: &PackedRasterScene,
    png_dir: impl AsRef<Path>,
) -> Result<GeomHiPngExperimentResult, Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    let decoded_hi = load_geom_hi_rgb_png(png_dir, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let xyz_rmse = [
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[0]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[1]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[2]),
    ];

    let rgb_size = fs::metadata(png_dir.join("geom_hi_rgb.png"))?.len();
    Ok(GeomHiPngExperimentResult {
        image_sizes: [rgb_size, 0, 0],
        total_coded_bytes: rgb_size,
        xyz_rmse,
    })
}

pub fn compute_roundtrip_diagnostics(
    original_sorted: &[Gaussian],
    reconstructed: &[Gaussian],
    packed: &PackedRasterScene,
) -> RoundtripDiagnostics {
    assert_eq!(
        original_sorted.len(),
        reconstructed.len(),
        "roundtrip diagnostics require equal-length gaussian arrays"
    );

    RoundtripDiagnostics {
        xyz_rmse: [
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[0]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[1]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[2]),
        ],
        sh_dc_rmse: [
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[0]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[1]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[2]),
        ],
        original_neighbor_deltas: compute_neighbor_delta_stats(original_sorted),
        reconstructed_neighbor_deltas: compute_neighbor_delta_stats(reconstructed),
        plane_stats: compute_plane_stats(packed),
        byte_split_plane_stats: compute_byte_split_plane_stats(packed),
    }
}

pub fn compute_neighbor_delta_stats(gaussians: &[Gaussian]) -> NeighborDeltaStats {
    NeighborDeltaStats {
        xyz: [
            delta_stats(gaussians, |g| g.xyz[0]),
            delta_stats(gaussians, |g| g.xyz[1]),
            delta_stats(gaussians, |g| g.xyz[2]),
        ],
        sh_dc: [
            delta_stats(gaussians, |g| g.sh_dc[0]),
            delta_stats(gaussians, |g| g.sh_dc[1]),
            delta_stats(gaussians, |g| g.sh_dc[2]),
        ],
    }
}

pub fn sort_gaussians_by_morton(scene: &Scene) -> Vec<Gaussian> {
    let mut sorted = scene.gaussians.clone();
    sorted.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));
    sorted
}

pub fn compute_plane_stats(packed: &PackedRasterScene) -> Vec<PlaneStats> {
    vec![
        plane_stats_u8(
            "occupancy",
            &packed.occupancy,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "geom_x",
            &packed.geom_x,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "geom_y",
            &packed.geom_y,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "geom_z",
            &packed.geom_z,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "sh_dc_r",
            &packed.sh_dc_r,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "sh_dc_g",
            &packed.sh_dc_g,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
        plane_stats_u16(
            "sh_dc_b",
            &packed.sh_dc_b,
            &packed.occupancy,
            packed.width,
            packed.height,
        ),
    ]
}

pub fn split_u16_plane_to_bytes(data: &[u16]) -> (Vec<u8>, Vec<u8>) {
    let mut hi = Vec::with_capacity(data.len());
    let mut lo = Vec::with_capacity(data.len());
    for &value in data {
        let [lo_byte, hi_byte] = value.to_le_bytes();
        hi.push(hi_byte);
        lo.push(lo_byte);
    }
    (hi, lo)
}

fn join_u16_plane_from_bytes(hi: &[u8], lo: &[u8]) -> std::io::Result<Vec<u16>> {
    if hi.len() != lo.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "byte-split length mismatch: hi={} lo={}",
                hi.len(),
                lo.len()
            ),
        ));
    }
    Ok(hi
        .iter()
        .zip(lo.iter())
        .map(|(&hi_byte, &lo_byte)| u16::from_le_bytes([lo_byte, hi_byte]))
        .collect())
}

fn write_u16_byte_split_deflate(
    output_dir: &Path,
    planes: [&[u16]; 3],
    files: [&str; 6],
) -> Result<(), Box<dyn std::error::Error>> {
    let level = 1;
    for (plane, [hi_file, lo_file]) in planes.into_iter().zip([
        [files[0], files[1]],
        [files[2], files[3]],
        [files[4], files[5]],
    ]) {
        let (hi, lo) = split_u16_plane_to_bytes(plane);
        let hi_payload = compress_u8_vec_with_level(hi, level)?;
        let lo_payload = compress_u8_vec_with_level(lo, level)?;
        write_u8_plane(output_dir.join(hi_file), &hi_payload)?;
        write_u8_plane(output_dir.join(lo_file), &lo_payload)?;
    }
    Ok(())
}

fn read_u16_byte_split_deflate(
    input_dir: &Path,
    plane_len: usize,
    files: [&str; 6],
) -> Result<[Vec<u16>; 3], Box<dyn std::error::Error>> {
    let decode_plane =
        |hi_file: &str, lo_file: &str| -> Result<Vec<u16>, Box<dyn std::error::Error>> {
            let hi = decompress_u8_vec(&read_u8_blob(input_dir.join(hi_file))?)?;
            let lo = decompress_u8_vec(&read_u8_blob(input_dir.join(lo_file))?)?;
            if hi.len() != plane_len || lo.len() != plane_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "byte-split decoded length mismatch: expected {plane_len}, got hi={} lo={}",
                        hi.len(),
                        lo.len()
                    ),
                )
                .into());
            }
            Ok(join_u16_plane_from_bytes(&hi, &lo)?)
        };
    Ok([
        decode_plane(files[0], files[1])?,
        decode_plane(files[2], files[3])?,
        decode_plane(files[4], files[5])?,
    ])
}

pub fn split_packed_u16_planes(packed: &PackedRasterScene) -> Vec<ByteSplitPlane> {
    let (geom_x_hi, geom_x_lo) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi, geom_y_lo) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi, geom_z_lo) = split_u16_plane_to_bytes(&packed.geom_z);
    let (sh_dc_r_hi, sh_dc_r_lo) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (sh_dc_g_hi, sh_dc_g_lo) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (sh_dc_b_hi, sh_dc_b_lo) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    vec![
        ByteSplitPlane {
            name: "geom_x",
            hi: geom_x_hi,
            lo: geom_x_lo,
        },
        ByteSplitPlane {
            name: "geom_y",
            hi: geom_y_hi,
            lo: geom_y_lo,
        },
        ByteSplitPlane {
            name: "geom_z",
            hi: geom_z_hi,
            lo: geom_z_lo,
        },
        ByteSplitPlane {
            name: "sh_dc_r",
            hi: sh_dc_r_hi,
            lo: sh_dc_r_lo,
        },
        ByteSplitPlane {
            name: "sh_dc_g",
            hi: sh_dc_g_hi,
            lo: sh_dc_g_lo,
        },
        ByteSplitPlane {
            name: "sh_dc_b",
            hi: sh_dc_b_hi,
            lo: sh_dc_b_lo,
        },
    ]
}

pub fn compute_byte_split_plane_stats(packed: &PackedRasterScene) -> Vec<ByteSplitPlaneStats> {
    let byte_split = split_packed_u16_planes(packed);
    let base_stats = compute_plane_stats(packed);

    byte_split
        .into_iter()
        .zip(base_stats.into_iter().skip(1))
        .map(|(plane, base_u16)| ByteSplitPlaneStats {
            name: plane.name,
            base_u16,
            hi: plane_stats_u8(
                match plane.name {
                    "geom_x" => "geom_x_hi",
                    "geom_y" => "geom_y_hi",
                    "geom_z" => "geom_z_hi",
                    "sh_dc_r" => "sh_dc_r_hi",
                    "sh_dc_g" => "sh_dc_g_hi",
                    "sh_dc_b" => "sh_dc_b_hi",
                    _ => "unknown_hi",
                },
                &plane.hi,
                &packed.occupancy,
                packed.width,
                packed.height,
            ),
            lo: plane_stats_u8(
                match plane.name {
                    "geom_x" => "geom_x_lo",
                    "geom_y" => "geom_y_lo",
                    "geom_z" => "geom_z_lo",
                    "sh_dc_r" => "sh_dc_r_lo",
                    "sh_dc_g" => "sh_dc_g_lo",
                    "sh_dc_b" => "sh_dc_b_lo",
                    _ => "unknown_lo",
                },
                &plane.lo,
                &packed.occupancy,
                packed.width,
                packed.height,
            ),
        })
        .collect()
}

fn write_u8_plane(path: impl AsRef<Path>, data: &[u8]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    writer.write_all(data)?;
    writer.flush()
}

fn write_u16_plane(path: impl AsRef<Path>, data: &[u16]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    for value in data {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()
}

fn read_u8_plane(path: impl AsRef<Path>, expected_len: usize) -> std::io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    if data.len() != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "u8 plane length mismatch: expected {expected_len}, got {}",
                data.len()
            ),
        ));
    }
    Ok(data)
}

fn read_u8_blob(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    Ok(data)
}

fn read_u16_plane(path: impl AsRef<Path>, expected_len: usize) -> std::io::Result<Vec<u16>> {
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let expected_bytes = expected_len * std::mem::size_of::<u16>();
    if bytes.len() != expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "u16 plane byte length mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            ),
        ));
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn encode_sh_rest_payload(
    gaussians: &[Gaussian],
    sh_rest_len: usize,
) -> Result<(Vec<QuantParams>, Vec<u8>, ShRestPackTimings), Box<dyn std::error::Error>> {
    if sh_rest_len == 0 || gaussians.is_empty() {
        return Ok((
            Vec::new(),
            Vec::new(),
            ShRestPackTimings {
                quantization: Duration::ZERO,
                buffer_construction: Duration::ZERO,
                atlas_construction: Duration::ZERO,
            },
        ));
    }

    let gaussian_len = gaussians.len();
    let quantization_start = Instant::now();
    let mut quant_params = vec![QuantParams { min: 0.0, max: 0.0 }; sh_rest_len];
    for (gaussian_idx, gaussian) in gaussians.iter().enumerate() {
        assert_eq!(
            gaussian.sh_rest.len(),
            sh_rest_len,
            "inconsistent sh_rest stream count at gaussian {gaussian_idx}"
        );
        if gaussian_idx == 0 {
            for (stream_idx, &value) in gaussian.sh_rest.iter().enumerate() {
                quant_params[stream_idx] = QuantParams {
                    min: value,
                    max: value,
                };
            }
        } else {
            for (stream_idx, &value) in gaussian.sh_rest.iter().enumerate() {
                update_quant_bounds(&mut quant_params[stream_idx], value);
            }
        }
    }
    let quantization = quantization_start.elapsed();

    let buffer_start = Instant::now();
    let mut quantized_payload = vec![0u8; gaussian_len * sh_rest_len * std::mem::size_of::<u16>()];
    let bytes_per_stream = gaussian_len * std::mem::size_of::<u16>();
    let worker_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(sh_rest_len)
        .max(1);
    let streams_per_worker = sh_rest_len.div_ceil(worker_count);
    thread::scope(|scope| {
        let mut remaining_payload = quantized_payload.as_mut_slice();
        for worker_idx in 0..worker_count {
            let start_stream = worker_idx * streams_per_worker;
            if start_stream >= sh_rest_len {
                break;
            }
            let end_stream = (start_stream + streams_per_worker).min(sh_rest_len);
            let chunk_bytes = (end_stream - start_stream) * bytes_per_stream;
            let (payload_chunk, rest) = remaining_payload.split_at_mut(chunk_bytes);
            remaining_payload = rest;
            let quant_chunk = &quant_params[start_stream..end_stream];
            scope.spawn(move || {
                for (relative_stream_idx, params) in quant_chunk.iter().copied().enumerate() {
                    let stream_idx = start_stream + relative_stream_idx;
                    let stream_payload = &mut payload_chunk[relative_stream_idx * bytes_per_stream
                        ..(relative_stream_idx + 1) * bytes_per_stream];
                    for (gaussian_idx, gaussian) in gaussians.iter().enumerate() {
                        let quantized =
                            quantize_to_u16(gaussian.sh_rest[stream_idx], params).to_le_bytes();
                        let byte_offset = gaussian_idx * std::mem::size_of::<u16>();
                        stream_payload[byte_offset] = quantized[0];
                        stream_payload[byte_offset + 1] = quantized[1];
                    }
                }
            });
        }
    });
    let buffer_construction = buffer_start.elapsed();
    let atlas_construction = Duration::ZERO;
    Ok((
        quant_params,
        quantized_payload,
        ShRestPackTimings {
            quantization,
            buffer_construction,
            atlas_construction,
        },
    ))
}

fn decode_sh_rest_payload(
    payload: &[u8],
    quant_params: &[QuantParams],
    gaussian_len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let sh_rest_len = quant_params.len();
    if sh_rest_len == 0 || gaussian_len == 0 {
        return Ok(Vec::new());
    }

    let expected_bytes = gaussian_len * sh_rest_len * std::mem::size_of::<u16>();
    if payload.len() != expected_bytes {
        return Err(format!(
            "sh_rest quantized byte length mismatch: expected {}, got {}",
            expected_bytes,
            payload.len()
        )
        .into());
    }

    let mut decoded = vec![0.0f32; gaussian_len * sh_rest_len];
    for stream_idx in 0..sh_rest_len {
        let params = quant_params[stream_idx];
        for gaussian_idx in 0..gaussian_len {
            let offset = (stream_idx * gaussian_len + gaussian_idx) * std::mem::size_of::<u16>();
            let quantized = u16::from_le_bytes([payload[offset], payload[offset + 1]]);
            decoded[gaussian_idx * sh_rest_len + stream_idx] =
                dequantize_from_u16(quantized, params);
        }
    }

    Ok(decoded)
}

fn compute_quant_params(gaussians: &[Gaussian]) -> ([QuantParams; 3], [QuantParams; 3]) {
    let mut geom = [
        QuantParams { min: 0.0, max: 0.0 },
        QuantParams { min: 0.0, max: 0.0 },
        QuantParams { min: 0.0, max: 0.0 },
    ];
    let mut sh_dc = geom;

    if let Some(first) = gaussians.first() {
        geom = [
            QuantParams {
                min: first.xyz[0],
                max: first.xyz[0],
            },
            QuantParams {
                min: first.xyz[1],
                max: first.xyz[1],
            },
            QuantParams {
                min: first.xyz[2],
                max: first.xyz[2],
            },
        ];
        sh_dc = [
            QuantParams {
                min: first.sh_dc[0],
                max: first.sh_dc[0],
            },
            QuantParams {
                min: first.sh_dc[1],
                max: first.sh_dc[1],
            },
            QuantParams {
                min: first.sh_dc[2],
                max: first.sh_dc[2],
            },
        ];
    }

    for gaussian in gaussians {
        update_quant_bounds(&mut geom[0], gaussian.xyz[0]);
        update_quant_bounds(&mut geom[1], gaussian.xyz[1]);
        update_quant_bounds(&mut geom[2], gaussian.xyz[2]);
        update_quant_bounds(&mut sh_dc[0], gaussian.sh_dc[0]);
        update_quant_bounds(&mut sh_dc[1], gaussian.sh_dc[1]);
        update_quant_bounds(&mut sh_dc[2], gaussian.sh_dc[2]);
    }

    (geom, sh_dc)
}

fn update_quant_bounds(params: &mut QuantParams, value: f32) {
    params.min = params.min.min(value);
    params.max = params.max.max(value);
}

fn quantize_to_u16(value: f32, params: QuantParams) -> u16 {
    if params.max <= params.min {
        return 0;
    }

    let normalized = ((value - params.min) / (params.max - params.min)).clamp(0.0, 1.0);
    (normalized * u16::MAX as f32).round() as u16
}

fn dequantize_from_u16(value: u16, params: QuantParams) -> f32 {
    if params.max <= params.min {
        return params.min;
    }

    params.min + (value as f32 / u16::MAX as f32) * (params.max - params.min)
}

fn compute_xyz_bounds(gaussians: &[Gaussian]) -> ([f32; 3], [f32; 3]) {
    let Some(first) = gaussians.first() else {
        return ([0.0; 3], [0.0; 3]);
    };

    let mut mins = first.xyz;
    let mut maxes = first.xyz;
    for gaussian in gaussians.iter().skip(1) {
        for axis in 0..3 {
            mins[axis] = mins[axis].min(gaussian.xyz[axis]);
            maxes[axis] = maxes[axis].max(gaussian.xyz[axis]);
        }
    }

    (mins, maxes)
}

fn rmse_for_channel(
    original: &[Gaussian],
    reconstructed: &[Gaussian],
    accessor: impl Fn(&Gaussian) -> f32,
) -> f64 {
    if original.is_empty() {
        return 0.0;
    }

    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| {
            let diff = accessor(lhs) as f64 - accessor(rhs) as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    mse.sqrt()
}

fn delta_stats(gaussians: &[Gaussian], accessor: impl Fn(&Gaussian) -> f32) -> ChannelDeltaStats {
    if gaussians.len() < 2 {
        return ChannelDeltaStats {
            count: 0,
            mean_abs: 0.0,
            std_abs: 0.0,
        };
    }

    let abs_deltas: Vec<f64> = gaussians
        .windows(2)
        .map(|pair| (accessor(&pair[1]) as f64 - accessor(&pair[0]) as f64).abs())
        .collect();

    let count = abs_deltas.len();
    let mean_abs = abs_deltas.iter().sum::<f64>() / count as f64;
    let variance = abs_deltas
        .iter()
        .map(|delta| {
            let centered = delta - mean_abs;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;

    ChannelDeltaStats {
        count,
        mean_abs,
        std_abs: variance.sqrt(),
    }
}

fn plane_stats_u8(
    name: &'static str,
    plane: &[u8],
    occupancy: &[u8],
    width: u32,
    height: u32,
) -> PlaneStats {
    let values: Vec<u32> = plane
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value as u32))
        .collect();
    PlaneStats {
        name,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        entropy_bits_per_symbol: entropy_bits(&values),
        neighbor_delta: delta_stats_from_u32(&values),
        horizontal_correlation: pixel_correlation_u8(plane, occupancy, width, height, true),
        vertical_correlation: pixel_correlation_u8(plane, occupancy, width, height, false),
    }
}

fn plane_stats_u16(
    name: &'static str,
    plane: &[u16],
    occupancy: &[u8],
    width: u32,
    height: u32,
) -> PlaneStats {
    let values: Vec<u32> = plane
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value as u32))
        .collect();
    PlaneStats {
        name,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        entropy_bits_per_symbol: entropy_bits(&values),
        neighbor_delta: delta_stats_from_u32(&values),
        horizontal_correlation: pixel_correlation_u16(plane, occupancy, width, height, true),
        vertical_correlation: pixel_correlation_u16(plane, occupancy, width, height, false),
    }
}

fn delta_stats_from_u32(values: &[u32]) -> ChannelDeltaStats {
    if values.len() < 2 {
        return ChannelDeltaStats {
            count: 0,
            mean_abs: 0.0,
            std_abs: 0.0,
        };
    }

    let abs_deltas: Vec<f64> = values
        .windows(2)
        .map(|pair| (pair[1] as f64 - pair[0] as f64).abs())
        .collect();
    let count = abs_deltas.len();
    let mean_abs = abs_deltas.iter().sum::<f64>() / count as f64;
    let variance = abs_deltas
        .iter()
        .map(|delta| {
            let centered = delta - mean_abs;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;

    ChannelDeltaStats {
        count,
        mean_abs,
        std_abs: variance.sqrt(),
    }
}

fn entropy_bits(values: &[u32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut histogram: HashMap<u32, usize> = HashMap::new();
    for &value in values {
        *histogram.entry(value).or_insert(0) += 1;
    }

    let total = values.len() as f64;
    histogram
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn pixel_correlation_u8(
    plane: &[u8],
    occupancy: &[u8],
    width: u32,
    height: u32,
    horizontal: bool,
) -> CorrelationStats {
    pixel_correlation(
        width,
        height,
        occupancy,
        |idx| plane[idx] as f64,
        horizontal,
    )
}

fn pixel_correlation_u16(
    plane: &[u16],
    occupancy: &[u8],
    width: u32,
    height: u32,
    horizontal: bool,
) -> CorrelationStats {
    pixel_correlation(
        width,
        height,
        occupancy,
        |idx| plane[idx] as f64,
        horizontal,
    )
}

fn pixel_correlation(
    width: u32,
    height: u32,
    occupancy: &[u8],
    value_at: impl Fn(usize) -> f64,
    horizontal: bool,
) -> CorrelationStats {
    let width = width as usize;
    let height = height as usize;
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let neighbor = if horizontal {
                (x + 1 < width).then_some(idx + 1)
            } else {
                (y + 1 < height).then_some(idx + width)
            };
            let Some(neighbor_idx) = neighbor else {
                continue;
            };
            if occupancy[idx] == 0 || occupancy[neighbor_idx] == 0 {
                continue;
            }
            xs.push(value_at(idx));
            ys.push(value_at(neighbor_idx));
        }
    }

    pearson_from_pairs(&xs, &ys)
}

fn pearson_from_pairs(xs: &[f64], ys: &[f64]) -> CorrelationStats {
    assert_eq!(
        xs.len(),
        ys.len(),
        "correlation inputs must match in length"
    );
    if xs.len() < 2 {
        return CorrelationStats {
            count: xs.len(),
            pearson_r: 0.0,
        };
    }

    let count = xs.len();
    let mean_x = xs.iter().sum::<f64>() / count as f64;
    let mean_y = ys.iter().sum::<f64>() / count as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    let pearson_r = if denom > 0.0 { cov / denom } else { 0.0 };
    CorrelationStats { count, pearson_r }
}

fn write_preview_u8_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u8],
    occupancy: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pixels = vec![0u8; data.len()];
    for (idx, &value) in data.iter().enumerate() {
        pixels[idx] = if occupancy[idx] != 0 {
            value.saturating_mul(255)
        } else {
            0
        };
    }
    write_grayscale_png(path, width, height, &pixels)
}

fn write_preview_u16_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u16],
    occupancy: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let occupied_values: Vec<u16> = data
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value))
        .collect();
    let min = occupied_values.iter().copied().min().unwrap_or(0);
    let max = occupied_values.iter().copied().max().unwrap_or(min);
    let range = max.saturating_sub(min);

    let mut pixels = vec![0u8; data.len()];
    for (idx, &value) in data.iter().enumerate() {
        if occupancy[idx] == 0 {
            continue;
        }
        pixels[idx] = if range == 0 {
            0
        } else {
            (((value.saturating_sub(min)) as f64 / range as f64) * 255.0).round() as u8
        };
    }

    write_grayscale_png(path, width, height, &pixels)
}

fn write_grayscale_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path.as_ref())?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(pixels)?;
    Ok(())
}

fn write_lossless_u8_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    write_grayscale_png(path, width, height, pixels)
}

fn write_rgb_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path.as_ref())?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(pixels)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_scene(num_gaussians: usize) -> Scene {
        let mut gaussians = Vec::with_capacity(num_gaussians);
        for i in 0..num_gaussians {
            let i_f = i as f32;
            gaussians.push(Gaussian {
                xyz: [
                    (i_f * 0.71).sin() * 8.0 + 3.0,
                    (i_f * 0.37).cos() * 5.0 - 1.5,
                    i_f * 0.125 + (i_f * 0.19).sin(),
                ],
                normals: Some([
                    (i_f * 0.05).sin(),
                    (i_f * 0.09).cos(),
                    (i_f * 0.03).sin() * 0.5,
                ]),
                sh_dc: [
                    (i_f * 0.11).sin() * 0.7,
                    (i_f * 0.07).cos() * 0.5,
                    ((i_f * 0.17).sin() + (i_f * 0.13).cos()) * 0.25,
                ],
                sh_rest: (0..45).map(|j| ((i + j) as f32 * 0.013).sin()).collect(),
                opacity: 0.5 + i_f * 0.001,
                scale: [i_f * 0.01, -i_f * 0.02, i_f * 0.03],
                rot: [0.1, 0.2, 0.3, 0.9],
            });
        }

        let (mins, maxes) = compute_xyz_bounds(&gaussians);
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "synthetic".to_string());
        meta.insert("vertex_count".to_string(), gaussians.len().to_string());

        Scene {
            gaussians,
            meta,
            mins,
            maxes,
        }
    }

    #[test]
    fn pack_scene_creates_expected_raster_shape() {
        let scene = synthetic_scene(17);
        let packed = pack_scene_to_raster(&scene, 4);

        assert_eq!(packed.num_gaussians, 17);
        assert_eq!(packed.width, 4);
        assert_eq!(packed.height, 5);
        assert_eq!(packed.occupancy.len(), 20);
        assert_eq!(packed.occupancy.iter().filter(|&&v| v == 1).count(), 17);
        assert_eq!(packed.sh_rest_len, 45);
    }

    #[test]
    fn pack_unpack_roundtrip_preserves_xyz_and_sh_dc_with_low_error() {
        let scene = synthetic_scene(64);
        let sorted = sort_gaussians_by_morton(&scene);
        let packed = pack_scene_to_raster(&scene, 8);
        let reconstructed = unpack_raster_to_scene(&packed);
        let diagnostics = compute_roundtrip_diagnostics(&sorted, &reconstructed.gaussians, &packed);

        for axis in diagnostics.xyz_rmse {
            assert!(axis < 0.0002, "xyz RMSE too high: {axis}");
        }
        for axis in diagnostics.sh_dc_rmse {
            assert!(axis < 0.00005, "sh_dc RMSE too high: {axis}");
        }

        assert_eq!(reconstructed.gaussians.len(), sorted.len());
        let sh_rest_max_abs = reconstructed.gaussians[0]
            .sh_rest
            .iter()
            .zip(sorted[0].sh_rest.iter())
            .map(|(recon, original)| (recon - original).abs())
            .fold(0.0f32, f32::max);
        assert!(
            sh_rest_max_abs < 0.0001,
            "sh_rest max abs error too high: {sh_rest_max_abs}"
        );
        assert_eq!(reconstructed.gaussians[0].normals, sorted[0].normals);
        assert_eq!(reconstructed.gaussians[0].opacity, sorted[0].opacity);
        assert_eq!(reconstructed.gaussians[0].scale, sorted[0].scale);
        assert_eq!(reconstructed.gaussians[0].rot, sorted[0].rot);
        let sh_rest_stats = sh_rest_payload_stats(&packed);
        assert_eq!(sh_rest_stats.storage, SH_REST_STORAGE_CODED_VIDEO_ATLAS);
        assert!(sh_rest_stats.coded_bytes > 0);
    }

    #[test]
    fn neighbor_delta_stats_match_expected_count() {
        let scene = synthetic_scene(32);
        let sorted = sort_gaussians_by_morton(&scene);
        let stats = compute_neighbor_delta_stats(&sorted);

        for channel in stats.xyz {
            assert_eq!(channel.count, 31);
            assert!(channel.mean_abs >= 0.0);
            assert!(channel.std_abs >= 0.0);
        }
        for channel in stats.sh_dc {
            assert_eq!(channel.count, 31);
            assert!(channel.mean_abs >= 0.0);
            assert!(channel.std_abs >= 0.0);
        }
    }

    #[test]
    fn empty_scene_roundtrip_is_supported() {
        let scene = synthetic_scene(0);
        let packed = pack_scene_to_raster(&scene, 16);
        let reconstructed = unpack_raster_to_scene(&packed);

        assert_eq!(packed.height, 0);
        assert!(packed.occupancy.is_empty());
        assert!(reconstructed.gaussians.is_empty());
        assert_eq!(reconstructed.mins, [0.0; 3]);
        assert_eq!(reconstructed.maxes, [0.0; 3]);
    }

    #[test]
    fn directory_container_roundtrip_preserves_packed_scene() {
        let scene = synthetic_scene(25);
        let packed = pack_scene_to_raster(&scene, 6);
        let unique = format!(
            "gaussian_packing_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_packed_raster_scene_dir(&packed, &dir).unwrap();
        let loaded = read_packed_raster_scene_dir(&dir).unwrap();

        let mut packed_for_compare = packed.clone();
        packed_for_compare.auxiliary_codec_metadata = loaded.auxiliary_codec_metadata.clone();
        assert_eq!(loaded, packed_for_compare);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn plane_stats_are_generated_for_all_planes() {
        let scene = synthetic_scene(12);
        let packed = pack_scene_to_raster(&scene, 4);
        let stats = compute_plane_stats(&packed);

        assert_eq!(stats.len(), 7);
        assert_eq!(stats[0].name, "occupancy");
        assert!(stats.iter().all(|plane| plane.max >= plane.min));
    }

    #[test]
    fn byte_split_planes_and_stats_are_generated() {
        let scene = synthetic_scene(10);
        let packed = pack_scene_to_raster(&scene, 4);
        let split = split_packed_u16_planes(&packed);
        let stats = compute_byte_split_plane_stats(&packed);

        assert_eq!(split.len(), 6);
        assert_eq!(stats.len(), 6);
        assert_eq!(split[0].hi.len(), packed.geom_x.len());
        assert_eq!(split[0].lo.len(), packed.geom_x.len());
        assert_eq!(stats[0].name, "geom_x");
        assert!(
            stats
                .iter()
                .all(|entry| entry.base_u16.max >= entry.base_u16.min)
        );
    }

    #[test]
    fn geom_hi_png_roundtrip_rebuilds_original_geom_planes() {
        let scene = synthetic_scene(18);
        let packed = pack_scene_to_raster(&scene, 5);
        let sorted = sort_gaussians_by_morton(&scene);
        let unique = format!(
            "gaussian_packing_geom_hi_png_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_geom_hi_pngs(&packed, &dir).unwrap();
        let decoded_hi = load_geom_hi_pngs(&dir, packed.width, packed.height).unwrap();
        let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
            &packed,
            [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
        );
        assert_eq!(rebuilt.geom_x, packed.geom_x);
        assert_eq!(rebuilt.geom_y, packed.geom_y);
        assert_eq!(rebuilt.geom_z, packed.geom_z);

        let result = evaluate_geom_hi_png_roundtrip(&sorted, &packed, &dir).unwrap();
        for axis in result.xyz_rmse {
            assert!(
                axis < 0.0002,
                "geometry RMSE too high after hi-byte png roundtrip: {axis}"
            );
        }

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn geom_hi_rgb_png_roundtrip_rebuilds_original_geom_planes() {
        let scene = synthetic_scene(18);
        let packed = pack_scene_to_raster(&scene, 5);
        let sorted = sort_gaussians_by_morton(&scene);
        let unique = format!(
            "gaussian_packing_geom_hi_rgb_png_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_geom_hi_rgb_pngs_helper(&packed, &dir).unwrap();
        let decoded_hi = load_geom_hi_rgb_png(&dir, packed.width, packed.height).unwrap();
        let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
            &packed,
            [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
        );
        assert_eq!(rebuilt.geom_x, packed.geom_x);
        assert_eq!(rebuilt.geom_y, packed.geom_y);
        assert_eq!(rebuilt.geom_z, packed.geom_z);

        let result = evaluate_geom_hi_rgb_png_roundtrip(&sorted, &packed, &dir).unwrap();
        for axis in result.xyz_rmse {
            assert!(
                axis < 0.0002,
                "geometry RMSE too high after rgb hi-byte png roundtrip: {axis}"
            );
        }

        fs::remove_dir_all(dir).unwrap();
    }

    fn write_geom_hi_rgb_pngs_helper(
        packed: &PackedRasterScene,
        dir: &Path,
    ) -> Result<GeomHiRgbPngPath, Box<dyn std::error::Error>> {
        write_geom_hi_rgb_png(packed, dir)
    }
}
