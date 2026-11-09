use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use gaussian_packing::{
    quantize_sh_rest_scalar, read_sh_rest_video_atlas_streams, write_sh_rest_video_atlas_streams,
    QualityPreset, QuantParams, ShRestVideoAtlasCodecConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SH_C0: f32 = 0.282_094_8;
const DEFAULT_GEOMETRY_BITDEPTH: u8 = 10;
const DEFAULT_3DGSCOMPARE_PYTHON: &str =
    "/mnt/c/Users/email/3dgsCompare/backend/.venv/Scripts/python.exe";
const DEFAULT_3DGSCOMPARE_BACKEND: &str = "/mnt/c/Users/email/3dgsCompare/backend";
const DEFAULT_3DGSCOMPARE_ORIGINAL_SCENE_ID: &str = "8982f310-c91f-4dcf-a0fa-b098fe87c762";
const DEFAULT_3DGSCOMPARE_CAMERA_BUNDLE_ID: &str = "d11c5f08-e5aa-4f28-be9d-444672eda4ff";

type DynError = Box<dyn Error>;

#[derive(Debug, Clone)]
struct Cli {
    source: PathBuf,
    output_dir: PathBuf,
    tmc2_root: PathBuf,
    max_points: Option<usize>,
    geometry_bitdepth: u8,
    threads: usize,
    rate: String,
    prepare_only: bool,
    lossless_geometry: bool,
    full_sidecar: bool,
    visual_sidecar: bool,
    anchor_grouped_sidecar: bool,
    collision_only_geometry_residuals: bool,
    packed_geometry_residuals: bool,
    vpcc_only_geometry: bool,
    high_low_geometry: bool,
    signed_log_geometry: bool,
    entropy_low_bits: bool,
    semantic_webp_grid: bool,
    geometry_residual_bits: u8,
    geometry_anchor_residuals: bool,
    sh_vq_codebook_size: Option<usize>,
    sh_vq_iterations: usize,
    sh_vq_training_samples: usize,
    sh_video_atlas: Option<ShVideoPreset>,
    sh_video_atlas_width: u32,
    vpcc_atlas_width: Option<u32>,
    vpcc_disable_patch_splitting: bool,
    export_patch_map: bool,
    benchmark_3dgscompare: bool,
    benchmark_python: Option<PathBuf>,
    benchmark_backend: Option<PathBuf>,
    benchmark_original_scene_id: String,
    benchmark_camera_bundle_id: String,
}

#[derive(Debug)]
struct DecodeCli {
    container_dir: PathBuf,
    output_ply: PathBuf,
    tmc2_root: PathBuf,
    threads: usize,
}

#[derive(Debug)]
struct LocalityCli {
    source: PathBuf,
    manifest_path: PathBuf,
    patch_map_path: PathBuf,
    report_path: PathBuf,
}

#[derive(Debug)]
struct GeometryFidelityCli {
    source: PathBuf,
    output_dir: PathBuf,
    tmc2_root: PathBuf,
    max_points: Option<usize>,
    geometry_bitdepths: Vec<u8>,
    threads: usize,
}

#[derive(Debug)]
struct SpatialTilesCli {
    source: PathBuf,
    output_dir: PathBuf,
    tiles: usize,
}

/// A complete spatially tiled container run.  Unlike `spatial-tiles`, this
/// owns the entire lifecycle: source partitioning, per-tile V-PCC round trips,
/// restoration, merge, and optional scene-level benchmark.
#[derive(Debug)]
struct TiledRunCli {
    base: Cli,
    tiles: usize,
    tile_jobs: usize,
}

#[derive(Debug, Serialize)]
struct TiledContainerManifest {
    schema_version: u32,
    mode: String,
    source_ply: String,
    source_points: usize,
    tiles: usize,
    tile_jobs: usize,
    geometry_bitdepth: u8,
    vpcc_only_geometry: bool,
    sh_vq_codebook_size: Option<usize>,
    tile_containers: Vec<TiledContainerEntry>,
    restored_ply: String,
    restored_points: usize,
    container_payload_bytes: u64,
    benchmark_3dgscompare: Option<String>,
}

#[derive(Debug, Serialize)]
struct TiledContainerEntry {
    index: usize,
    source_ply: String,
    container_dir: String,
    selected_points: usize,
    vpcc_points: usize,
    container_payload_bytes: u64,
}

#[derive(Debug, Serialize)]
struct GeometryFidelityReport {
    schema_version: u32,
    method: String,
    source_ply: String,
    max_points: Option<usize>,
    results: Vec<GeometryFidelityResult>,
}

#[derive(Debug, Serialize)]
struct GeometryFidelityResult {
    geometry_bitdepth: u8,
    status: String,
    error: Option<String>,
    selected_points: Option<usize>,
    quantized_unique_points: Option<usize>,
    quantization_collisions: Option<usize>,
    decoded_records: Option<usize>,
    decoded_unique_points: Option<usize>,
    missing_points: Option<usize>,
    unexpected_points: Option<usize>,
    exact_point_set: Option<bool>,
    vpcc_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct LocalityReport {
    schema_version: u32,
    method: String,
    source_points_selected: usize,
    vpcc_atlas_points: usize,
    populated_atlas_points: usize,
    unresolved_source_gaussians: usize,
    atlas_neighbor_pairs: usize,
    random_pairs: usize,
    atlas_neighbors: AttributeDistances,
    random_baseline: AttributeDistances,
}

#[derive(Debug, Serialize, Default)]
struct AttributeDistances {
    sh_dc_l2_mean: f64,
    sh_rest_rmse_mean: f64,
    opacity_abs_mean: f64,
    scale_rmse_mean: f64,
    rotation_angle_radians_mean: f64,
}

#[derive(Debug)]
struct PatchMapEntry {
    patch: u32,
    u: u32,
    v: u32,
    layer: u32,
}

#[derive(Debug)]
struct AtlasCell {
    location: PatchMapEntry,
    count: usize,
    dc: [f64; 3],
    sh_rest: Vec<f64>,
    opacity: f64,
    scale: [f64; 3],
    rotation: [f64; 4],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PlyFormat {
    BinaryLittleEndian,
    Ascii,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Property {
    name: String,
    scalar_type: ScalarType,
    offset: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(value: &str) -> Result<Self, DynError> {
        match value {
            "char" | "int8" => Ok(Self::I8),
            "uchar" | "uint8" => Ok(Self::U8),
            "short" | "int16" => Ok(Self::I16),
            "ushort" | "uint16" => Ok(Self::U16),
            "int" | "int32" => Ok(Self::I32),
            "uint" | "uint32" => Ok(Self::U32),
            "float" | "float32" => Ok(Self::F32),
            "double" | "float64" => Ok(Self::F64),
            _ => Err(format!("unsupported PLY scalar type: {value}").into()),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn ply_name(self) -> &'static str {
        match self {
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    fn read(self, bytes: &[u8]) -> f64 {
        match self {
            Self::I8 => i8::from_le_bytes([bytes[0]]) as f64,
            Self::U8 => bytes[0] as f64,
            Self::I16 => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
            Self::U16 => u16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
            Self::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            Self::U32 => u32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            Self::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            Self::F64 => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
        }
    }
}

#[derive(Debug, Clone)]
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    properties: Vec<Property>,
    record_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct RequiredColumns {
    xyz: [usize; 3],
    sh_dc: [usize; 3],
}

#[derive(Debug, Serialize, Deserialize)]
struct TransformManifest {
    method: String,
    geometry_bitdepth: u8,
    quantized_max: u32,
    source_min: [f64; 3],
    source_max: [f64; 3],
    source_longest_extent: f64,
    scale_to_quantized: f64,
    inverse_scale_to_source: f64,
    inverse_formula: String,
    /// When true, quantization operates in SOG's signed-log1p coordinate
    /// domain, with one shared range for x/y/z just as SOG-web does.
    #[serde(default)]
    signed_log_coordinates: bool,
    #[serde(default)]
    coordinate_domain_min: Option<f64>,
    #[serde(default)]
    coordinate_domain_max: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConversionManifest {
    schema_version: u32,
    created_unix_seconds: u64,
    mode: String,
    source_ply: String,
    #[serde(default)]
    source_file_bytes: u64,
    source_format: PlyFormat,
    source_points: usize,
    selected_points: usize,
    vpcc_points: usize,
    selection: String,
    output_ply: String,
    output_properties: Vec<String>,
    color_mapping: String,
    duplicate_quantized_positions: usize,
    transform: TransformManifest,
    omitted_3dgs_fields: Vec<String>,
    tmc2_root: String,
    tmc2_commit: Option<String>,
    tmc2_profile: Tmc2Profile,
    artifacts: ArtifactManifest,
}

#[derive(Debug, Serialize, Deserialize)]
struct Tmc2Profile {
    common_config: String,
    condition_config: String,
    rate_config: String,
    video_codec: String,
    video_codec_id: u8,
    threads: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ArtifactManifest {
    compressed_stream: Option<String>,
    compressed_bytes: Option<u64>,
    encoder_reconstruction: Option<String>,
    decoded_point_cloud: Option<String>,
    encoder_log: Option<String>,
    decoder_log: Option<String>,
    #[serde(default)]
    vpcc_patch_map: Option<String>,
    #[serde(default)]
    benchmark_3dgscompare: Option<String>,
    sidecar: Option<SidecarManifest>,
    #[serde(default)]
    visual_sidecar: Option<VisualSidecarManifest>,
    restored_ply: Option<String>,
    container_payload_bytes: Option<u64>,
    manifest_bytes: Option<u64>,
    container_total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SidecarManifest {
    codec: String,
    path: String,
    compressed_bytes: u64,
    compressed_sha256: String,
    uncompressed_bytes: u64,
    record_size: usize,
    records: usize,
    properties: Vec<Property>,
    #[serde(default)]
    sh_vq: Option<ShVqManifest>,
    #[serde(default)]
    sh_video: Option<ShVideoManifest>,
    #[serde(default)]
    geometry_anchor_residuals: Option<GeometryAnchorResidualManifest>,
}

/// A renderer-oriented sidecar. Unlike `SidecarManifest`, this never stores a
/// copied PLY record: positions reference decoded V-PCC anchors and remaining
/// scalar values are stored as semantic quantized streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisualSidecarManifest {
    codec: String,
    path: String,
    compressed_bytes: u64,
    compressed_sha256: String,
    records: usize,
    record_size: usize,
    properties: Vec<Property>,
    anchor_index_bytes: usize,
    anchor_set_sha256: String,
    semantic_columns: Vec<usize>,
    quantization: Vec<ShVideoQuantization>,
    sh_vq: ShVqManifest,
    #[serde(default)]
    anchor_groups: Option<AnchorGroupsManifest>,
    #[serde(default)]
    semantic_webp: Option<SemanticWebpManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticWebpManifest {
    path: String,
    total_bytes: u64,
    manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnchorGroupsManifest {
    order: String,
    counts_path: String,
    counts_bytes: u64,
    counts_sha256: String,
    residual_path: String,
    residual_bytes: u64,
    residual_sha256: String,
    residual_bits: u8,
    #[serde(default = "anchor_group_residuals_for_all_records")]
    residuals_for_singletons: bool,
    #[serde(default)]
    residual_records: usize,
    #[serde(default)]
    residuals_bitpacked: bool,
    #[serde(default = "anchor_group_residuals_for_all_records")]
    residuals_for_collisions: bool,
    #[serde(default)]
    high_low_geometry: bool,
    #[serde(default)]
    entropy_low_bits: bool,
}

fn anchor_group_residuals_for_all_records() -> bool {
    true
}

#[derive(Debug, Clone, Copy)]
enum ShVideoPreset {
    Av1,
    Lossless,
}

impl ShVideoPreset {
    fn parse(value: &str) -> Result<Self, DynError> {
        match value {
            "av1" => Ok(Self::Av1),
            "lossless" => Ok(Self::Lossless),
            _ => Err("SH video preset must be av1 or lossless".into()),
        }
    }
    fn codec_config(self) -> ShRestVideoAtlasCodecConfig {
        match self {
            // AV1 is intentionally 4:4:4, avoiding chroma subsampling of the byte planes.
            Self::Av1 => ShRestVideoAtlasCodecConfig {
                quality_preset: QualityPreset::VeryGood,
                // The bundled SVT-AV1 build only accepts 4:2:0; libaom preserves 4:4:4.
                codec: "libaom-av1",
                pixel_format: "yuv444p",
                ffmpeg_preset: None,
                crf: Some(18),
                lossless: false,
            },
            Self::Lossless => {
                ShRestVideoAtlasCodecConfig::from_quality_preset(QualityPreset::Lossless)
            }
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Av1 => "av1_libaom_yuv444p_crf18",
            Self::Lossless => "ffv1_rgb24_lossless",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShVideoQuantization {
    min: f32,
    max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShVideoManifest {
    preset: String,
    codec: String,
    width: u32,
    height: u32,
    active_len: usize,
    vector_dimensions: usize,
    quantization: Vec<ShVideoQuantization>,
    hi_path: String,
    hi_bytes: u64,
    hi_sha256: String,
    lo_path: String,
    lo_bytes: u64,
    lo_sha256: String,
    rmse: f64,
    max_abs_error: f64,
}

#[derive(Debug, Clone)]
struct ShVqConfig {
    codebook_size: usize,
    iterations: usize,
    training_samples: usize,
    columns: Vec<usize>,
    codebook: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShVqManifest {
    codebook_entries: usize,
    vector_dimensions: usize,
    index_bytes: usize,
    indices_path: String,
    indices_compressed_bytes: u64,
    indices_compressed_sha256: String,
    indices_uncompressed_bytes: u64,
    codebook_path: String,
    codebook_bytes: u64,
    codebook_sha256: String,
    training_samples: usize,
    kmeans_iterations: usize,
    rmse: f64,
    max_abs_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeometryAnchorResidualManifest {
    anchor_order: String,
    index_bytes: usize,
    indices_path: String,
    indices_compressed_bytes: u64,
    indices_compressed_sha256: String,
    indices_uncompressed_bytes: u64,
    xor_path: String,
    xor_compressed_bytes: u64,
    xor_compressed_sha256: String,
    xor_uncompressed_bytes: u64,
    residual_method: String,
}

impl SidecarManifest {
    fn total_bytes(&self) -> u64 {
        self.compressed_bytes
            + self
                .sh_vq
                .as_ref()
                .map(|vq| vq.indices_compressed_bytes + vq.codebook_bytes)
                .unwrap_or(0)
            + self
                .sh_video
                .as_ref()
                .map(|video| video.hi_bytes + video.lo_bytes)
                .unwrap_or(0)
            + self
                .geometry_anchor_residuals
                .as_ref()
                .map(|geometry| geometry.indices_compressed_bytes + geometry.xor_compressed_bytes)
                .unwrap_or(0)
    }
}

impl VisualSidecarManifest {
    fn total_bytes(&self) -> u64 {
        self.compressed_bytes
            + self.sh_vq.indices_compressed_bytes
            + self.sh_vq.codebook_bytes
            + self
                .anchor_groups
                .as_ref()
                .map(|groups| groups.counts_bytes + groups.residual_bytes)
                .unwrap_or(0)
    }
}

fn main() -> Result<(), DynError> {
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("decode-full")) {
        return decode_full_command(&parse_decode_cli()?);
    }
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("locality-report")) {
        return locality_report_command(&parse_locality_cli()?);
    }
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("geometry-fidelity")) {
        return geometry_fidelity_command(&parse_geometry_fidelity_cli()?);
    }
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("spatial-tiles")) {
        return spatial_tiles_command(&parse_spatial_tiles_cli()?);
    }
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("tiled-run")) {
        return tiled_run_command(&parse_tiled_run_cli()?);
    }
    let cli = parse_cli()?;
    let started = Instant::now();
    let notifier = discord_bot::DiscordNotifier::from_env(discord_bot::RunContext::new(
        "tmc2_vpcc",
        cli.source.display().to_string(),
        cli.output_dir.display().to_string(),
    ));
    notifier.run_started();

    let result = run(&cli, &notifier);
    match &result {
        Ok(manifest) => {
            let mut completion = discord_bot::CompletionDetails::new(started.elapsed())
                .add("selected points", manifest.selected_points.to_string());
            if let Some(bytes) = manifest.artifacts.compressed_bytes {
                completion = completion.add("V-PCC bytes", bytes.to_string());
            }
            if let Some(sidecar) = &manifest.artifacts.sidecar {
                completion = completion.add("sidecar bytes", sidecar.total_bytes().to_string());
                if let Some(vq) = &sidecar.sh_vq {
                    completion = completion.add("SH VQ RMSE", format!("{:.6}", vq.rmse));
                }
            }
            if let Some(sidecar) = &manifest.artifacts.visual_sidecar {
                completion =
                    completion.add("visual sidecar bytes", sidecar.total_bytes().to_string());
                completion = completion.add("SH VQ RMSE", format!("{:.6}", sidecar.sh_vq.rmse));
            }
            if let Some(bytes) = manifest.artifacts.container_total_bytes {
                completion = completion.with_total_bytes(bytes);
                if manifest.selected_points == manifest.source_points
                    && manifest.source_file_bytes > 0
                    && bytes > 0
                {
                    let ratio = manifest.source_file_bytes as f64 / bytes as f64;
                    let reduction =
                        100.0 * (1.0 - bytes as f64 / manifest.source_file_bytes as f64);
                    completion = completion.add(
                        "compression ratio",
                        format!("{ratio:.3}:1 ({reduction:.2}% smaller)"),
                    );
                }
            }
            notifier.run_completed(completion);
        }
        Err(error) => notifier.run_failed("tmc2 adapter", &error.to_string(), started.elapsed()),
    }
    result.map(|_| ())
}

fn parse_spatial_tiles_cli() -> Result<SpatialTilesCli, DynError> {
    let mut args = env::args_os().skip(2);
    let source = PathBuf::from(args.next().ok_or("spatial-tiles requires <source.ply> <output-dir>")?);
    let output_dir = PathBuf::from(args.next().ok_or("spatial-tiles requires <source.ply> <output-dir>")?);
    let mut tiles = 4usize;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--tiles" => tiles = parse_value(next_value(&mut args, "--tiles")?, "tile count")?,
            unknown => return Err(format!("unknown spatial-tiles argument: {unknown}").into()),
        }
    }
    if tiles < 2 {
        return Err("spatial-tiles requires at least two tiles".into());
    }
    Ok(SpatialTilesCli { source, output_dir, tiles })
}

fn spatial_tiles_command(cli: &SpatialTilesCli) -> Result<(), DynError> {
    let started = Instant::now();
    let notifier = discord_bot::DiscordNotifier::from_env(discord_bot::RunContext::new(
        "tmc2_spatial_tiles",
        cli.source.display().to_string(),
        cli.output_dir.display().to_string(),
    ));
    notifier.run_started();
    notifier.stage_changed(&format!("partition source into {} balanced spatial tiles", cli.tiles));
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spatial_tiles.py");
    let result = Command::new("python3")
        .arg(script)
        .arg("tile")
        .arg(&cli.source)
        .arg(&cli.output_dir)
        .arg("--tiles")
        .arg(cli.tiles.to_string())
        .status()
        .map_err(|error| format!("could not launch spatial tiling utility: {error}"))
        .and_then(|status| {
            if status.success() { Ok(()) } else { Err(format!("spatial tiling utility exited with {status}")) }
        });
    match &result {
        Ok(()) => notifier.run_completed(
            discord_bot::CompletionDetails::new(started.elapsed())
                .add("tiles", cli.tiles.to_string()),
        ),
        Err(error) => notifier.run_failed("spatial tile preparation", error, started.elapsed()),
    }
    result.map_err(Into::into)
}

fn parse_tiled_run_cli() -> Result<TiledRunCli, DynError> {
    let mut args = env::args_os().skip(2);
    let source = args.next().ok_or(usage())?;
    let output_dir = args.next().ok_or(usage())?;
    let mut tiles = 4usize;
    let mut tile_jobs = 2usize;
    let mut base_args = vec![source, output_dir];
    while let Some(arg) = args.next() {
        if arg == "--tiles" {
            tiles = parse_value(next_value(&mut args, "--tiles")?, "tile count")?;
        } else if arg == "--tile-jobs" {
            tile_jobs = parse_value(next_value(&mut args, "--tile-jobs")?, "tile jobs")?;
        } else {
            base_args.push(arg);
        }
    }
    if tiles < 2 {
        return Err("tiled-run requires at least two tiles".into());
    }
    if tile_jobs == 0 {
        return Err("tiled-run requires --tile-jobs to be at least 1".into());
    }
    Ok(TiledRunCli {
        base: parse_cli_args(base_args.into_iter())?,
        tiles,
        tile_jobs: tile_jobs.min(tiles),
    })
}

fn tiled_run_command(cli: &TiledRunCli) -> Result<(), DynError> {
    if !(cli.base.full_sidecar || cli.base.visual_sidecar || cli.base.anchor_grouped_sidecar) {
        return Err("tiled-run requires a sidecar mode so it can restore and merge full 3DGS PLYs".into());
    }
    validate_cli(&cli.base)?;
    fs::create_dir_all(&cli.base.output_dir)?;
    ensure_empty_output(&cli.base.output_dir)?;

    let started = Instant::now();
    let notifier = discord_bot::DiscordNotifier::from_env(discord_bot::RunContext::new(
        "tmc2_tiled_vpcc",
        cli.base.source.display().to_string(),
        cli.base.output_dir.display().to_string(),
    ));
    notifier.run_started();
    let result = (|| -> Result<TiledContainerManifest, DynError> {
        let tile_sources = cli.base.output_dir.join("sources");
        notifier.stage_changed(&format!(
            "partition source into {} balanced spatial tiles",
            cli.tiles
        ));
        run_spatial_tile_script("tile", &cli.base.source, &tile_sources, cli.tiles, &[])?;

        let tiles_root = cli.base.output_dir.join("tiles");
        let mut entries = Vec::with_capacity(cli.tiles);
        let mut payload_bytes = 0u64;
        let per_tile_threads = (cli.base.threads / cli.tile_jobs).max(1);
        for batch_start in (0..cli.tiles).step_by(cli.tile_jobs) {
            let batch_end = (batch_start + cli.tile_jobs).min(cli.tiles);
            notifier.stage_changed(&format!(
                "encode, decode, and restore tiles {}-{} of {} in parallel ({} jobs, {} TMC2 threads/job)",
                batch_start + 1,
                batch_end,
                cli.tiles,
                cli.tile_jobs,
                per_tile_threads
            ));
            let mut batch = Vec::new();
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for index in batch_start..batch_end {
                    let source = tile_sources.join(format!("tile_{index:02}.ply"));
                    let output_dir = tiles_root.join(format!("tile_{index:02}"));
                    let mut tile_cli = cli.base.clone();
                    tile_cli.source = source.clone();
                    tile_cli.output_dir = output_dir.clone();
                    tile_cli.threads = per_tile_threads;
                    // A per-tile render benchmark is both redundant and incomparable;
                    // benchmark only the merged full-scene reconstruction below.
                    tile_cli.benchmark_3dgscompare = false;
                    handles.push(scope.spawn(move || {
                        let tile_notifier = discord_bot::DiscordNotifier::disabled(
                            discord_bot::RunContext::new(
                                "tmc2_tiled_child",
                                source.display().to_string(),
                                output_dir.display().to_string(),
                            ),
                        );
                        run(&tile_cli, &tile_notifier).map_err(|error| error.to_string())
                    }));
                }
                for handle in handles {
                    batch.push(handle.join().map_err(|_| "tiled worker thread panicked")?);
                }
                Ok::<(), DynError>(())
            })?;
            for (offset, result) in batch.into_iter().enumerate() {
                let index = batch_start + offset;
                let manifest = result.map_err(|error| format!("tile {} failed: {error}", index + 1))?;
                let bytes = manifest.artifacts.container_payload_bytes.unwrap_or(0);
                payload_bytes += bytes;
                entries.push(TiledContainerEntry {
                    index,
                    source_ply: format!("sources/tile_{index:02}.ply"),
                    container_dir: format!("tiles/tile_{index:02}"),
                    selected_points: manifest.selected_points,
                    vpcc_points: manifest.vpcc_points,
                    container_payload_bytes: bytes,
                });
            }
        }

        notifier.stage_changed("merge restored spatial tiles");
        let restored = cli.base.output_dir.join("restored_0000.ply");
        let restored_inputs: Vec<PathBuf> = (0..cli.tiles)
            .map(|index| tiles_root.join(format!("tile_{index:02}/restored_0000.ply")))
            .collect();
        run_spatial_tile_script("merge", &restored, Path::new(""), 0, &restored_inputs)?;
        let (_, restored_header) = read_header(&restored)?;
        let (_, source_header) = read_header(&cli.base.source)?;
        if restored_header.vertex_count != source_header.vertex_count {
            return Err(format!(
                "tiled reconstruction count mismatch: restored={} source={}",
                restored_header.vertex_count, source_header.vertex_count
            )
            .into());
        }

        let benchmark = if cli.base.benchmark_3dgscompare {
            notifier.stage_changed("run canonical 3dgsCompare benchmark on merged reconstruction");
            let output = cli.base.output_dir.join("benchmark_3dgscompare.json");
            let log = cli.base.output_dir.join("benchmark_3dgscompare.log");
            run_3dgscompare_benchmark(&cli.base, &restored, &output, &log)?;
            Some("benchmark_3dgscompare.json".into())
        } else {
            None
        };

        let manifest = TiledContainerManifest {
            schema_version: 1,
            mode: "mpeg_vpcc_3dgs_spatial_tiled".into(),
            source_ply: cli.base.source.display().to_string(),
            source_points: source_header.vertex_count,
            tiles: cli.tiles,
            tile_jobs: cli.tile_jobs,
            geometry_bitdepth: cli.base.geometry_bitdepth,
            vpcc_only_geometry: cli.base.vpcc_only_geometry,
            sh_vq_codebook_size: cli.base.sh_vq_codebook_size,
            tile_containers: entries,
            restored_ply: "restored_0000.ply".into(),
            restored_points: restored_header.vertex_count,
            container_payload_bytes: payload_bytes,
            benchmark_3dgscompare: benchmark,
        };
        write_tiled_manifest(&cli.base.output_dir, &manifest)?;
        Ok(manifest)
    })();
    match &result {
        Ok(manifest) => notifier.run_completed(
            discord_bot::CompletionDetails::new(started.elapsed())
                .add("tiles", manifest.tiles.to_string())
                .add("restored points", manifest.restored_points.to_string())
                .with_total_bytes(manifest.container_payload_bytes),
        ),
        Err(error) => notifier.run_failed("tiled V-PCC run", &error.to_string(), started.elapsed()),
    }
    result.map(|_| ())
}

fn run_spatial_tile_script(
    command: &str,
    primary: &Path,
    output: &Path,
    tiles: usize,
    inputs: &[PathBuf],
) -> Result<(), DynError> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spatial_tiles.py");
    let mut process = Command::new("python3");
    process.arg(script).arg(command).arg(primary);
    if command == "tile" {
        process.arg(output).arg("--tiles").arg(tiles.to_string());
    } else if command == "merge" {
        process.args(inputs);
    } else {
        return Err(format!("unsupported spatial tiling command: {command}").into());
    }
    let status = process.status().map_err(|error| format!("could not launch spatial tiling utility: {error}"))?;
    if !status.success() {
        return Err(format!("spatial tiling utility exited with {status}").into());
    }
    Ok(())
}

fn run(cli: &Cli, notifier: &discord_bot::DiscordNotifier) -> Result<ConversionManifest, DynError> {
    validate_cli(cli)?;
    fs::create_dir_all(&cli.output_dir)?;
    ensure_empty_output(&cli.output_dir)?;

    let input_dir = cli.output_dir.join("input");
    fs::create_dir_all(&input_dir)?;
    let converted_path = input_dir.join("frame_0000.ply");
    let sidecar_path = (cli.full_sidecar || cli.visual_sidecar || cli.anchor_grouped_sidecar)
        .then(|| cli.output_dir.join("gaussians.deflate"));

    notifier.stage_changed("scan source bounds");
    let started = Instant::now();
    let (header, mins, maxes) = scan_bounds(&cli.source)?;
    notifier.stage_changed(&format!(
        "source bounds scanned: points={} elapsed={:.1}s",
        header.vertex_count,
        started.elapsed().as_secs_f64()
    ));

    notifier.stage_changed("convert positions and DC color");
    let conversion_started = Instant::now();
    let sh_vq = if let Some(codebook_size) = cli.sh_vq_codebook_size {
        notifier.stage_changed("train SH vector-quantization codebook");
        Some(train_sh_vq_codebook(
            &cli.source,
            &header,
            codebook_size,
            cli.sh_vq_iterations,
            cli.sh_vq_training_samples,
        )?)
    } else {
        None
    };
    let position_bitdepth = if cli.high_low_geometry { 16 } else { cli.geometry_bitdepth };
    let mut conversion = convert_ply(
        &cli.source,
        &converted_path,
        &header,
        mins,
        maxes,
        cli.max_points,
        position_bitdepth,
        cli.high_low_geometry.then_some(cli.geometry_bitdepth),
        cli.signed_log_geometry,
        cli.lossless_geometry,
        if cli.sh_video_atlas.is_some() || cli.visual_sidecar || cli.anchor_grouped_sidecar {
            None
        } else {
            sidecar_path.as_deref()
        },
        sh_vq.as_ref(),
        cli.geometry_anchor_residuals,
    )?;
    if let Some(preset) = cli.sh_video_atlas {
        notifier.stage_changed("rasterize SH sidecar and encode SH video");
        conversion.sidecar = Some(write_sh_video_sidecar(
            &cli.source,
            &header,
            conversion.selected_points,
            sidecar_path
                .as_deref()
                .ok_or("SH video requires a full sidecar")?,
            preset,
            cli.sh_video_atlas_width,
        )?);
    }
    let mut visual_sidecar = if cli.anchor_grouped_sidecar {
        notifier.stage_changed("write anchor-grouped semantic sidecar");
        Some(write_anchor_grouped_sidecar(
            &cli.source,
            &header,
            conversion.selected_points,
            &conversion.transform,
            sh_vq
                .as_ref()
                .ok_or("anchor-grouped sidecar requires an SH VQ codebook")?,
            cli.geometry_residual_bits,
            cli.collision_only_geometry_residuals,
            cli.packed_geometry_residuals,
            cli.vpcc_only_geometry,
            cli.high_low_geometry,
            cli.entropy_low_bits,
            cli.geometry_bitdepth,
            sidecar_path
                .as_deref()
                .ok_or("anchor-grouped sidecar requires an output path")?,
        )?)
    } else if cli.visual_sidecar {
        notifier.stage_changed("write compact visual semantic sidecar");
        Some(write_visual_sidecar(
            &cli.source,
            &header,
            conversion.selected_points,
            &conversion.transform,
            sh_vq
                .as_ref()
                .ok_or("visual sidecar requires an SH VQ codebook")?,
            sidecar_path
                .as_deref()
                .ok_or("visual sidecar requires an output path")?,
        )?)
    } else {
        None
    };
    if cli.semantic_webp_grid {
        let visual = visual_sidecar
            .as_mut()
            .ok_or("--semantic-webp-grid requires an anchor-grouped sidecar")?;
        notifier.stage_changed("encode lossless WebP semantic grid");
        let grid = encode_semantic_webp_grid(&cli.output_dir, sidecar_path.as_deref().ok_or("semantic grid requires sidecar path")?, visual)?;
        fs::remove_file(sidecar_path.as_deref().ok_or("semantic grid requires sidecar path")?)?;
        visual.path = grid.path.clone();
        visual.compressed_bytes = grid.total_bytes;
        visual.compressed_sha256 = grid.manifest_sha256.clone();
        visual.semantic_webp = Some(grid);
    }
    notifier.stage_changed(&format!(
        "point cloud prepared: selected={} vpcc_points={} collisions={} elapsed={:.1}s",
        conversion.selected_points,
        conversion.vpcc_points,
        conversion.duplicate_positions,
        conversion_started.elapsed().as_secs_f64()
    ));

    let tmc2_commit = git_commit(&cli.tmc2_root);
    let mut manifest = ConversionManifest {
        schema_version: 1,
        created_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        mode: if cli.signed_log_geometry {
            "mpeg_vpcc_3dgs_hybrid_v18_signed_log_high_low_geometry"
        } else if cli.semantic_webp_grid {
            "mpeg_vpcc_3dgs_hybrid_v12_semantic_webp_grid"
        } else if cli.anchor_grouped_sidecar
            && cli.high_low_geometry
            && cli.geometry_bitdepth == 12
            && !cli.entropy_low_bits
        {
            "mpeg_vpcc_3dgs_hybrid_v11_high_low_12bit_geometry"
        } else if cli.anchor_grouped_sidecar && cli.entropy_low_bits {
            "mpeg_vpcc_3dgs_hybrid_v10_high_low_entropy_geometry"
        } else if cli.anchor_grouped_sidecar && cli.high_low_geometry {
            "mpeg_vpcc_3dgs_hybrid_v9_high_low_16bit_geometry"
        } else if cli.anchor_grouped_sidecar && cli.vpcc_only_geometry {
            "mpeg_vpcc_3dgs_hybrid_v6_vpcc_only_geometry"
        } else if cli.anchor_grouped_sidecar && cli.packed_geometry_residuals {
            "mpeg_vpcc_3dgs_hybrid_v5_packed_geometry_residuals"
        } else if cli.anchor_grouped_sidecar && cli.collision_only_geometry_residuals {
            "mpeg_vpcc_3dgs_hybrid_v4_collision_residuals"
        } else if cli.anchor_grouped_sidecar {
            "mpeg_vpcc_3dgs_hybrid_v3_anchor_grouped"
        } else if cli.visual_sidecar {
            "mpeg_vpcc_3dgs_hybrid_v2_visual"
        } else if cli.full_sidecar {
            "mpeg_vpcc_3dgs_hybrid_v1"
        } else if cli.lossless_geometry {
            "tmc2_vpcc_lossless_geometry_baseline"
        } else {
            "tmc2_vpcc_point_cloud_baseline"
        }
        .into(),
        source_ply: cli.source.display().to_string(),
        source_file_bytes: fs::metadata(&cli.source)?.len(),
        source_format: header.format,
        source_points: header.vertex_count,
        selected_points: conversion.selected_points,
        vpcc_points: conversion.vpcc_points,
        selection: "evenly_spaced_source_indices".into(),
        output_ply: converted_path.display().to_string(),
        output_properties: ["x", "y", "z", "red", "green", "blue"]
            .into_iter()
            .map(String::from)
            .collect(),
        color_mapping: "rgb8=round(clamp(0.5+0.28209479177387814*f_dc,0,1)*255)".into(),
        duplicate_quantized_positions: conversion.duplicate_positions,
        transform: conversion.transform,
        omitted_3dgs_fields: if cli.full_sidecar || cli.visual_sidecar || cli.anchor_grouped_sidecar
        {
            Vec::new()
        } else {
            ["sh_rest", "opacity", "scale", "rotation"]
                .into_iter()
                .map(String::from)
                .collect()
        },
        tmc2_root: cli.tmc2_root.display().to_string(),
        tmc2_commit,
        tmc2_profile: Tmc2Profile {
            common_config: if cli.lossless_geometry {
                "cfg/common/ctc-common-lossless-geometry.cfg".into()
            } else {
                "cfg/common/ctc-common.cfg".into()
            },
            condition_config: if cli.lossless_geometry {
                "cfg/condition/ctc-all-intra-lossless-geometry.cfg".into()
            } else {
                "cfg/condition/ctc-all-intra.cfg".into()
            },
            rate_config: format!(
                "cfg/rate/ctc-{}.cfg",
                if cli.lossless_geometry {
                    "r5-lossless"
                } else {
                    &cli.rate
                }
            ),
            video_codec: if cli.lossless_geometry {
                "HM-16.20+SCM-8.8 standalone reference application"
            } else {
                "HM-16.20+SCM-8.8 in-process reference library"
            }
            .into(),
            video_codec_id: if cli.lossless_geometry { 1 } else { 4 },
            threads: cli.threads,
        },
        artifacts: ArtifactManifest {
            sidecar: conversion.sidecar,
            visual_sidecar,
            ..ArtifactManifest::default()
        },
    };
    write_manifest(&cli.output_dir, &manifest)?;

    if cli.prepare_only {
        return Ok(manifest);
    }

    notifier.stage_changed("TMC2 encode");
    let stream = cli.output_dir.join("stream.bin");
    let encoder_reconstruction = cli.output_dir.join("encoder_reconstructed_%04d.ply");
    let encoder_log = cli.output_dir.join("encoder.log");
    run_encoder(
        cli,
        &converted_path,
        &stream,
        &encoder_reconstruction,
        &encoder_log,
    )?;
    let compressed_bytes = fs::metadata(&stream)?.len();
    notifier.stage_changed(&format!(
        "TMC2 encode complete: bitstream_bytes={compressed_bytes}"
    ));

    notifier.stage_changed("TMC2 decode");
    let decoded = cli.output_dir.join("decoded_%04d.ply");
    let decoder_log = cli.output_dir.join("decoder.log");
    let patch_map = cli
        .export_patch_map
        .then(|| cli.output_dir.join("vpcc_patch_map.csv"));
    run_decoder(cli, &stream, &decoded, &decoder_log, patch_map.as_deref())?;
    notifier.stage_changed("TMC2 decode complete: decoded frame 0");

    let restored = if cli.full_sidecar || cli.visual_sidecar || cli.anchor_grouped_sidecar {
        notifier.stage_changed("validate V-PCC anchors and restore full 3DGS PLY");
        let restored = cli.output_dir.join("restored_0000.ply");
        restore_full_ply(
            &manifest,
            &cli.output_dir.join("decoded_0000.ply"),
            &restored,
        )?;
        notifier.stage_changed("full 3DGS PLY restored and anchor set validated");
        Some(restored)
    } else {
        None
    };
    let benchmark = if cli.benchmark_3dgscompare {
        let restored = restored
            .as_deref()
            .ok_or("--benchmark-3dgscompare requires a reconstructed full 3DGS PLY")?;
        notifier.stage_changed("run canonical 3dgsCompare benchmark");
        let output = cli.output_dir.join("benchmark_3dgscompare.json");
        let log = cli.output_dir.join("benchmark_3dgscompare.log");
        run_3dgscompare_benchmark(cli, restored, &output, &log)?;
        notifier.stage_changed("canonical 3dgsCompare benchmark complete");
        Some("benchmark_3dgscompare.json".into())
    } else {
        None
    };

    let sidecar_bytes = manifest
        .artifacts
        .sidecar
        .as_ref()
        .map(SidecarManifest::total_bytes)
        .unwrap_or(0)
        + manifest
            .artifacts
            .visual_sidecar
            .as_ref()
            .map(VisualSidecarManifest::total_bytes)
            .unwrap_or(0);

    manifest.artifacts = ArtifactManifest {
        compressed_stream: Some("stream.bin".into()),
        compressed_bytes: Some(compressed_bytes),
        encoder_reconstruction: Some("encoder_reconstructed_0000.ply".into()),
        decoded_point_cloud: Some("decoded_0000.ply".into()),
        encoder_log: Some("encoder.log".into()),
        decoder_log: Some("decoder.log".into()),
        vpcc_patch_map: patch_map.map(|_| "vpcc_patch_map.csv".into()),
        benchmark_3dgscompare: benchmark,
        sidecar: manifest.artifacts.sidecar.take(),
        visual_sidecar: manifest.artifacts.visual_sidecar.take(),
        restored_ply: restored.map(|_| "restored_0000.ply".into()),
        container_payload_bytes: Some(compressed_bytes + sidecar_bytes),
        manifest_bytes: None,
        container_total_bytes: None,
    };
    finalize_manifest_size(&cli.output_dir, &mut manifest)?;
    Ok(manifest)
}

fn parse_cli() -> Result<Cli, DynError> {
    parse_cli_args(env::args_os().skip(1))
}

fn parse_cli_args(mut args: impl Iterator<Item = OsString>) -> Result<Cli, DynError> {
    let source = PathBuf::from(args.next().ok_or(usage())?);
    let output_dir = PathBuf::from(args.next().ok_or(usage())?);
    let mut tmc2_root = env::var_os("TMC2_ROOT")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join("tools/mpeg-pcc-tmc2")))
        .ok_or("set TMC2_ROOT or pass --tmc2-root")?;
    let mut max_points = None;
    let mut geometry_bitdepth = DEFAULT_GEOMETRY_BITDEPTH;
    let mut threads = 8usize;
    let mut rate = "r3".to_string();
    let mut prepare_only = false;
    let mut lossless_geometry = false;
    let mut full_sidecar = false;
    let mut visual_sidecar = false;
    let mut anchor_grouped_sidecar = false;
    let mut collision_only_geometry_residuals = false;
    let mut packed_geometry_residuals = false;
    let mut vpcc_only_geometry = false;
    let mut high_low_geometry = false;
    let mut signed_log_geometry = false;
    let mut entropy_low_bits = false;
    let mut semantic_webp_grid = false;
    let mut geometry_residual_bits = 14u8;
    let mut geometry_anchor_residuals = false;
    let mut sh_vq_codebook_size = None;
    let mut sh_vq_iterations = 8usize;
    let mut sh_vq_training_samples = 20_000usize;
    let mut sh_video_atlas = None;
    let mut sh_video_atlas_width = 2048u32;
    let mut vpcc_atlas_width = None;
    let mut vpcc_disable_patch_splitting = false;
    let mut export_patch_map = false;
    let mut benchmark_3dgscompare = false;
    let mut benchmark_python = None;
    let mut benchmark_backend = None;
    let mut benchmark_original_scene_id = DEFAULT_3DGSCOMPARE_ORIGINAL_SCENE_ID.to_string();
    let mut benchmark_camera_bundle_id = DEFAULT_3DGSCOMPARE_CAMERA_BUNDLE_ID.to_string();

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--tmc2-root" => tmc2_root = PathBuf::from(next_value(&mut args, "--tmc2-root")?),
            "--max-points" => {
                max_points = Some(parse_value(
                    next_value(&mut args, "--max-points")?,
                    "max points",
                )?)
            }
            "--geometry-bitdepth" => {
                geometry_bitdepth = parse_value(
                    next_value(&mut args, "--geometry-bitdepth")?,
                    "geometry bitdepth",
                )?
            }
            "--threads" => threads = parse_value(next_value(&mut args, "--threads")?, "threads")?,
            "--rate" => {
                rate = next_value(&mut args, "--rate")?
                    .to_string_lossy()
                    .into_owned()
            }
            "--prepare-only" => prepare_only = true,
            "--lossless-geometry" => lossless_geometry = true,
            "--full-sidecar" => {
                full_sidecar = true;
                lossless_geometry = true;
            }
            "--visual-sidecar" => {
                visual_sidecar = true;
                lossless_geometry = true;
            }
            "--anchor-grouped-sidecar" => {
                anchor_grouped_sidecar = true;
                lossless_geometry = true;
            }
            "--collision-only-geometry-residuals" => {
                collision_only_geometry_residuals = true;
            }
            "--packed-geometry-residuals" => {
                packed_geometry_residuals = true;
            }
            "--vpcc-only-geometry" => {
                vpcc_only_geometry = true;
            }
            "--high-low-geometry" => {
                high_low_geometry = true;
            }
            "--signed-log-geometry" => {
                signed_log_geometry = true;
            }
            "--entropy-low-bits" => {
                entropy_low_bits = true;
                high_low_geometry = true;
            }
            "--semantic-webp-grid" => semantic_webp_grid = true,
            "--geometry-residual-bits" => {
                geometry_residual_bits = parse_value(
                    next_value(&mut args, "--geometry-residual-bits")?,
                    "geometry residual bits",
                )?
            }
            "--geometry-anchor-residuals" => {
                geometry_anchor_residuals = true;
                full_sidecar = true;
                lossless_geometry = true;
            }
            "--sh-vq-codebook-size" => {
                sh_vq_codebook_size = Some(parse_value(
                    next_value(&mut args, "--sh-vq-codebook-size")?,
                    "SH VQ codebook size",
                )?);
                full_sidecar = true;
                lossless_geometry = true;
            }
            "--sh-vq-iterations" => {
                sh_vq_iterations = parse_value(
                    next_value(&mut args, "--sh-vq-iterations")?,
                    "SH VQ iterations",
                )?
            }
            "--sh-vq-training-samples" => {
                sh_vq_training_samples = parse_value(
                    next_value(&mut args, "--sh-vq-training-samples")?,
                    "SH VQ training samples",
                )?
            }
            "--sh-video-atlas" => {
                sh_video_atlas = Some(ShVideoPreset::parse(
                    &next_value(&mut args, "--sh-video-atlas")?.to_string_lossy(),
                )?);
                full_sidecar = true;
                lossless_geometry = true;
            }
            "--sh-video-atlas-width" => {
                sh_video_atlas_width = parse_value(
                    next_value(&mut args, "--sh-video-atlas-width")?,
                    "SH video atlas width",
                )?
            }
            "--vpcc-atlas-width" => {
                vpcc_atlas_width = Some(parse_value(
                    next_value(&mut args, "--vpcc-atlas-width")?,
                    "V-PCC atlas width",
                )?)
            }
            "--vpcc-disable-patch-splitting" => vpcc_disable_patch_splitting = true,
            "--export-patch-map" => export_patch_map = true,
            "--benchmark-3dgscompare" => benchmark_3dgscompare = true,
            "--benchmark-python" => {
                benchmark_python = Some(PathBuf::from(next_value(&mut args, "--benchmark-python")?))
            }
            "--benchmark-backend" => {
                benchmark_backend =
                    Some(PathBuf::from(next_value(&mut args, "--benchmark-backend")?))
            }
            "--benchmark-original-scene-id" => {
                benchmark_original_scene_id =
                    next_value(&mut args, "--benchmark-original-scene-id")?
                        .to_string_lossy()
                        .into_owned()
            }
            "--benchmark-camera-bundle-id" => {
                benchmark_camera_bundle_id = next_value(&mut args, "--benchmark-camera-bundle-id")?
                    .to_string_lossy()
                    .into_owned()
            }
            unknown => return Err(format!("unknown argument: {unknown}\n{}", usage()).into()),
        }
    }
    // SH-VQ historically implied the v1 full-record sidecar.  Visual mode
    // intentionally uses its own semantic sidecar instead.
    if visual_sidecar || anchor_grouped_sidecar {
        full_sidecar = false;
    }
    if high_low_geometry {
        packed_geometry_residuals = !entropy_low_bits;
        vpcc_only_geometry = false;
        if !(6..=15).contains(&geometry_bitdepth) {
            return Err("--high-low-geometry requires --geometry-bitdepth between 6 and 15".into());
        }
        geometry_residual_bits = 16 - geometry_bitdepth;
        anchor_grouped_sidecar = true;
        lossless_geometry = true;
        full_sidecar = false;
    }
    if signed_log_geometry && !high_low_geometry {
        return Err("--signed-log-geometry currently requires --high-low-geometry".into());
    }
    // 256 entries is the standard operating point for comparable SH-VQ
    // experiments. Explicit --sh-vq-codebook-size remains an override.
    if (visual_sidecar || anchor_grouped_sidecar) && sh_vq_codebook_size.is_none() {
        sh_vq_codebook_size = Some(256);
    }

    Ok(Cli {
        source,
        output_dir,
        tmc2_root,
        max_points,
        geometry_bitdepth,
        threads,
        rate,
        prepare_only,
        lossless_geometry,
        full_sidecar,
        visual_sidecar,
        anchor_grouped_sidecar,
        collision_only_geometry_residuals,
        packed_geometry_residuals,
        vpcc_only_geometry,
        high_low_geometry,
        signed_log_geometry,
        entropy_low_bits,
        semantic_webp_grid,
        geometry_residual_bits,
        geometry_anchor_residuals,
        sh_vq_codebook_size,
        sh_vq_iterations,
        sh_vq_training_samples,
        sh_video_atlas,
        sh_video_atlas_width,
        vpcc_atlas_width,
        vpcc_disable_patch_splitting,
        export_patch_map,
        benchmark_3dgscompare,
        benchmark_python,
        benchmark_backend,
        benchmark_original_scene_id,
        benchmark_camera_bundle_id,
    })
}

fn usage() -> &'static str {
    "usage:\n  tmc2_adapter <3dgs.ply> <output-dir> [codec options]\n  tmc2_adapter tiled-run <3dgs.ply> <output-dir> [--tiles N] [--tile-jobs N] [codec options]\n  tmc2_adapter spatial-tiles <3dgs.ply> <output-dir> [--tiles N]\n  tmc2_adapter geometry-fidelity <3dgs.ply> <output-dir> [--tmc2-root PATH] [--max-points N] [--bitdepths 10,11,12] [--threads N]\n  tmc2_adapter decode-full <container-dir> <output.ply> [--tmc2-root PATH] [--threads N]\n  tmc2_adapter locality-report <source.ply> <manifest.json> <vpcc_patch_map.csv> <report.json>\n\ncodec options: [--tmc2-root PATH] [--max-points N] [--geometry-bitdepth 8..16] [--threads N] [--rate r1..r5] [--lossless-geometry] [--full-sidecar | --visual-sidecar | --anchor-grouped-sidecar] [--collision-only-geometry-residuals] [--packed-geometry-residuals] [--vpcc-only-geometry] [--high-low-geometry] [--signed-log-geometry] [--entropy-low-bits] [--geometry-residual-bits 10..16] [--geometry-anchor-residuals] [--sh-vq-codebook-size N] [--sh-vq-iterations N] [--sh-vq-training-samples N] [--sh-video-atlas av1|lossless] [--sh-video-atlas-width N] [--vpcc-atlas-width N] [--vpcc-disable-patch-splitting] [--export-patch-map] [--benchmark-3dgscompare] [--benchmark-python PATH] [--benchmark-backend PATH] [--benchmark-original-scene-id ID] [--benchmark-camera-bundle-id ID] [--prepare-only]\n\n--high-low-geometry uses an 8..12-bit V-PCC base plus the complementary low XYZ bits, yielding a 16-bit position grid without duplicated XYZ. --signed-log-geometry applies SOG-web's signed-log1p coordinate domain before that split and records the shared SOG range in the manifest. --entropy-low-bits additionally Morton-sorts low coordinates inside each V-PCC anchor and Deflate-codes their deltas. --vpcc-disable-patch-splitting is an experiment control for sparse high-bit geometry. --anchor-grouped-sidecar and --visual-sidecar default to a 256-entry SH-VQ codebook unless explicitly overridden. tiled-run defaults to two parallel tile jobs and divides --threads between them."
}

fn parse_decode_cli() -> Result<DecodeCli, DynError> {
    let mut args = env::args_os().skip(2);
    let container_dir = PathBuf::from(args.next().ok_or(usage())?);
    let output_ply = PathBuf::from(args.next().ok_or(usage())?);
    let mut tmc2_root = env::var_os("TMC2_ROOT")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join("tools/mpeg-pcc-tmc2")))
        .ok_or("set TMC2_ROOT or pass --tmc2-root")?;
    let mut threads = 8usize;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--tmc2-root" => tmc2_root = PathBuf::from(next_value(&mut args, "--tmc2-root")?),
            "--threads" => threads = parse_value(next_value(&mut args, "--threads")?, "threads")?,
            unknown => return Err(format!("unknown argument: {unknown}\n{}", usage()).into()),
        }
    }
    if threads == 0 {
        return Err("threads must be at least 1".into());
    }
    Ok(DecodeCli {
        container_dir,
        output_ply,
        tmc2_root,
        threads,
    })
}

fn parse_locality_cli() -> Result<LocalityCli, DynError> {
    let mut args = env::args_os().skip(2);
    let source = PathBuf::from(args.next().ok_or(
        "usage: tmc2_adapter locality-report <source.ply> <manifest.json> <vpcc_patch_map.csv> <report.json>",
    )?);
    let manifest_path = PathBuf::from(args.next().ok_or(
        "usage: tmc2_adapter locality-report <source.ply> <manifest.json> <vpcc_patch_map.csv> <report.json>",
    )?);
    let patch_map_path = PathBuf::from(args.next().ok_or(
        "usage: tmc2_adapter locality-report <source.ply> <manifest.json> <vpcc_patch_map.csv> <report.json>",
    )?);
    let report_path = PathBuf::from(args.next().ok_or(
        "usage: tmc2_adapter locality-report <source.ply> <manifest.json> <vpcc_patch_map.csv> <report.json>",
    )?);
    if args.next().is_some() {
        return Err("locality-report received unexpected arguments".into());
    }
    Ok(LocalityCli {
        source,
        manifest_path,
        patch_map_path,
        report_path,
    })
}

fn parse_geometry_fidelity_cli() -> Result<GeometryFidelityCli, DynError> {
    let mut args = env::args_os().skip(2);
    let source = PathBuf::from(args.next().ok_or(usage())?);
    let output_dir = PathBuf::from(args.next().ok_or(usage())?);
    let mut tmc2_root = env::var_os("TMC2_ROOT")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join("tools/mpeg-pcc-tmc2")))
        .ok_or("set TMC2_ROOT or pass --tmc2-root")?;
    let mut max_points = None;
    // The pinned lossless HM profile uses a negative QP and therefore needs
    // at least a 10-bit geometry video input.  This profile has been validated
    // only through 12-bit 3D coordinates; a high-bit-depth HM profile is a
    // separate experiment rather than an implicit lossless assumption.
    let mut geometry_bitdepths = vec![10, 11, 12];
    let mut threads = 8usize;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--tmc2-root" => tmc2_root = PathBuf::from(next_value(&mut args, "--tmc2-root")?),
            "--max-points" => {
                max_points = Some(parse_value(
                    next_value(&mut args, "--max-points")?,
                    "max points",
                )?)
            }
            "--bitdepths" => {
                geometry_bitdepths =
                    parse_geometry_bitdepths(&next_value(&mut args, "--bitdepths")?)?
            }
            "--threads" => threads = parse_value(next_value(&mut args, "--threads")?, "threads")?,
            unknown => return Err(format!("unknown argument: {unknown}\n{}", usage()).into()),
        }
    }

    if !source.is_file() {
        return Err(format!("source PLY does not exist: {}", source.display()).into());
    }
    if max_points == Some(0) || threads == 0 {
        return Err("max points and threads must be at least 1".into());
    }
    Ok(GeometryFidelityCli {
        source,
        output_dir,
        tmc2_root,
        max_points,
        geometry_bitdepths,
        threads,
    })
}

fn parse_geometry_bitdepths(value: &OsString) -> Result<Vec<u8>, DynError> {
    let value = value
        .to_str()
        .ok_or("geometry bit depths are not valid UTF-8")?;
    let mut bitdepths = Vec::new();
    for part in value.split(',') {
        let bitdepth = part.trim().parse::<u8>()?;
        if !(10..=12).contains(&bitdepth) {
            return Err("the pinned lossless profile supports geometry bit depths 10 through 12; high-bit-depth HM configuration is not implemented yet".into());
        }
        if !bitdepths.contains(&bitdepth) {
            bitdepths.push(bitdepth);
        }
    }
    if bitdepths.is_empty() {
        return Err("supply at least one geometry bit depth".into());
    }
    Ok(bitdepths)
}

fn next_value(args: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<OsString, DynError> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn parse_value<T>(value: OsString, label: &str) -> Result<T, DynError>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    Ok(value
        .to_str()
        .ok_or_else(|| format!("{label} is not valid UTF-8"))?
        .parse()?)
}

fn validate_cli(cli: &Cli) -> Result<(), DynError> {
    if !cli.source.is_file() {
        return Err(format!("source PLY does not exist: {}", cli.source.display()).into());
    }
    if !(8..=16).contains(&cli.geometry_bitdepth)
        && !(cli.high_low_geometry && (6..=7).contains(&cli.geometry_bitdepth)) {
        return Err("geometry bitdepth must be between 8 and 16 (6 and 7 also supported in high-low mode)".into());
    }
    if cli.lossless_geometry
        && !(6..=12).contains(&cli.geometry_bitdepth)
        && !cli.vpcc_only_geometry
    {
        return Err(
            "the pinned lossless TMC2 profile accepts geometry bit depths 6 through 12; higher-bit-depth HM configuration is not implemented yet"
                .into(),
        );
    }
    if (cli.full_sidecar || cli.visual_sidecar || cli.anchor_grouped_sidecar)
        && !matches!(
            read_header(&cli.source)?.1.format,
            PlyFormat::BinaryLittleEndian
        )
    {
        return Err("--full-sidecar currently requires binary_little_endian PLY input".into());
    }
    if cli.geometry_anchor_residuals {
        let header = read_header(&cli.source)?.1;
        for name in ["x", "y", "z"] {
            let property = header
                .properties
                .iter()
                .find(|property| property.name == name)
                .ok_or_else(|| format!("geometry anchor residuals require {name}"))?;
            if !matches!(property.scalar_type, ScalarType::F32) {
                return Err("geometry anchor residuals currently require float32 xyz".into());
            }
        }
    }
    if cli.threads == 0 {
        return Err("threads must be at least 1".into());
    }
    if !matches!(cli.rate.as_str(), "r1" | "r2" | "r3" | "r4" | "r5") {
        return Err("rate must be one of r1, r2, r3, r4, r5".into());
    }
    if let Some(0) = cli.max_points {
        return Err("max points must be at least 1".into());
    }
    if let Some(size) = cli.sh_vq_codebook_size {
        if !(2..=u16::MAX as usize + 1).contains(&size) {
            return Err("SH VQ codebook size must be between 2 and 65536".into());
        }
        if cli.sh_vq_iterations == 0 || cli.sh_vq_training_samples == 0 {
            return Err("SH VQ iterations and training samples must be at least 1".into());
        }
    }
    if cli.sh_vq_codebook_size.is_some() && cli.sh_video_atlas.is_some() {
        return Err("choose either SH VQ or SH video atlas, not both".into());
    }
    if cli.visual_sidecar && cli.sh_vq_codebook_size.is_none() {
        return Err("--visual-sidecar requires --sh-vq-codebook-size N".into());
    }
    if cli.anchor_grouped_sidecar && cli.sh_vq_codebook_size.is_none() {
        return Err("--anchor-grouped-sidecar requires --sh-vq-codebook-size N".into());
    }
    if cli.collision_only_geometry_residuals && !cli.anchor_grouped_sidecar {
        return Err("--collision-only-geometry-residuals requires --anchor-grouped-sidecar".into());
    }
    if cli.packed_geometry_residuals && !cli.anchor_grouped_sidecar {
        return Err("--packed-geometry-residuals requires --anchor-grouped-sidecar".into());
    }
    if cli.vpcc_only_geometry && !cli.anchor_grouped_sidecar {
        return Err("--vpcc-only-geometry requires --anchor-grouped-sidecar".into());
    }
    if cli.vpcc_only_geometry
        && (cli.collision_only_geometry_residuals || cli.packed_geometry_residuals)
    {
        return Err("--vpcc-only-geometry cannot be combined with residual geometry flags".into());
    }
    if cli.high_low_geometry && cli.vpcc_only_geometry {
        return Err("--high-low-geometry cannot be combined with --vpcc-only-geometry".into());
    }
    if cli.high_low_geometry
        && (!(6..=15).contains(&cli.geometry_bitdepth)
            || cli.geometry_residual_bits != 16 - cli.geometry_bitdepth
            || (!cli.packed_geometry_residuals && !cli.entropy_low_bits))
    {
        return Err("--high-low-geometry requires a 6..15-bit V-PCC base and a matching packed low-coordinate stream".into());
    }
    if cli.anchor_grouped_sidecar
        && !(10..=16).contains(&cli.geometry_residual_bits)
        && !cli.high_low_geometry
    {
        return Err("geometry residual bits must be between 10 and 16".into());
    }
    if cli.anchor_grouped_sidecar && (cli.sh_video_atlas.is_some() || cli.geometry_anchor_residuals)
    {
        return Err("--anchor-grouped-sidecar currently supports SH VQ only".into());
    }
    if cli.visual_sidecar && cli.sh_video_atlas.is_some() {
        return Err("--visual-sidecar currently supports SH VQ, not the SH video atlas".into());
    }
    if cli.visual_sidecar && cli.geometry_anchor_residuals {
        return Err("--visual-sidecar derives positions from V-PCC anchors and cannot use exact XYZ residuals".into());
    }
    if cli.sh_video_atlas.is_some() && cli.geometry_anchor_residuals {
        return Err("SH video atlas does not yet support geometry anchor residuals".into());
    }
    if cli.sh_video_atlas_width == 0 {
        return Err("SH video atlas width must be at least 1".into());
    }
    if let Some(width) = cli.vpcc_atlas_width {
        if width < 128 || width % 64 != 0 {
            return Err("V-PCC atlas width must be a multiple of 64 and at least 128".into());
        }
    }
    if !cli.prepare_only {
        for relative in [
            "bin/PccAppEncoder",
            "bin/PccAppDecoder",
            "cfg/common/ctc-common.cfg",
            "cfg/condition/ctc-all-intra.cfg",
        ] {
            let path = cli.tmc2_root.join(relative);
            if !path.is_file() {
                return Err(format!("missing TMC2 file: {}", path.display()).into());
            }
        }
        if cli.lossless_geometry {
            for relative in [
                "cfg/common/ctc-common-lossless-geometry.cfg",
                "cfg/condition/ctc-all-intra-lossless-geometry.cfg",
                "cfg/rate/ctc-r5-lossless.cfg",
                "dependencies/HM-16.20+SCM-8.8-app/bin/TAppEncoderStatic",
                "dependencies/HM-16.20+SCM-8.8-app/bin/TAppDecoderStatic",
            ] {
                let path = cli.tmc2_root.join(relative);
                if !path.is_file() {
                    return Err(format!("missing lossless TMC2/HM file: {}", path.display()).into());
                }
            }
        }
    }
    Ok(())
}

fn ensure_empty_output(path: &Path) -> Result<(), DynError> {
    if fs::read_dir(path)?.next().is_some() {
        return Err(format!(
            "output directory is not empty; choose a new experiment directory: {}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn read_header(path: &Path) -> Result<(BufReader<File>, PlyHeader), DynError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut format = None;
    let mut vertex_count = None;
    let mut properties = Vec::new();
    let mut record_size = 0usize;
    let mut current_element = None::<String>;
    let mut saw_signature = false;

    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err("unexpected end of file in PLY header".into());
        }
        let line = line.trim();
        if !saw_signature {
            if line != "ply" {
                return Err("missing PLY signature".into());
            }
            saw_signature = true;
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["format", "binary_little_endian", _] => format = Some(PlyFormat::BinaryLittleEndian),
            ["format", "ascii", _] => format = Some(PlyFormat::Ascii),
            ["format", other, _] => return Err(format!("unsupported PLY format: {other}").into()),
            ["element", name, count] => {
                current_element = Some((*name).to_string());
                if *name == "vertex" {
                    vertex_count = Some(count.parse()?);
                }
            }
            ["property", "list", ..] if current_element.as_deref() == Some("vertex") => {
                return Err("list properties in the vertex element are unsupported".into());
            }
            ["property", scalar, name] if current_element.as_deref() == Some("vertex") => {
                let scalar_type = ScalarType::parse(scalar)?;
                properties.push(Property {
                    name: (*name).to_string(),
                    scalar_type,
                    offset: record_size,
                });
                record_size += scalar_type.size();
            }
            ["end_header"] => break,
            _ => {}
        }
    }

    Ok((
        reader,
        PlyHeader {
            format: format.ok_or("PLY header has no format")?,
            vertex_count: vertex_count.ok_or("PLY header has no vertex element")?,
            properties,
            record_size,
        },
    ))
}

fn required_columns(header: &PlyHeader) -> Result<RequiredColumns, DynError> {
    let find = |name: &str| {
        header
            .properties
            .iter()
            .position(|property| property.name == name)
            .ok_or_else(|| format!("required PLY property missing: {name}"))
    };
    Ok(RequiredColumns {
        xyz: [find("x")?, find("y")?, find("z")?],
        sh_dc: [find("f_dc_0")?, find("f_dc_1")?, find("f_dc_2")?],
    })
}

fn scan_bounds(path: &Path) -> Result<(PlyHeader, [f64; 3], [f64; 3]), DynError> {
    let (mut reader, header) = read_header(path)?;
    if header.vertex_count == 0 {
        return Err("source PLY contains no vertices".into());
    }
    let columns = required_columns(&header)?;
    let mut mins = [f64::INFINITY; 3];
    let mut maxes = [f64::NEG_INFINITY; 3];
    visit_vertices(&mut reader, &header, |_, row| {
        for axis in 0..3 {
            let value = property_value(&header, row, columns.xyz[axis])?;
            if !value.is_finite() {
                return Err(format!("non-finite coordinate in source PLY: {value}").into());
            }
            mins[axis] = mins[axis].min(value);
            maxes[axis] = maxes[axis].max(value);
        }
        Ok(())
    })?;
    Ok((header, mins, maxes))
}

fn locality_report_command(cli: &LocalityCli) -> Result<(), DynError> {
    let manifest: ConversionManifest =
        serde_json::from_reader(BufReader::new(File::open(&cli.manifest_path)?))?;
    let patch_map = read_patch_map(&cli.patch_map_path)?;
    let (mut reader, header) = read_header(&cli.source)?;
    let required = required_columns(&header)?;
    let sh_rest = sh_rest_columns(&header)?;
    let opacity = property_column(&header, "opacity")?;
    let scale = [
        property_column(&header, "scale_0")?,
        property_column(&header, "scale_1")?,
        property_column(&header, "scale_2")?,
    ];
    let rotation = [
        property_column(&header, "rot_0")?,
        property_column(&header, "rot_1")?,
        property_column(&header, "rot_2")?,
        property_column(&header, "rot_3")?,
    ];
    let mut cells = HashMap::<[u32; 3], AtlasCell>::with_capacity(patch_map.len());
    let mut emitted = 0usize;
    let mut unresolved = 0usize;
    visit_vertices(&mut reader, &header, |index, row| {
        if !selected_vertex(
            index,
            emitted,
            header.vertex_count,
            manifest.selected_points,
        ) {
            return Ok(());
        }
        emitted += 1;
        let anchor = quantized_anchor(&header.properties, row, &manifest.transform)?;
        let Some(location) = patch_map.get(&anchor) else {
            unresolved += 1;
            return Ok(());
        };
        let cell = cells.entry(anchor).or_insert_with(|| AtlasCell {
            location: PatchMapEntry {
                patch: location.patch,
                u: location.u,
                v: location.v,
                layer: location.layer,
            },
            count: 0,
            dc: [0.0; 3],
            sh_rest: vec![0.0; sh_rest.len()],
            opacity: 0.0,
            scale: [0.0; 3],
            rotation: [0.0; 4],
        });
        cell.count += 1;
        for axis in 0..3 {
            cell.dc[axis] += property_value(&header, row, required.sh_dc[axis])?;
            cell.scale[axis] += property_value(&header, row, scale[axis])?;
        }
        for coefficient in 0..sh_rest.len() {
            cell.sh_rest[coefficient] += property_value(&header, row, sh_rest[coefficient])?;
        }
        cell.opacity += property_value(&header, row, opacity)?;
        for axis in 0..4 {
            cell.rotation[axis] += property_value(&header, row, rotation[axis])?;
        }
        Ok(())
    })?;
    if emitted != manifest.selected_points {
        return Err(format!(
            "selection mismatch: expected {} source Gaussians, saw {emitted}",
            manifest.selected_points
        )
        .into());
    }
    let mut cells = cells.into_values().collect::<Vec<_>>();
    for cell in &mut cells {
        let divisor = cell.count as f64;
        for value in &mut cell.dc {
            *value /= divisor;
        }
        for value in &mut cell.sh_rest {
            *value /= divisor;
        }
        cell.opacity /= divisor;
        for value in &mut cell.scale {
            *value /= divisor;
        }
        for value in &mut cell.rotation {
            *value /= divisor;
        }
    }
    let mut atlas_index = HashMap::<[u32; 4], usize>::with_capacity(cells.len());
    for (index, cell) in cells.iter().enumerate() {
        atlas_index.insert(
            [
                cell.location.patch,
                cell.location.layer,
                cell.location.u,
                cell.location.v,
            ],
            index,
        );
    }
    let mut atlas_sum = AttributeDistances::default();
    let mut atlas_pairs = 0usize;
    for cell in &cells {
        for (du, dv) in [(1u32, 0u32), (0u32, 1u32)] {
            let key = [
                cell.location.patch,
                cell.location.layer,
                cell.location.u.saturating_add(du),
                cell.location.v.saturating_add(dv),
            ];
            if let Some(&other) = atlas_index.get(&key) {
                add_attribute_distance(&mut atlas_sum, cell, &cells[other]);
                atlas_pairs += 1;
            }
        }
    }
    divide_attribute_distances(&mut atlas_sum, atlas_pairs);
    let mut random_sum = AttributeDistances::default();
    let random_pairs = atlas_pairs.min(cells.len().saturating_mul(8));
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..random_pairs {
        let first = random_index(&mut state, cells.len());
        let mut second = random_index(&mut state, cells.len().saturating_sub(1));
        if second >= first {
            second += 1;
        }
        add_attribute_distance(&mut random_sum, &cells[first], &cells[second]);
    }
    divide_attribute_distances(&mut random_sum, random_pairs);
    let report = LocalityReport {
        schema_version: 1,
        method: "decoded_vpcc_patch_layer_uv_4_neighborhood; attributes are mean-aggregated for source Gaussians sharing one quantized VPCC anchor".into(),
        source_points_selected: manifest.selected_points,
        vpcc_atlas_points: patch_map.len(),
        populated_atlas_points: cells.len(),
        unresolved_source_gaussians: unresolved,
        atlas_neighbor_pairs: atlas_pairs,
        random_pairs,
        atlas_neighbors: atlas_sum,
        random_baseline: random_sum,
    };
    if let Some(parent) = cli.report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&cli.report_path)?);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!(
        "wrote V-PCC patch-locality report to {}",
        cli.report_path.display()
    );
    println!(
        "atlas neighbor pairs={} random pairs={}",
        report.atlas_neighbor_pairs, report.random_pairs
    );
    Ok(())
}

fn property_column(header: &PlyHeader, name: &str) -> Result<usize, DynError> {
    header
        .properties
        .iter()
        .position(|property| property.name == name)
        .ok_or_else(|| format!("required PLY property missing: {name}").into())
}

fn geometry_fidelity_command(cli: &GeometryFidelityCli) -> Result<(), DynError> {
    fs::create_dir_all(&cli.output_dir)?;
    ensure_empty_output(&cli.output_dir)?;
    let mut results = Vec::with_capacity(cli.geometry_bitdepths.len());

    for &geometry_bitdepth in &cli.geometry_bitdepths {
        let output_dir = cli.output_dir.join(format!("bitdepth_{geometry_bitdepth}"));
        let run_cli = Cli {
            source: cli.source.clone(),
            output_dir: output_dir.clone(),
            tmc2_root: cli.tmc2_root.clone(),
            max_points: cli.max_points,
            geometry_bitdepth,
            threads: cli.threads,
            rate: "r5".into(),
            prepare_only: false,
            lossless_geometry: true,
            full_sidecar: false,
            visual_sidecar: false,
            anchor_grouped_sidecar: false,
            collision_only_geometry_residuals: false,
            packed_geometry_residuals: false,
            vpcc_only_geometry: false,
            high_low_geometry: false,
            signed_log_geometry: false,
            entropy_low_bits: false,
            semantic_webp_grid: false,
            geometry_residual_bits: 14,
            geometry_anchor_residuals: false,
            sh_vq_codebook_size: None,
            sh_vq_iterations: 8,
            sh_vq_training_samples: 20_000,
            sh_video_atlas: None,
            sh_video_atlas_width: 2048,
            vpcc_atlas_width: None,
            vpcc_disable_patch_splitting: false,
            export_patch_map: false,
            benchmark_3dgscompare: false,
            benchmark_python: None,
            benchmark_backend: None,
            benchmark_original_scene_id: DEFAULT_3DGSCOMPARE_ORIGINAL_SCENE_ID.into(),
            benchmark_camera_bundle_id: DEFAULT_3DGSCOMPARE_CAMERA_BUNDLE_ID.into(),
        };
        let notifier = discord_bot::DiscordNotifier::disabled(discord_bot::RunContext::new(
            "tmc2_geometry_fidelity",
            run_cli.source.display().to_string(),
            output_dir.display().to_string(),
        ));

        let result = match run(&run_cli, &notifier) {
            Ok(manifest) => {
                let (expected, expected_records) =
                    decoded_anchor_set(&output_dir.join("input/frame_0000.ply"))?;
                let (decoded, decoded_records) =
                    decoded_anchor_set(&output_dir.join("decoded_0000.ply"))?;
                let missing = expected.difference(&decoded).count();
                let unexpected = decoded.difference(&expected).count();
                GeometryFidelityResult {
                    geometry_bitdepth,
                    status: "completed".into(),
                    error: None,
                    selected_points: Some(manifest.selected_points),
                    quantized_unique_points: Some(expected.len()),
                    quantization_collisions: Some(manifest.duplicate_quantized_positions),
                    decoded_records: Some(decoded_records),
                    decoded_unique_points: Some(decoded.len()),
                    missing_points: Some(missing),
                    unexpected_points: Some(unexpected),
                    exact_point_set: Some(
                        expected_records == expected.len()
                            && decoded_records == decoded.len()
                            && expected == decoded,
                    ),
                    vpcc_bytes: manifest.artifacts.compressed_bytes,
                }
            }
            Err(error) => GeometryFidelityResult {
                geometry_bitdepth,
                status: "failed".into(),
                error: Some(error.to_string()),
                selected_points: None,
                quantized_unique_points: None,
                quantization_collisions: None,
                decoded_records: None,
                decoded_unique_points: None,
                missing_points: None,
                unexpected_points: None,
                exact_point_set: None,
                vpcc_bytes: None,
            },
        };
        println!(
            "geometry fidelity bitdepth={} status={} exact_point_set={:?}",
            result.geometry_bitdepth, result.status, result.exact_point_set
        );
        results.push(result);
    }

    let report = GeometryFidelityReport {
        schema_version: 1,
        method: "lossless_tmc2_quantized_point_set_fidelity".into(),
        source_ply: cli.source.display().to_string(),
        max_points: cli.max_points,
        results,
    };
    let report_path = cli.output_dir.join("geometry_fidelity_report.json");
    serde_json::to_writer_pretty(BufWriter::new(File::create(&report_path)?), &report)?;
    println!(
        "wrote geometry fidelity report to {}",
        report_path.display()
    );
    Ok(())
}

fn read_patch_map(path: &Path) -> Result<HashMap<[u32; 3], PatchMapEntry>, DynError> {
    let reader = BufReader::new(File::open(path)?);
    let mut mappings = HashMap::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line_number == 0 {
            if line != "frame,point,x,y,z,patch,u,v,layer" {
                return Err("unexpected V-PCC patch-map header".into());
            }
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 9 {
            return Err(format!("malformed V-PCC patch-map row {}", line_number + 1).into());
        }
        let parse = |field: usize| {
            fields[field]
                .parse::<u32>()
                .map_err(|error| -> DynError { error.into() })
        };
        let anchor = [parse(2)?, parse(3)?, parse(4)?];
        mappings.insert(
            anchor,
            PatchMapEntry {
                patch: parse(5)?,
                u: parse(6)?,
                v: parse(7)?,
                layer: parse(8)?,
            },
        );
    }
    if mappings.is_empty() {
        return Err("V-PCC patch map contains no points".into());
    }
    Ok(mappings)
}

fn add_attribute_distance(sum: &mut AttributeDistances, left: &AtlasCell, right: &AtlasCell) {
    sum.sh_dc_l2_mean += left
        .dc
        .iter()
        .zip(right.dc)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    sum.sh_rest_rmse_mean += (left
        .sh_rest
        .iter()
        .zip(&right.sh_rest)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / left.sh_rest.len() as f64)
        .sqrt();
    sum.opacity_abs_mean += (left.opacity - right.opacity).abs();
    sum.scale_rmse_mean += (left
        .scale
        .iter()
        .zip(right.scale)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    let norm = |rotation: [f64; 4]| {
        rotation
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    };
    let left_norm = norm(left.rotation);
    let right_norm = norm(right.rotation);
    if left_norm > f64::EPSILON && right_norm > f64::EPSILON {
        let dot = left
            .rotation
            .iter()
            .zip(right.rotation)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (left_norm * right_norm);
        sum.rotation_angle_radians_mean += 2.0 * dot.abs().clamp(-1.0, 1.0).acos();
    }
}

fn divide_attribute_distances(distances: &mut AttributeDistances, count: usize) {
    if count == 0 {
        return;
    }
    let divisor = count as f64;
    distances.sh_dc_l2_mean /= divisor;
    distances.sh_rest_rmse_mean /= divisor;
    distances.opacity_abs_mean /= divisor;
    distances.scale_rmse_mean /= divisor;
    distances.rotation_angle_radians_mean /= divisor;
}

fn random_index(state: &mut u64, upper_bound: usize) -> usize {
    if upper_bound == 0 {
        return 0;
    }
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 32) as usize) % upper_bound
}

struct ConversionResult {
    selected_points: usize,
    vpcc_points: usize,
    duplicate_positions: usize,
    transform: TransformManifest,
    sidecar: Option<SidecarManifest>,
}

#[derive(Debug, Clone, Copy)]
struct PreparedPoint {
    xyz: [u32; 3],
    rgb: [u8; 3],
}

fn sh_rest_columns(header: &PlyHeader) -> Result<Vec<usize>, DynError> {
    let mut columns = header
        .properties
        .iter()
        .enumerate()
        .filter_map(|(index, property)| {
            property
                .name
                .strip_prefix("f_rest_")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|suffix| (suffix, index))
        })
        .collect::<Vec<_>>();
    columns.sort_unstable_by_key(|(suffix, _)| *suffix);
    if columns.is_empty() {
        return Err("SH VQ requires f_rest_* properties in the source PLY".into());
    }
    for (expected, (suffix, index)) in columns.iter().enumerate() {
        if *suffix != expected {
            return Err("SH VQ requires contiguous f_rest_0 through f_rest_N properties".into());
        }
        if !matches!(header.properties[*index].scalar_type, ScalarType::F32) {
            return Err("SH VQ currently requires f_rest_* properties to be float32".into());
        }
    }
    Ok(columns.into_iter().map(|(_, index)| index).collect())
}

fn sh_vector(header: &PlyHeader, row: &[u8], columns: &[usize]) -> Vec<f32> {
    columns
        .iter()
        .map(|column| {
            let property = &header.properties[*column];
            let bytes = &row[property.offset..property.offset + 4];
            f32::from_le_bytes(bytes.try_into().unwrap())
        })
        .collect()
}

fn selected_vertex(index: usize, emitted: usize, total: usize, selected: usize) -> bool {
    emitted < selected && index == emitted.saturating_mul(total) / selected
}

fn dequantize_sh(value: u16, params: &ShVideoQuantization) -> f32 {
    if params.max <= params.min {
        params.min
    } else {
        params.min + (value as f32 / u16::MAX as f32) * (params.max - params.min)
    }
}

fn write_sh_video_sidecar(
    source: &Path,
    header: &PlyHeader,
    selected_points: usize,
    sidecar_path: &Path,
    preset: ShVideoPreset,
    width: u32,
) -> Result<SidecarManifest, DynError> {
    let columns = sh_rest_columns(header)?;
    let mut ranges = vec![ShVideoQuantization { min: 0.0, max: 0.0 }; columns.len()];
    let (mut reader, scanned_header) = read_header(source)?;
    let mut emitted = 0usize;
    visit_vertices(&mut reader, &scanned_header, |index, row| {
        if selected_vertex(index, emitted, scanned_header.vertex_count, selected_points) {
            for (stream, column) in columns.iter().enumerate() {
                let property = &scanned_header.properties[*column];
                let value = f32::from_le_bytes(
                    row[property.offset..property.offset + 4]
                        .try_into()
                        .unwrap(),
                );
                if emitted == 0 {
                    ranges[stream] = ShVideoQuantization {
                        min: value,
                        max: value,
                    };
                } else {
                    ranges[stream].min = ranges[stream].min.min(value);
                    ranges[stream].max = ranges[stream].max.max(value);
                }
            }
            emitted += 1;
        }
        Ok(())
    })?;
    if emitted != selected_points {
        return Err("could not scan every selected SH record".into());
    }
    let height = (selected_points as u64).div_ceil(width as u64) as u32;
    let plane_len = width as usize * height as usize;
    let mut streams = vec![vec![0u16; plane_len]; columns.len()];
    let mut encoder = DeflateEncoder::new(File::create(sidecar_path)?, Compression::default());
    let (mut reader, source_header) = read_header(source)?;
    let mut emitted = 0usize;
    let mut squared_error = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    visit_vertices(&mut reader, &source_header, |index, row| {
        if selected_vertex(index, emitted, source_header.vertex_count, selected_points) {
            let mut template = row.to_vec();
            for (stream, column) in columns.iter().enumerate() {
                let property = &source_header.properties[*column];
                let value = f32::from_le_bytes(
                    row[property.offset..property.offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let quantization = QuantParams {
                    min: ranges[stream].min,
                    max: ranges[stream].max,
                };
                let quantized = quantize_sh_rest_scalar(value, quantization);
                streams[stream][emitted] = quantized;
                let error = value as f64 - dequantize_sh(quantized, &ranges[stream]) as f64;
                squared_error += error * error;
                max_abs_error = max_abs_error.max(error.abs());
                template[property.offset..property.offset + 4].fill(0);
            }
            encoder.write_all(&template)?;
            emitted += 1;
        }
        Ok(())
    })?;
    let mut file = encoder.finish()?;
    file.flush()?;
    let parent = sidecar_path
        .parent()
        .ok_or("sidecar path has no parent directory")?;
    let codec = preset.codec_config();
    write_sh_rest_video_atlas_streams(&streams, width, height, selected_points, parent, codec)?;
    let hi_path = parent.join("sh_rest_hi.mkv");
    let lo_path = parent.join("sh_rest_lo.mkv");
    Ok(SidecarManifest {
        codec: "raw_binary_non_sh_records+deflate; sh_rest_raster_video_atlas".into(),
        path: "gaussians.deflate".into(),
        compressed_bytes: fs::metadata(sidecar_path)?.len(),
        compressed_sha256: sha256_file(sidecar_path)?,
        uncompressed_bytes: (selected_points * header.record_size) as u64,
        record_size: header.record_size,
        records: selected_points,
        properties: header.properties.clone(),
        sh_vq: None,
        sh_video: Some(ShVideoManifest {
            preset: preset.as_str().into(),
            codec: codec.codec.into(),
            width,
            height,
            active_len: selected_points,
            vector_dimensions: columns.len(),
            quantization: ranges,
            hi_path: "sh_rest_hi.mkv".into(),
            hi_bytes: fs::metadata(&hi_path)?.len(),
            hi_sha256: sha256_file(&hi_path)?,
            lo_path: "sh_rest_lo.mkv".into(),
            lo_bytes: fs::metadata(&lo_path)?.len(),
            lo_sha256: sha256_file(&lo_path)?,
            rmse: (squared_error / (selected_points * columns.len()) as f64).sqrt(),
            max_abs_error,
        }),
        geometry_anchor_residuals: None,
    })
}

fn visual_semantic_columns(
    header: &PlyHeader,
    sh_columns: &[usize],
) -> Result<Vec<usize>, DynError> {
    let xyz = xyz_columns(&header.properties)?;
    let mut columns = Vec::new();
    for (index, property) in header.properties.iter().enumerate() {
        if xyz.contains(&index) || sh_columns.contains(&index) {
            continue;
        }
        if !matches!(property.scalar_type, ScalarType::F32) {
            return Err(format!(
                "visual sidecar currently requires float32 semantic properties; {} is not float32",
                property.name
            )
            .into());
        }
        columns.push(index);
    }
    Ok(columns)
}

/// Writes the v2 renderer-oriented payload.  The V-PCC anchor is referenced
/// by compact index, SH-rest is represented by the supplied VQ codebook, and
/// every other float property is independently quantized to u16.  No source
/// PLY row is copied into this container.
fn write_visual_sidecar(
    source: &Path,
    header: &PlyHeader,
    selected_points: usize,
    transform: &TransformManifest,
    sh_vq: &ShVqConfig,
    sidecar_path: &Path,
) -> Result<VisualSidecarManifest, DynError> {
    let semantic_columns = visual_semantic_columns(header, &sh_vq.columns)?;
    let mut ranges = vec![ShVideoQuantization { min: 0.0, max: 0.0 }; semantic_columns.len()];
    let mut anchors = HashSet::new();
    let (mut reader, scan_header) = read_header(source)?;
    let mut emitted = 0usize;
    visit_vertices(&mut reader, &scan_header, |index, row| {
        if emitted >= selected_points {
            return Ok(());
        }
        if selected_vertex(index, emitted, scan_header.vertex_count, selected_points) {
            anchors.insert(quantized_anchor(&scan_header.properties, row, transform)?);
            for (stream, column) in semantic_columns.iter().enumerate() {
                let value = property_value(&scan_header, row, *column)? as f32;
                if emitted == 0 {
                    ranges[stream] = ShVideoQuantization {
                        min: value,
                        max: value,
                    };
                } else {
                    ranges[stream].min = ranges[stream].min.min(value);
                    ranges[stream].max = ranges[stream].max.max(value);
                }
            }
            emitted += 1;
        }
        Ok(())
    })?;
    if emitted != selected_points {
        return Err("could not scan every selected visual-sidecar record".into());
    }
    let mut anchors = anchors.into_iter().collect::<Vec<_>>();
    anchors.sort_unstable();
    let anchor_index_bytes = compact_index_bytes(anchors.len());
    let anchor_set_sha256 = sha256_anchor_set(&anchors);
    let parent = sidecar_path
        .parent()
        .ok_or("visual sidecar path has no parent")?;
    let codebook_path = parent.join("visual_sh_vq_codebook.bin");
    let indices_path = parent.join("visual_sh_vq_indices.deflate");
    let mut codebook_writer = BufWriter::new(File::create(&codebook_path)?);
    for value in &sh_vq.codebook {
        codebook_writer.write_all(&value.to_le_bytes())?;
    }
    codebook_writer.flush()?;

    let mut semantic_encoder =
        DeflateEncoder::new(File::create(sidecar_path)?, Compression::default());
    let mut index_encoder =
        DeflateEncoder::new(File::create(&indices_path)?, Compression::default());
    let (mut reader, data_header) = read_header(source)?;
    let mut emitted = 0usize;
    let mut squared_error = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    visit_vertices(&mut reader, &data_header, |index, row| {
        if emitted >= selected_points {
            return Ok(());
        }
        if selected_vertex(index, emitted, data_header.vertex_count, selected_points) {
            let anchor = quantized_anchor(&data_header.properties, row, transform)?;
            let anchor_id = anchors
                .binary_search(&anchor)
                .map_err(|_| "visual anchor missing")?;
            write_compact_index(&mut semantic_encoder, anchor_id, anchor_index_bytes)?;
            for (stream, column) in semantic_columns.iter().enumerate() {
                let params = QuantParams {
                    min: ranges[stream].min,
                    max: ranges[stream].max,
                };
                let value = property_value(&data_header, row, *column)? as f32;
                semantic_encoder
                    .write_all(&quantize_sh_rest_scalar(value, params).to_le_bytes())?;
            }
            let vector = sh_vector(&data_header, row, &sh_vq.columns);
            let (entry, _) = nearest_codebook_entry(&vector, &sh_vq.codebook, sh_vq.columns.len());
            let centroid =
                &sh_vq.codebook[entry * sh_vq.columns.len()..(entry + 1) * sh_vq.columns.len()];
            for (value, reconstructed) in vector.iter().zip(centroid) {
                let error = *value as f64 - *reconstructed as f64;
                squared_error += error * error;
                max_abs_error = max_abs_error.max(error.abs());
            }
            write_compact_index(
                &mut index_encoder,
                entry,
                compact_index_bytes(sh_vq.codebook_size),
            )?;
            emitted += 1;
        }
        Ok(())
    })?;
    if emitted != selected_points {
        return Err("could not write every selected visual-sidecar record".into());
    }
    let mut file = semantic_encoder.finish()?;
    file.flush()?;
    let mut file = index_encoder.finish()?;
    file.flush()?;
    Ok(VisualSidecarManifest {
        codec: "vpcc_anchor_indices+u16_semantic_quantization+sh_rest_vector_quantization".into(),
        path: "gaussians.deflate".into(),
        compressed_bytes: fs::metadata(sidecar_path)?.len(),
        compressed_sha256: sha256_file(sidecar_path)?,
        records: selected_points,
        record_size: header.record_size,
        properties: header.properties.clone(),
        anchor_index_bytes,
        anchor_set_sha256,
        semantic_columns,
        quantization: ranges,
        sh_vq: ShVqManifest {
            codebook_entries: sh_vq.codebook_size,
            vector_dimensions: sh_vq.columns.len(),
            index_bytes: compact_index_bytes(sh_vq.codebook_size),
            indices_path: "visual_sh_vq_indices.deflate".into(),
            indices_compressed_bytes: fs::metadata(&indices_path)?.len(),
            indices_compressed_sha256: sha256_file(&indices_path)?,
            indices_uncompressed_bytes: (selected_points * compact_index_bytes(sh_vq.codebook_size))
                as u64,
            codebook_path: "visual_sh_vq_codebook.bin".into(),
            codebook_bytes: fs::metadata(&codebook_path)?.len(),
            codebook_sha256: sha256_file(&codebook_path)?,
            training_samples: sh_vq.training_samples,
            kmeans_iterations: sh_vq.iterations,
            rmse: (squared_error / (selected_points * sh_vq.columns.len()) as f64).sqrt(),
            max_abs_error,
        },
        anchor_groups: None,
        semantic_webp: None,
    })
}

#[derive(Debug)]
struct AnchorGroupedRecord {
    anchor_id: usize,
    residual: [u16; 3],
    semantic: [u16; 14],
    vq_index: u16,
}

fn anchor_morton(anchor: [u32; 3], bits: u8) -> u64 {
    let mut code = 0u64;
    for bit in 0..bits {
        let shift = u32::from(bit) * 3;
        code |= ((u64::from(anchor[0]) >> bit) & 1) << shift;
        code |= ((u64::from(anchor[1]) >> bit) & 1) << (shift + 1);
        code |= ((u64::from(anchor[2]) >> bit) & 1) << (shift + 2);
    }
    code
}

fn anchor_from_morton(code: u32, bits: u8) -> [u16; 3] {
    let mut anchor = [0u16; 3];
    for bit in 0..bits {
        let shift = u32::from(bit) * 3;
        for axis in 0..3 {
            anchor[axis] |= (((code >> (shift + axis as u32)) & 1) as u16) << bit;
        }
    }
    anchor
}

fn sort_anchors_morton(anchors: &mut [[u32; 3]], bits: u8) {
    anchors.sort_unstable_by_key(|anchor| (anchor_morton(*anchor, bits), *anchor));
}

fn quantize_local_residual(
    value: f32,
    anchor: u32,
    axis: usize,
    transform: &TransformManifest,
    bits: u8,
) -> u16 {
    let cell = transform.inverse_scale_to_source as f32;
    let base = anchor_base_coordinate(anchor, axis, transform);
    let max = (1u32 << bits) - 1;
    (((value - base + cell * 0.5) / cell) * max as f32)
        .round()
        .clamp(0.0, max as f32) as u16
}

fn dequantize_local_residual(
    value: u16,
    anchor: u32,
    axis: usize,
    transform: &TransformManifest,
    bits: u8,
) -> f32 {
    let cell = transform.inverse_scale_to_source as f32;
    let max = (1u32 << bits) - 1;
    anchor_base_coordinate(anchor, axis, transform) + (value as f32 / max as f32 - 0.5) * cell
}

fn write_varint(writer: &mut DeflateEncoder<File>, mut value: u32) -> Result<(), DynError> {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        writer.write_all(&[byte])?;
        if value == 0 {
            return Ok(());
        }
    }
}

fn pack_geometry_residuals(records: &[[u16; 3]], bits: u8) -> Vec<u8> {
    let mut packed = Vec::with_capacity((records.len() * 3 * usize::from(bits) + 7) / 8);
    let mut accumulator = 0u64;
    let mut pending_bits = 0u8;
    for record in records {
        for value in record {
            accumulator |= u64::from(*value) << pending_bits;
            pending_bits += bits;
            while pending_bits >= 8 {
                packed.push(accumulator as u8);
                accumulator >>= 8;
                pending_bits -= 8;
            }
        }
    }
    if pending_bits != 0 {
        packed.push(accumulator as u8);
    }
    packed
}

struct PackedResidualReader {
    bytes: Vec<u8>,
    bit_offset: usize,
    bits: u8,
}

impl PackedResidualReader {
    fn new(bytes: Vec<u8>, bits: u8) -> Self {
        Self {
            bytes,
            bit_offset: 0,
            bits,
        }
    }

    fn read(&mut self) -> Result<u16, DynError> {
        let end = self.bit_offset + usize::from(self.bits);
        if end > self.bytes.len() * 8 {
            return Err("packed geometry residual stream ended early".into());
        }
        let byte_offset = self.bit_offset / 8;
        let shift = self.bit_offset % 8;
        let mut window = 0u64;
        for index in 0..4 {
            if let Some(byte) = self.bytes.get(byte_offset + index) {
                window |= u64::from(*byte) << (index * 8);
            }
        }
        let mask = (1u64 << self.bits) - 1;
        self.bit_offset = end;
        Ok(((window >> shift) & mask) as u16)
    }

    fn validate_padding(&self) -> Result<(), DynError> {
        for bit in self.bit_offset..self.bytes.len() * 8 {
            if self.bytes[bit / 8] & (1 << (bit % 8)) != 0 {
                return Err("packed geometry residual stream has nonzero trailing bits".into());
            }
        }
        Ok(())
    }
}

fn write_anchor_grouped_sidecar(
    source: &Path,
    header: &PlyHeader,
    selected_points: usize,
    transform: &TransformManifest,
    sh_vq: &ShVqConfig,
    residual_bits: u8,
    collision_only_residuals: bool,
    packed_residuals: bool,
    vpcc_only_geometry: bool,
    high_low_geometry: bool,
    entropy_low_bits: bool,
    vpcc_geometry_bitdepth: u8,
    sidecar_path: &Path,
) -> Result<VisualSidecarManifest, DynError> {
    let semantic_columns = visual_semantic_columns(header, &sh_vq.columns)?;
    if semantic_columns.len() != 14 {
        return Err(
            "anchor-grouped sidecar currently expects the standard 14 non-SH float properties"
                .into(),
        );
    }
    let mut ranges = vec![ShVideoQuantization { min: 0.0, max: 0.0 }; 14];
    let mut anchor_set = HashSet::new();
    let (mut reader, scanned) = read_header(source)?;
    let mut emitted = 0usize;
    visit_vertices(&mut reader, &scanned, |index, row| {
        if selected_vertex(index, emitted, scanned.vertex_count, selected_points) {
            let position = quantized_anchor(&scanned.properties, row, transform)?;
            let anchor = if high_low_geometry {
                let low_bits = transform.geometry_bitdepth - vpcc_geometry_bitdepth;
                position.map(|value| value >> low_bits)
            } else {
                position
            };
            anchor_set.insert(anchor);
            for (stream, column) in semantic_columns.iter().enumerate() {
                let value = property_value(&scanned, row, *column)? as f32;
                if emitted == 0 {
                    ranges[stream] = ShVideoQuantization {
                        min: value,
                        max: value,
                    };
                } else {
                    ranges[stream].min = ranges[stream].min.min(value);
                    ranges[stream].max = ranges[stream].max.max(value);
                }
            }
            emitted += 1;
        }
        Ok(())
    })?;
    let mut anchors = anchor_set.into_iter().collect::<Vec<_>>();
    sort_anchors_morton(
        &mut anchors,
        if high_low_geometry {
            vpcc_geometry_bitdepth
        } else {
            transform.geometry_bitdepth
        },
    );
    let anchor_set_sha256 = sha256_anchor_set(&anchors);
    let anchor_ids = anchors
        .iter()
        .enumerate()
        .map(|(i, anchor)| (*anchor, i))
        .collect::<HashMap<_, _>>();
    let xyz = xyz_columns(&header.properties)?;
    let (mut reader, data_header) = read_header(source)?;
    let mut records = Vec::with_capacity(selected_points);
    let mut emitted = 0usize;
    let mut squared_error = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    visit_vertices(&mut reader, &data_header, |index, row| {
        if selected_vertex(index, emitted, data_header.vertex_count, selected_points) {
            let full_position = quantized_anchor(&data_header.properties, row, transform)?;
            let anchor = if high_low_geometry {
                let low_bits = transform.geometry_bitdepth - vpcc_geometry_bitdepth;
                full_position.map(|value| value >> low_bits)
            } else {
                full_position
            };
            let anchor_id = *anchor_ids
                .get(&anchor)
                .ok_or("anchor-grouped record has unknown anchor")?;
            let mut residual = [0u16; 3];
            for axis in 0..3 {
                residual[axis] = if high_low_geometry {
                    full_position[axis] as u16 & ((1u16 << residual_bits) - 1)
                } else {
                    quantize_local_residual(
                        property_value(&data_header, row, xyz[axis])? as f32,
                        anchor[axis],
                        axis,
                        transform,
                        residual_bits,
                    )
                };
            }
            let mut semantic = [0u16; 14];
            for (stream, column) in semantic_columns.iter().enumerate() {
                semantic[stream] = quantize_sh_rest_scalar(
                    property_value(&data_header, row, *column)? as f32,
                    QuantParams {
                        min: ranges[stream].min,
                        max: ranges[stream].max,
                    },
                );
            }
            let vector = sh_vector(&data_header, row, &sh_vq.columns);
            let (entry, _) = nearest_codebook_entry(&vector, &sh_vq.codebook, sh_vq.columns.len());
            let centroid =
                &sh_vq.codebook[entry * sh_vq.columns.len()..(entry + 1) * sh_vq.columns.len()];
            for (value, reconstructed) in vector.iter().zip(centroid) {
                let error = *value as f64 - *reconstructed as f64;
                squared_error += error * error;
                max_abs_error = max_abs_error.max(error.abs());
            }
            records.push(AnchorGroupedRecord {
                anchor_id,
                residual,
                semantic,
                vq_index: entry as u16,
            });
            emitted += 1;
        }
        Ok(())
    })?;
    records.sort_unstable_by_key(|record| {
        let low_morton = if entropy_low_bits {
            anchor_morton(
                [
                    u32::from(record.residual[0]),
                    u32::from(record.residual[1]),
                    u32::from(record.residual[2]),
                ],
                residual_bits,
            )
        } else {
            0
        };
        (record.anchor_id, low_morton)
    });
    let parent = sidecar_path
        .parent()
        .ok_or("anchor-grouped sidecar path has no parent")?;
    let counts_path = parent.join("anchor_counts.deflate");
    let residual_path = parent.join("geometry_residuals.deflate");
    let index_path = parent.join("anchor_grouped_sh_vq_indices.deflate");
    let codebook_path = parent.join("anchor_grouped_sh_vq_codebook.bin");
    let mut counts = DeflateEncoder::new(File::create(&counts_path)?, Compression::default());
    let mut cursor = 0usize;
    let mut group_counts = vec![0usize; anchors.len()];
    for anchor_id in 0..anchors.len() {
        let start = cursor;
        while cursor < records.len() && records[cursor].anchor_id == anchor_id {
            cursor += 1;
        }
        group_counts[anchor_id] = cursor - start;
        write_varint(&mut counts, group_counts[anchor_id] as u32)?;
    }
    counts.finish()?;
    let mut semantic_file =
        DeflateEncoder::new(File::create(sidecar_path)?, Compression::default());
    let mut residual_file =
        DeflateEncoder::new(File::create(&residual_path)?, Compression::default());
    let mut index_file = DeflateEncoder::new(File::create(&index_path)?, Compression::default());
    let mut packed_residual_records = Vec::new();
    for record in &records {
        for value in record.semantic {
            semantic_file.write_all(&value.to_le_bytes())?;
        }
        if !vpcc_only_geometry
            && (!collision_only_residuals || group_counts[record.anchor_id] > 1)
        {
            if entropy_low_bits {
                // Written below as Morton-code deltas per coarse V-PCC anchor.
            } else if packed_residuals {
                packed_residual_records.push(record.residual);
            } else {
                for value in record.residual {
                    residual_file.write_all(&value.to_le_bytes())?;
                }
            }
        }
        write_compact_index(
            &mut index_file,
            record.vq_index as usize,
            compact_index_bytes(sh_vq.codebook_size),
        )?;
    }
    if packed_residuals {
        residual_file.write_all(&pack_geometry_residuals(
            &packed_residual_records,
            residual_bits,
        ))?;
    }
    if entropy_low_bits {
        let mut cursor = 0usize;
        for &group_count in &group_counts {
            let mut previous = 0u32;
            for record in &records[cursor..cursor + group_count] {
                let code = anchor_morton(
                    [
                        u32::from(record.residual[0]),
                        u32::from(record.residual[1]),
                        u32::from(record.residual[2]),
                    ],
                    residual_bits,
                ) as u32;
                write_varint(&mut residual_file, code - previous)?;
                previous = code;
            }
            cursor += group_count;
        }
    }
    semantic_file.finish()?;
    residual_file.finish()?;
    index_file.finish()?;
    let mut codebook = BufWriter::new(File::create(&codebook_path)?);
    for value in &sh_vq.codebook {
        codebook.write_all(&value.to_le_bytes())?;
    }
    codebook.flush()?;
    Ok(VisualSidecarManifest {
        codec: if entropy_low_bits {
            format!(
                "vpcc_{}bit_anchor_groups+morton_delta_entropy_coded_{}bit_xyz_low_coordinates+u16_semantic_quantization+sh_rest_vector_quantization",
                vpcc_geometry_bitdepth, residual_bits
            )
        } else if high_low_geometry {
            format!(
                "vpcc_{}bit_anchor_groups+packed_{}bit_xyz_low_coordinates+u16_semantic_quantization+sh_rest_vector_quantization",
                vpcc_geometry_bitdepth, residual_bits
            )
        } else if vpcc_only_geometry {
            "vpcc_anchor_groups+vpcc_only_geometry+u16_semantic_quantization+sh_rest_vector_quantization".to_string()
        } else if packed_residuals {
            "vpcc_anchor_groups+bitpacked_quantized_local_xyz_residuals+u16_semantic_quantization+sh_rest_vector_quantization".to_string()
        } else if collision_only_residuals {
            "vpcc_anchor_groups+collision_only_quantized_local_xyz_residuals+u16_semantic_quantization+sh_rest_vector_quantization".to_string()
        } else {
            "vpcc_anchor_groups+quantized_local_xyz_residuals+u16_semantic_quantization+sh_rest_vector_quantization".to_string()
        },
        path: "gaussians.deflate".into(), compressed_bytes: fs::metadata(sidecar_path)?.len(), compressed_sha256: sha256_file(sidecar_path)?, records: selected_points, record_size: header.record_size, properties: header.properties.clone(), anchor_index_bytes: 0, anchor_set_sha256, semantic_columns, quantization: ranges,
        sh_vq: ShVqManifest { codebook_entries: sh_vq.codebook_size, vector_dimensions: sh_vq.columns.len(), index_bytes: compact_index_bytes(sh_vq.codebook_size), indices_path: "anchor_grouped_sh_vq_indices.deflate".into(), indices_compressed_bytes: fs::metadata(&index_path)?.len(), indices_compressed_sha256: sha256_file(&index_path)?, indices_uncompressed_bytes: (selected_points * compact_index_bytes(sh_vq.codebook_size)) as u64, codebook_path: "anchor_grouped_sh_vq_codebook.bin".into(), codebook_bytes: fs::metadata(&codebook_path)?.len(), codebook_sha256: sha256_file(&codebook_path)?, training_samples: sh_vq.training_samples, kmeans_iterations: sh_vq.iterations, rmse: (squared_error / (selected_points * sh_vq.columns.len()) as f64).sqrt(), max_abs_error },
        anchor_groups: Some(AnchorGroupsManifest { order: "morton_quantized_xyz".into(), counts_path: "anchor_counts.deflate".into(), counts_bytes: fs::metadata(&counts_path)?.len(), counts_sha256: sha256_file(&counts_path)?, residual_path: "geometry_residuals.deflate".into(), residual_bytes: fs::metadata(&residual_path)?.len(), residual_sha256: sha256_file(&residual_path)?, residual_bits, residuals_for_singletons: !collision_only_residuals && !vpcc_only_geometry, residual_records: if vpcc_only_geometry { 0 } else { records.iter().filter(|record| !collision_only_residuals || group_counts[record.anchor_id] > 1).count() }, residuals_bitpacked: packed_residuals, residuals_for_collisions: !vpcc_only_geometry, high_low_geometry, entropy_low_bits }),
        semantic_webp: None,
    })
}

fn encode_semantic_webp_grid(
    output_dir: &Path,
    sidecar_path: &Path,
    visual: &VisualSidecarManifest,
) -> Result<SemanticWebpManifest, DynError> {
    let bootstrap = output_dir.join("manifest.json");
    serde_json::to_writer(
        BufWriter::new(File::create(&bootstrap)?),
        &serde_json::json!({"artifacts": {"visual_sidecar": visual}}),
    )?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/semantic_webp_grid.py");
    let python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(Path::parent)
        .ok_or("cannot resolve repository root")?
        .join("evaluation/venv/bin/python");
    let grid_dir = output_dir.join("semantic_webp");
    let status = Command::new(&python).args([script.as_os_str(), output_dir.as_os_str(), grid_dir.as_os_str()]).status()?;
    let _ = fs::remove_file(&bootstrap);
    if !status.success() { return Err("semantic WebP grid encoder failed".into()); }
    let grid_manifest = grid_dir.join("manifest.json");
    let value: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(&grid_manifest)?))?;
    let total_bytes = value["total_bytes"].as_u64().ok_or("semantic WebP manifest has no total_bytes")?;
    let _ = sidecar_path;
    Ok(SemanticWebpManifest { path: "semantic_webp/manifest.json".into(), total_bytes, manifest_sha256: sha256_file(&grid_manifest)? })
}

fn decode_semantic_webp_grid(container_dir: &Path, grid: &SemanticWebpManifest) -> Result<PathBuf, DynError> {
    let grid_manifest = resolve_artifact(container_dir, &grid.path);
    if sha256_file(&grid_manifest)? != grid.manifest_sha256 { return Err("semantic WebP manifest checksum mismatch".into()); }
    let grid_dir = grid_manifest.parent().ok_or("semantic WebP manifest has no parent")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/semantic_webp_decode.py");
    let python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(Path::parent)
        .ok_or("cannot resolve repository root")?
        .join("evaluation/venv/bin/python");
    let temporary = container_dir.join("semantic_webp_decoded.deflate");
    let status = Command::new(&python).args([script.as_os_str(), grid_dir.as_os_str(), temporary.as_os_str()]).status()?;
    if !status.success() { return Err("semantic WebP grid decoder failed".into()); }
    Ok(temporary)
}

fn nearest_codebook_entry(vector: &[f32], codebook: &[f32], dimensions: usize) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best_distance = f64::INFINITY;
    for (index, candidate) in codebook.chunks_exact(dimensions).enumerate() {
        let mut distance = 0.0_f64;
        for (value, centroid) in vector.iter().zip(candidate) {
            let delta = *value as f64 - *centroid as f64;
            distance += delta * delta;
        }
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    (best_index, best_distance)
}

fn train_sh_vq_codebook(
    source: &Path,
    header: &PlyHeader,
    codebook_size: usize,
    iterations: usize,
    training_samples: usize,
) -> Result<ShVqConfig, DynError> {
    let columns = sh_rest_columns(header)?;
    let dimensions = columns.len();
    let target_samples = training_samples.min(header.vertex_count);
    if codebook_size > target_samples {
        return Err(format!(
            "SH VQ codebook size {codebook_size} exceeds available training samples {target_samples}"
        )
        .into());
    }
    let (mut reader, training_header) = read_header(source)?;
    let mut samples = Vec::with_capacity(target_samples * dimensions);
    let mut emitted = 0usize;
    visit_vertices(&mut reader, &training_header, |index, row| {
        if emitted >= target_samples {
            return Ok(());
        }
        let target_index = emitted.saturating_mul(training_header.vertex_count) / target_samples;
        if index == target_index {
            samples.extend(sh_vector(&training_header, row, &columns));
            emitted += 1;
        }
        Ok(())
    })?;
    if emitted != target_samples {
        return Err("could not collect the requested SH VQ training samples".into());
    }

    let mut codebook = Vec::with_capacity(codebook_size * dimensions);
    for entry in 0..codebook_size {
        let sample = entry * target_samples / codebook_size;
        codebook.extend_from_slice(&samples[sample * dimensions..(sample + 1) * dimensions]);
    }
    for _ in 0..iterations {
        let mut sums = vec![0.0_f64; codebook.len()];
        let mut counts = vec![0usize; codebook_size];
        for sample in samples.chunks_exact(dimensions) {
            let (entry, _) = nearest_codebook_entry(sample, &codebook, dimensions);
            counts[entry] += 1;
            for dimension in 0..dimensions {
                sums[entry * dimensions + dimension] += sample[dimension] as f64;
            }
        }
        for entry in 0..codebook_size {
            if counts[entry] == 0 {
                continue;
            }
            for dimension in 0..dimensions {
                codebook[entry * dimensions + dimension] =
                    (sums[entry * dimensions + dimension] / counts[entry] as f64) as f32;
            }
        }
    }

    Ok(ShVqConfig {
        codebook_size,
        iterations,
        training_samples: target_samples,
        columns,
        codebook,
    })
}

fn compact_index_bytes(entry_count: usize) -> usize {
    if entry_count <= u8::MAX as usize + 1 {
        1
    } else if entry_count <= 1 << 16 {
        2
    } else if entry_count <= 1 << 24 {
        3
    } else {
        4
    }
}

fn write_compact_index(
    writer: &mut DeflateEncoder<File>,
    value: usize,
    bytes: usize,
) -> Result<(), DynError> {
    let encoded = (value as u32).to_le_bytes();
    writer.write_all(&encoded[..bytes])?;
    Ok(())
}

fn anchor_base_coordinate(anchor: u32, axis: usize, transform: &TransformManifest) -> f32 {
    dequantize_coordinate(anchor, axis, transform) as f32
}

/// SOG-web's `signed-exp` decoder remapping has the encoder-side inverse
/// `sign(x) * ln(1 + abs(x))`.  We use the same shared scalar range for all
/// three axes before V-PCC's high/low split.
fn signed_log1p(value: f64) -> f64 {
    value.signum() * value.abs().ln_1p()
}

fn signed_expm1(value: f64) -> f64 {
    value.signum() * value.abs().exp_m1()
}

fn quantize_coordinate(value: f64, axis: usize, transform: &TransformManifest) -> u32 {
    let domain = if transform.signed_log_coordinates {
        signed_log1p(value)
    } else {
        value - transform.source_min[axis]
    };
    let offset = if transform.signed_log_coordinates {
        transform.coordinate_domain_min.unwrap_or(0.0)
    } else {
        0.0
    };
    ((domain - offset) * transform.scale_to_quantized)
        .round()
        .clamp(0.0, transform.quantized_max as f64) as u32
}

fn dequantize_coordinate(anchor: u32, axis: usize, transform: &TransformManifest) -> f64 {
    let domain = anchor as f64 / transform.scale_to_quantized
        + if transform.signed_log_coordinates {
            transform.coordinate_domain_min.unwrap_or(0.0)
        } else {
            0.0
        };
    if transform.signed_log_coordinates {
        signed_expm1(domain)
    } else {
        domain + transform.source_min[axis]
    }
}

fn write_anchor_residual_sidecar(
    source: &Path,
    header: &PlyHeader,
    selected_points: usize,
    transform: &TransformManifest,
    sorted_anchors: &[PreparedPoint],
    sidecar_path: &Path,
    sh_vq: Option<&ShVqConfig>,
) -> Result<SidecarManifest, DynError> {
    let parent = sidecar_path
        .parent()
        .ok_or("sidecar path has no parent directory")?;
    let anchor_indices_path = parent.join("geometry_anchor_indices.deflate");
    let geometry_xor_path = parent.join("geometry_anchor_xor.deflate");
    let codebook_path = parent.join("sh_vq_codebook.bin");
    let sh_indices_path = parent.join("sh_vq_indices.deflate");
    let anchor_index_bytes = compact_index_bytes(sorted_anchors.len());
    let xyz = xyz_columns(&header.properties)?;

    if let Some(config) = sh_vq {
        let mut writer = BufWriter::new(File::create(&codebook_path)?);
        for value in &config.codebook {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
    }

    let mut template_encoder =
        DeflateEncoder::new(File::create(sidecar_path)?, Compression::default());
    let mut anchor_encoder =
        DeflateEncoder::new(File::create(&anchor_indices_path)?, Compression::default());
    let mut xor_encoder =
        DeflateEncoder::new(File::create(&geometry_xor_path)?, Compression::default());
    let mut sh_index_encoder = match sh_vq {
        Some(_) => Some(DeflateEncoder::new(
            File::create(&sh_indices_path)?,
            Compression::default(),
        )),
        None => None,
    };
    let (mut reader, sidecar_header) = read_header(source)?;
    let mut emitted = 0usize;
    let mut sh_squared_error = 0.0_f64;
    let mut sh_max_abs_error = 0.0_f64;
    visit_vertices(&mut reader, &sidecar_header, |index, row| {
        if emitted >= selected_points {
            return Ok(());
        }
        let target_index = emitted.saturating_mul(sidecar_header.vertex_count) / selected_points;
        if index != target_index {
            return Ok(());
        }
        let anchor = quantized_anchor(&sidecar_header.properties, row, transform)?;
        let anchor_id = sorted_anchors
            .binary_search_by_key(&anchor, |point| point.xyz)
            .map_err(|_| "source anchor was absent from the V-PCC anchor table")?;
        write_compact_index(&mut anchor_encoder, anchor_id, anchor_index_bytes)?;
        let mut template = row.to_vec();
        for axis in 0..3 {
            let property = &sidecar_header.properties[xyz[axis]];
            let raw = u32::from_le_bytes(
                row[property.offset..property.offset + 4]
                    .try_into()
                    .unwrap(),
            );
            let base = anchor_base_coordinate(anchor[axis], axis, transform).to_bits();
            xor_encoder.write_all(&(raw ^ base).to_le_bytes())?;
            template[property.offset..property.offset + 4].fill(0);
        }
        if let Some(config) = sh_vq {
            let vector = sh_vector(&sidecar_header, row, &config.columns);
            let (entry, _) =
                nearest_codebook_entry(&vector, &config.codebook, config.columns.len());
            let centroid =
                &config.codebook[entry * config.columns.len()..(entry + 1) * config.columns.len()];
            for (value, reconstructed) in vector.iter().zip(centroid) {
                let error = *value as f64 - *reconstructed as f64;
                sh_squared_error += error * error;
                sh_max_abs_error = sh_max_abs_error.max(error.abs());
            }
            let index_bytes = compact_index_bytes(config.codebook_size);
            write_compact_index(
                sh_index_encoder
                    .as_mut()
                    .ok_or("SH VQ index encoder was not initialized")?,
                entry,
                index_bytes,
            )?;
            for column in &config.columns {
                let property = &sidecar_header.properties[*column];
                template[property.offset..property.offset + 4].fill(0);
            }
        }
        template_encoder.write_all(&template)?;
        emitted += 1;
        Ok(())
    })?;
    if emitted != selected_points {
        return Err("could not write every selected record to the geometry sidecar".into());
    }
    let mut file = template_encoder.finish()?;
    file.flush()?;
    let mut file = anchor_encoder.finish()?;
    file.flush()?;
    let mut file = xor_encoder.finish()?;
    file.flush()?;
    if let Some(encoder) = sh_index_encoder {
        let mut file = encoder.finish()?;
        file.flush()?;
    }
    let sh_vq_manifest = match sh_vq {
        Some(config) => Some(ShVqManifest {
            codebook_entries: config.codebook_size,
            vector_dimensions: config.columns.len(),
            index_bytes: compact_index_bytes(config.codebook_size),
            indices_path: "sh_vq_indices.deflate".into(),
            indices_compressed_bytes: fs::metadata(&sh_indices_path)?.len(),
            indices_compressed_sha256: sha256_file(&sh_indices_path)?,
            indices_uncompressed_bytes: (selected_points
                * compact_index_bytes(config.codebook_size))
                as u64,
            codebook_path: "sh_vq_codebook.bin".into(),
            codebook_bytes: fs::metadata(&codebook_path)?.len(),
            codebook_sha256: sha256_file(&codebook_path)?,
            training_samples: config.training_samples,
            kmeans_iterations: config.iterations,
            rmse: (sh_squared_error / (selected_points * config.columns.len()) as f64).sqrt(),
            max_abs_error: sh_max_abs_error,
        }),
        None => None,
    };
    Ok(SidecarManifest {
        codec: if sh_vq_manifest.is_some() {
            "raw_binary_non_sh_non_xyz_records+deflate; exact_anchor_xor_geometry; sh_rest_vector_quantization".into()
        } else {
            "raw_binary_non_xyz_records+deflate; exact_anchor_xor_geometry".into()
        },
        path: "gaussians.deflate".into(),
        compressed_bytes: fs::metadata(sidecar_path)?.len(),
        compressed_sha256: sha256_file(sidecar_path)?,
        uncompressed_bytes: (selected_points * header.record_size) as u64,
        record_size: header.record_size,
        records: selected_points,
        properties: header.properties.clone(),
        sh_vq: sh_vq_manifest,
        sh_video: None,
        geometry_anchor_residuals: Some(GeometryAnchorResidualManifest {
            anchor_order: "lexicographic_quantized_xyz".into(),
            index_bytes: anchor_index_bytes,
            indices_path: "geometry_anchor_indices.deflate".into(),
            indices_compressed_bytes: fs::metadata(&anchor_indices_path)?.len(),
            indices_compressed_sha256: sha256_file(&anchor_indices_path)?,
            indices_uncompressed_bytes: (selected_points * anchor_index_bytes) as u64,
            xor_path: "geometry_anchor_xor.deflate".into(),
            xor_compressed_bytes: fs::metadata(&geometry_xor_path)?.len(),
            xor_compressed_sha256: sha256_file(&geometry_xor_path)?,
            xor_uncompressed_bytes: (selected_points * 12) as u64,
            residual_method: "raw_float32_bits XOR float32(anchor/scale_to_quantized + source_min)"
                .into(),
        }),
    })
}

fn convert_ply(
    source: &Path,
    destination: &Path,
    expected_header: &PlyHeader,
    mins: [f64; 3],
    maxes: [f64; 3],
    max_points: Option<usize>,
    geometry_bitdepth: u8,
    vpcc_geometry_bitdepth: Option<u8>,
    signed_log_geometry: bool,
    unique_anchors: bool,
    sidecar_path: Option<&Path>,
    sh_vq: Option<&ShVqConfig>,
    geometry_anchor_residuals: bool,
) -> Result<ConversionResult, DynError> {
    let (mut reader, header) = read_header(source)?;
    if header.vertex_count != expected_header.vertex_count
        || header.record_size != expected_header.record_size
    {
        return Err("source PLY changed between scan and conversion".into());
    }
    let columns = required_columns(&header)?;
    let selected_points = max_points
        .unwrap_or(header.vertex_count)
        .min(header.vertex_count);
    let quantized_max = (1u32 << geometry_bitdepth) - 1;
    let extents = [maxes[0] - mins[0], maxes[1] - mins[1], maxes[2] - mins[2]];
    let longest_extent = extents.into_iter().fold(0.0_f64, f64::max);
    if longest_extent <= f64::EPSILON {
        return Err("source geometry has zero extent".into());
    }
    let (coordinate_domain_min, coordinate_domain_max) = if signed_log_geometry {
        let domain_min = mins
            .into_iter()
            .map(signed_log1p)
            .fold(f64::INFINITY, f64::min);
        let domain_max = maxes
            .into_iter()
            .map(signed_log1p)
            .fold(f64::NEG_INFINITY, f64::max);
        (Some(domain_min), Some(domain_max))
    } else {
        (None, None)
    };
    let quantized_extent = match (coordinate_domain_min, coordinate_domain_max) {
        (Some(min), Some(max)) => max - min,
        _ => longest_extent,
    };
    if quantized_extent <= f64::EPSILON {
        return Err("coordinate quantization domain has zero extent".into());
    }
    let scale = quantized_max as f64 / quantized_extent;
    let transform = TransformManifest {
        method: if signed_log_geometry {
            "sog_signed_log1p_global_range".into()
        } else {
            "uniform_longest_axis_min_origin".into()
        },
        geometry_bitdepth,
        quantized_max,
        source_min: mins,
        source_max: maxes,
        source_longest_extent: longest_extent,
        scale_to_quantized: scale,
        inverse_scale_to_source: 1.0 / scale,
        inverse_formula: if signed_log_geometry {
            "source_xyz = signed_expm1(quantized_xyz / scale_to_quantized + coordinate_domain_min)"
                .into()
        } else {
            "source_xyz = quantized_xyz / scale_to_quantized + source_min".into()
        },
        signed_log_coordinates: signed_log_geometry,
        coordinate_domain_min,
        coordinate_domain_max,
    };

    let mut emitted = 0usize;
    let mut prepared = Vec::with_capacity(selected_points);
    let mut sidecar_encoder: Option<DeflateEncoder<File>> =
        match sidecar_path.filter(|_| !geometry_anchor_residuals) {
            Some(path) => Some(DeflateEncoder::new(
                File::create(path)?,
                Compression::default(),
            )),
            None => None,
        };
    let sh_vq_paths = match (sidecar_path, sh_vq, geometry_anchor_residuals) {
        (Some(sidecar_path), Some(_), false) => {
            let parent = sidecar_path
                .parent()
                .ok_or("sidecar path has no parent directory")?;
            Some((
                parent.join("sh_vq_indices.deflate"),
                parent.join("sh_vq_codebook.bin"),
            ))
        }
        _ => None,
    };
    if let (Some(config), Some((_, codebook_path))) = (sh_vq, &sh_vq_paths) {
        let mut writer = BufWriter::new(File::create(codebook_path)?);
        for value in &config.codebook {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
    }
    let mut sh_vq_encoder = match &sh_vq_paths {
        Some((indices_path, _)) => Some(DeflateEncoder::new(
            File::create(indices_path)?,
            Compression::default(),
        )),
        None => None,
    };
    let mut sh_vq_squared_error = 0.0_f64;
    let mut sh_vq_max_abs_error = 0.0_f64;
    visit_vertices(&mut reader, &header, |index, row| {
        if emitted >= selected_points {
            return Ok(());
        }
        let target_index = emitted.saturating_mul(header.vertex_count) / selected_points;
        if index != target_index {
            return Ok(());
        }

        let mut quantized = [0u32; 3];
        for axis in 0..3 {
            let value = property_value(&header, row, columns.xyz[axis])?;
            quantized[axis] = quantize_coordinate(value, axis, &transform);
        }
        let mut rgb = [0u8; 3];
        for (output, channel) in rgb.iter_mut().zip(columns.sh_dc) {
            let dc = property_value(&header, row, channel)? as f32;
            *output = ((0.5 + SH_C0 * dc).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        let vpcc_quantized = if let Some(vpcc_bits) = vpcc_geometry_bitdepth {
            if vpcc_bits >= geometry_bitdepth {
                return Err("V-PCC high-bit depth must be lower than position precision".into());
            }
            let low_bits = geometry_bitdepth - vpcc_bits;
            quantized.map(|value| value >> low_bits)
        } else {
            quantized
        };
        prepared.push(PreparedPoint {
            xyz: vpcc_quantized,
            rgb,
        });
        if let Some(encoder) = sidecar_encoder.as_mut() {
            if let Some(config) = sh_vq {
                let vector = sh_vector(&header, row, &config.columns);
                let (entry, _) =
                    nearest_codebook_entry(&vector, &config.codebook, config.columns.len());
                let centroid = &config.codebook
                    [entry * config.columns.len()..(entry + 1) * config.columns.len()];
                for (value, reconstructed) in vector.iter().zip(centroid) {
                    let error = *value as f64 - *reconstructed as f64;
                    sh_vq_squared_error += error * error;
                    sh_vq_max_abs_error = sh_vq_max_abs_error.max(error.abs());
                }
                let index_encoder = sh_vq_encoder
                    .as_mut()
                    .ok_or("SH VQ index encoder was not initialized")?;
                if config.codebook_size <= u8::MAX as usize + 1 {
                    index_encoder.write_all(&[entry as u8])?;
                } else {
                    index_encoder.write_all(&(entry as u16).to_le_bytes())?;
                }
                let mut template = row.to_vec();
                for column in &config.columns {
                    let property = &header.properties[*column];
                    template[property.offset..property.offset + 4].fill(0);
                }
                encoder.write_all(&template)?;
            } else {
                encoder.write_all(row)?;
            }
        }
        emitted += 1;
        Ok(())
    })?;

    if emitted != selected_points {
        return Err(format!("selected {emitted} points, expected {selected_points}").into());
    }

    let duplicate_positions = if unique_anchors {
        prepared.sort_unstable_by_key(|point| point.xyz);
        prepared.dedup_by_key(|point| point.xyz);
        selected_points - prepared.len()
    } else {
        let occupied: HashSet<_> = prepared.iter().map(|point| point.xyz).collect();
        selected_points - occupied.len()
    };
    let vpcc_points = prepared.len();

    let deferred_sidecar = if geometry_anchor_residuals {
        match sidecar_path {
            Some(path) => Some(write_anchor_residual_sidecar(
                source,
                &header,
                selected_points,
                &transform,
                &prepared,
                path,
                sh_vq,
            )?),
            None => None,
        }
    } else {
        None
    };

    if let Some(encoder) = sidecar_encoder {
        let mut file = encoder.finish()?;
        file.flush()?;
    }
    if let Some(encoder) = sh_vq_encoder {
        let mut file = encoder.finish()?;
        file.flush()?;
    }
    let sidecar = if let Some(sidecar) = deferred_sidecar {
        Some(sidecar)
    } else {
        match sidecar_path {
            Some(path) => {
                let sh_vq_manifest = match (sh_vq, sh_vq_paths) {
                    (Some(config), Some((indices_path, codebook_path))) => {
                        let index_bytes = if config.codebook_size <= u8::MAX as usize + 1 {
                            1
                        } else {
                            2
                        };
                        Some(ShVqManifest {
                            codebook_entries: config.codebook_size,
                            vector_dimensions: config.columns.len(),
                            index_bytes,
                            indices_path: "sh_vq_indices.deflate".into(),
                            indices_compressed_bytes: fs::metadata(&indices_path)?.len(),
                            indices_compressed_sha256: sha256_file(&indices_path)?,
                            indices_uncompressed_bytes: (selected_points * index_bytes) as u64,
                            codebook_path: "sh_vq_codebook.bin".into(),
                            codebook_bytes: fs::metadata(&codebook_path)?.len(),
                            codebook_sha256: sha256_file(&codebook_path)?,
                            training_samples: config.training_samples,
                            kmeans_iterations: config.iterations,
                            rmse: (sh_vq_squared_error
                                / (selected_points * config.columns.len()) as f64)
                                .sqrt(),
                            max_abs_error: sh_vq_max_abs_error,
                        })
                    }
                    _ => None,
                };
                Some(SidecarManifest {
                    codec: if sh_vq_manifest.is_some() {
                        "raw_binary_non_sh_records+deflate; sh_rest_vector_quantization".into()
                    } else {
                        "raw_binary_vertex_records+deflate".into()
                    },
                    path: "gaussians.deflate".into(),
                    compressed_bytes: fs::metadata(path)?.len(),
                    compressed_sha256: sha256_file(path)?,
                    uncompressed_bytes: (selected_points * header.record_size) as u64,
                    record_size: header.record_size,
                    records: selected_points,
                    properties: header.properties.clone(),
                    sh_vq: sh_vq_manifest,
                    sh_video: None,
                    geometry_anchor_residuals: None,
                })
            }
            None => None,
        }
    };

    let mut writer = BufWriter::new(File::create(destination)?);
    write!(
        writer,
        "ply\nformat binary_little_endian 1.0\ncomment generated by tmc2_adapter; point-cloud baseline only\nelement vertex {vpcc_points}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )?;
    for point in prepared {
        for coordinate in point.xyz {
            writer.write_all(&(coordinate as f32).to_le_bytes())?;
        }
        writer.write_all(&point.rgb)?;
    }
    writer.flush()?;

    Ok(ConversionResult {
        selected_points,
        vpcc_points,
        duplicate_positions,
        transform,
        sidecar,
    })
}

fn visit_vertices(
    reader: &mut BufReader<File>,
    header: &PlyHeader,
    mut visitor: impl FnMut(usize, &[u8]) -> Result<(), DynError>,
) -> Result<(), DynError> {
    match header.format {
        PlyFormat::BinaryLittleEndian => {
            let mut row = vec![0u8; header.record_size];
            for index in 0..header.vertex_count {
                reader.read_exact(&mut row)?;
                visitor(index, &row)?;
            }
        }
        PlyFormat::Ascii => {
            let mut encoded = Vec::new();
            for index in 0..header.vertex_count {
                let mut line = String::new();
                if reader.read_line(&mut line)? == 0 {
                    return Err("unexpected end of ASCII PLY vertex data".into());
                }
                let values = line
                    .split_whitespace()
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                if values.len() < header.properties.len() {
                    return Err(format!("ASCII PLY vertex {index} has too few fields").into());
                }
                encoded.clear();
                for value in values.into_iter().take(header.properties.len()) {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                visitor(index, &encoded)?;
            }
        }
    }
    Ok(())
}

fn property_value(header: &PlyHeader, row: &[u8], column: usize) -> Result<f64, DynError> {
    let property = header
        .properties
        .get(column)
        .ok_or("PLY property index is out of range")?;
    match header.format {
        PlyFormat::BinaryLittleEndian => {
            let end = property.offset + property.scalar_type.size();
            Ok(property.scalar_type.read(&row[property.offset..end]))
        }
        PlyFormat::Ascii => {
            let start = column * 8;
            Ok(f64::from_le_bytes(
                row[start..start + 8].try_into().unwrap(),
            ))
        }
    }
}

fn xyz_columns(properties: &[Property]) -> Result<[usize; 3], DynError> {
    let find = |name: &str| -> Result<usize, DynError> {
        properties
            .iter()
            .position(|property| property.name == name)
            .ok_or_else(|| format!("required PLY property missing: {name}").into())
    };
    Ok([find("x")?, find("y")?, find("z")?])
}

fn quantized_anchor(
    properties: &[Property],
    row: &[u8],
    transform: &TransformManifest,
) -> Result<[u32; 3], DynError> {
    let xyz = xyz_columns(properties)?;
    let mut anchor = [0u32; 3];
    for axis in 0..3 {
        let property = &properties[xyz[axis]];
        let end = property.offset + property.scalar_type.size();
        let value = property.scalar_type.read(&row[property.offset..end]);
        anchor[axis] = quantize_coordinate(value, axis, transform);
    }
    Ok(anchor)
}

fn decoded_anchor_set(path: &Path) -> Result<(HashSet<[u32; 3]>, usize), DynError> {
    let (mut reader, header) = read_header(path)?;
    let xyz = xyz_columns(&header.properties)?;
    let mut anchors = HashSet::with_capacity(header.vertex_count);
    visit_vertices(&mut reader, &header, |_, row| {
        let mut anchor = [0u32; 3];
        for axis in 0..3 {
            let value = property_value(&header, row, xyz[axis])?;
            if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 {
                return Err(format!("invalid decoded V-PCC coordinate: {value}").into());
            }
            anchor[axis] = value.round() as u32;
        }
        anchors.insert(anchor);
        Ok(())
    })?;
    Ok((anchors, header.vertex_count))
}

fn resolve_artifact(container_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        container_dir.join(path)
    }
}

fn sha256_file(path: &Path) -> Result<String, DynError> {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn sha256_anchor_set(anchors: &[[u32; 3]]) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    for anchor in anchors {
        for coordinate in anchor {
            context.update(&coordinate.to_le_bytes());
        }
    }
    context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn load_sh_vq_codebook(
    sidecar: &SidecarManifest,
    container_dir: &Path,
) -> Result<Option<(ShVqManifest, Vec<f32>, Vec<usize>)>, DynError> {
    let Some(vq) = sidecar.sh_vq.as_ref() else {
        return Ok(None);
    };
    let header = PlyHeader {
        format: PlyFormat::BinaryLittleEndian,
        vertex_count: sidecar.records,
        properties: sidecar.properties.clone(),
        record_size: sidecar.record_size,
    };
    let columns = sh_rest_columns(&header)?;
    if vq.vector_dimensions != columns.len() {
        return Err(format!(
            "SH VQ dimension {} does not match PLY SH-rest dimension {}",
            vq.vector_dimensions,
            columns.len()
        )
        .into());
    }
    if vq.index_bytes != 1 && vq.index_bytes != 2 {
        return Err("SH VQ index width must be one or two bytes".into());
    }
    let expected_codebook_bytes = vq.codebook_entries * vq.vector_dimensions * 4;
    if vq.codebook_bytes != expected_codebook_bytes as u64 {
        return Err("SH VQ codebook byte count does not match its declared shape".into());
    }
    let codebook_path = resolve_artifact(container_dir, &vq.codebook_path);
    if sha256_file(&codebook_path)? != vq.codebook_sha256 {
        return Err("SH VQ codebook SHA-256 mismatch".into());
    }
    let bytes = fs::read(&codebook_path)?;
    if bytes.len() != expected_codebook_bytes {
        return Err("SH VQ codebook has an unexpected file size".into());
    }
    let codebook = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    Ok(Some((vq.clone(), codebook, columns)))
}

fn load_sh_video(
    sidecar: &SidecarManifest,
    container_dir: &Path,
) -> Result<Option<(ShVideoManifest, Vec<Vec<u16>>, Vec<usize>)>, DynError> {
    let Some(video) = sidecar.sh_video.as_ref() else {
        return Ok(None);
    };
    let header = PlyHeader {
        format: PlyFormat::BinaryLittleEndian,
        vertex_count: sidecar.records,
        properties: sidecar.properties.clone(),
        record_size: sidecar.record_size,
    };
    let columns = sh_rest_columns(&header)?;
    if video.vector_dimensions != columns.len() || video.quantization.len() != columns.len() {
        return Err("SH video dimensions do not match the sidecar schema".into());
    }
    if video.active_len != sidecar.records || video.width == 0 || video.height == 0 {
        return Err("SH video atlas dimensions are invalid".into());
    }
    let hi = resolve_artifact(container_dir, &video.hi_path);
    let lo = resolve_artifact(container_dir, &video.lo_path);
    if sha256_file(&hi)? != video.hi_sha256 || sha256_file(&lo)? != video.lo_sha256 {
        return Err("SH video stream SHA-256 mismatch".into());
    }
    let streams = read_sh_rest_video_atlas_streams(
        container_dir,
        video.width,
        video.height,
        video.vector_dimensions,
    )?;
    if streams.iter().any(|stream| stream.len() < video.active_len) {
        return Err("decoded SH video stream is shorter than declared".into());
    }
    Ok(Some((video.clone(), streams, columns)))
}

fn read_compact_index(reader: &mut impl Read, bytes: usize) -> Result<usize, DynError> {
    if !(1..=4).contains(&bytes) {
        return Err("compact index width must be between one and four bytes".into());
    }
    let mut encoded = [0u8; 4];
    reader.read_exact(&mut encoded[..bytes])?;
    Ok(u32::from_le_bytes(encoded) as usize)
}

fn load_geometry_anchor_residuals(
    sidecar: &SidecarManifest,
    container_dir: &Path,
) -> Result<Option<GeometryAnchorResidualManifest>, DynError> {
    let Some(geometry) = sidecar.geometry_anchor_residuals.as_ref() else {
        return Ok(None);
    };
    if geometry.anchor_order != "lexicographic_quantized_xyz" {
        return Err("unsupported geometry anchor ordering".into());
    }
    if !(1..=4).contains(&geometry.index_bytes) {
        return Err("geometry anchor index width must be between one and four bytes".into());
    }
    if geometry.indices_uncompressed_bytes != (sidecar.records * geometry.index_bytes) as u64
        || geometry.xor_uncompressed_bytes != (sidecar.records * 12) as u64
    {
        return Err("geometry anchor residual stream sizes do not match record count".into());
    }
    let index_path = resolve_artifact(container_dir, &geometry.indices_path);
    let xor_path = resolve_artifact(container_dir, &geometry.xor_path);
    if sha256_file(&index_path)? != geometry.indices_compressed_sha256 {
        return Err("geometry anchor index stream SHA-256 mismatch".into());
    }
    if sha256_file(&xor_path)? != geometry.xor_compressed_sha256 {
        return Err("geometry anchor XOR stream SHA-256 mismatch".into());
    }
    Ok(Some(geometry.clone()))
}

fn restore_full_ply(
    manifest: &ConversionManifest,
    decoded_anchors: &Path,
    destination: &Path,
) -> Result<(), DynError> {
    if let Some(visual) = manifest.artifacts.visual_sidecar.as_ref() {
        return restore_visual_ply(manifest, visual, decoded_anchors, destination);
    }
    let sidecar = manifest
        .artifacts
        .sidecar
        .as_ref()
        .ok_or("container manifest has no full 3DGS sidecar")?;
    let container_dir = decoded_anchors
        .parent()
        .ok_or("decoded anchor path has no parent directory")?;
    let sidecar_path = resolve_artifact(container_dir, &sidecar.path);
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    if sidecar_sha256 != sidecar.compressed_sha256 {
        return Err(format!(
            "sidecar SHA-256 mismatch: expected {}, got {}",
            sidecar.compressed_sha256, sidecar_sha256
        )
        .into());
    }
    let sh_vq = load_sh_vq_codebook(sidecar, container_dir)?;
    let sh_video = load_sh_video(sidecar, container_dir)?;
    if sh_vq.is_some() && sh_video.is_some() {
        return Err("sidecar cannot contain both SH VQ and SH video".into());
    }
    let geometry_anchor_residuals = load_geometry_anchor_residuals(sidecar, container_dir)?;
    if let Some((vq, _, _)) = &sh_vq {
        let indices_path = resolve_artifact(container_dir, &vq.indices_path);
        if sha256_file(&indices_path)? != vq.indices_compressed_sha256 {
            return Err("SH VQ index stream SHA-256 mismatch".into());
        }
    }
    let (decoded, decoded_records) = decoded_anchor_set(decoded_anchors)?;
    if decoded_records != decoded.len() {
        return Err(format!(
            "decoded V-PCC output contains {} duplicate anchors",
            decoded_records - decoded.len()
        )
        .into());
    }
    if decoded.len() != manifest.vpcc_points {
        return Err(format!(
            "decoded V-PCC anchor count {} does not match manifest {}",
            decoded.len(),
            manifest.vpcc_points
        )
        .into());
    }
    let mut decoded_anchor_table = decoded.iter().copied().collect::<Vec<_>>();
    decoded_anchor_table.sort_unstable();
    if destination.exists() {
        return Err(format!(
            "refusing to overwrite output PLY: {}",
            destination.display()
        )
        .into());
    }
    if let Some(parent) = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let temporary = destination.with_extension("ply.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(
        writer,
        "comment restored by tmc2_adapter hybrid V-PCC container"
    )?;
    writeln!(writer, "element vertex {}", sidecar.records)?;
    for property in &sidecar.properties {
        writeln!(
            writer,
            "property {} {}",
            property.scalar_type.ply_name(),
            property.name
        )?;
    }
    writeln!(writer, "end_header")?;

    let mut decoder = DeflateDecoder::new(BufReader::new(File::open(&sidecar_path)?));
    let mut sh_vq_index_decoder = match &sh_vq {
        Some((vq, _, _)) => Some(DeflateDecoder::new(BufReader::new(File::open(
            resolve_artifact(container_dir, &vq.indices_path),
        )?))),
        None => None,
    };
    let mut geometry_anchor_index_decoder = match &geometry_anchor_residuals {
        Some(geometry) => Some(DeflateDecoder::new(BufReader::new(File::open(
            resolve_artifact(container_dir, &geometry.indices_path),
        )?))),
        None => None,
    };
    let mut geometry_xor_decoder = match &geometry_anchor_residuals {
        Some(geometry) => Some(DeflateDecoder::new(BufReader::new(File::open(
            resolve_artifact(container_dir, &geometry.xor_path),
        )?))),
        None => None,
    };
    let xyz = xyz_columns(&sidecar.properties)?;
    let mut row = vec![0u8; sidecar.record_size];
    let mut expected = HashSet::with_capacity(manifest.vpcc_points);
    for record_index in 0..sidecar.records {
        decoder.read_exact(&mut row)?;
        if let Some(geometry) = &geometry_anchor_residuals {
            let anchor_id = read_compact_index(
                geometry_anchor_index_decoder
                    .as_mut()
                    .ok_or("geometry anchor index decoder was not initialized")?,
                geometry.index_bytes,
            )?;
            let anchor = decoded_anchor_table
                .get(anchor_id)
                .ok_or("geometry anchor index is outside the decoded anchor table")?;
            for axis in 0..3 {
                let mut xor = [0u8; 4];
                geometry_xor_decoder
                    .as_mut()
                    .ok_or("geometry XOR decoder was not initialized")?
                    .read_exact(&mut xor)?;
                let value = u32::from_le_bytes(xor)
                    ^ anchor_base_coordinate(anchor[axis], axis, &manifest.transform).to_bits();
                let property = &sidecar.properties[xyz[axis]];
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        if let Some((vq, codebook, columns)) = &sh_vq {
            let mut index_bytes = [0u8; 2];
            sh_vq_index_decoder
                .as_mut()
                .ok_or("SH VQ index decoder was not initialized")?
                .read_exact(&mut index_bytes[..vq.index_bytes])?;
            let entry = if vq.index_bytes == 1 {
                index_bytes[0] as usize
            } else {
                u16::from_le_bytes(index_bytes) as usize
            };
            if entry >= vq.codebook_entries {
                return Err("SH VQ index is outside the codebook".into());
            }
            let vector =
                &codebook[entry * vq.vector_dimensions..(entry + 1) * vq.vector_dimensions];
            for (column, value) in columns.iter().zip(vector) {
                let property = &sidecar.properties[*column];
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        if let Some((video, streams, columns)) = &sh_video {
            for (stream, column) in columns.iter().enumerate() {
                let property = &sidecar.properties[*column];
                let value =
                    dequantize_sh(streams[stream][record_index], &video.quantization[stream]);
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        expected.insert(quantized_anchor(
            &sidecar.properties,
            &row,
            &manifest.transform,
        )?);
        writer.write_all(&row)?;
    }
    let mut trailing = [0u8; 1];
    if decoder.read(&mut trailing)? != 0 {
        return Err("sidecar contains more vertex data than its manifest declares".into());
    }
    if let Some(index_decoder) = sh_vq_index_decoder.as_mut() {
        if index_decoder.read(&mut trailing)? != 0 {
            return Err(
                "SH VQ index stream contains more indices than its manifest declares".into(),
            );
        }
    }
    if let Some(index_decoder) = geometry_anchor_index_decoder.as_mut() {
        if index_decoder.read(&mut trailing)? != 0 {
            return Err("geometry anchor index stream contains more entries than declared".into());
        }
    }
    if let Some(xor_decoder) = geometry_xor_decoder.as_mut() {
        if xor_decoder.read(&mut trailing)? != 0 {
            return Err("geometry anchor XOR stream contains more entries than declared".into());
        }
    }
    writer.flush()?;

    if expected != decoded {
        let missing = expected.difference(&decoded).count();
        let unexpected = decoded.difference(&expected).count();
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "V-PCC anchor validation failed: missing={missing}, unexpected={unexpected}"
        )
        .into());
    }
    fs::rename(temporary, destination)?;
    Ok(())
}

fn restore_visual_ply(
    manifest: &ConversionManifest,
    visual: &VisualSidecarManifest,
    decoded_anchors: &Path,
    destination: &Path,
) -> Result<(), DynError> {
    if visual.anchor_groups.is_some() {
        return restore_anchor_grouped_ply(manifest, visual, decoded_anchors, destination);
    }
    let container_dir = decoded_anchors
        .parent()
        .ok_or("decoded anchor path has no parent directory")?;
    let payload_path = resolve_artifact(container_dir, &visual.path);
    if sha256_file(&payload_path)? != visual.compressed_sha256 {
        return Err("visual sidecar SHA-256 mismatch".into());
    }
    let codebook_path = resolve_artifact(container_dir, &visual.sh_vq.codebook_path);
    let index_path = resolve_artifact(container_dir, &visual.sh_vq.indices_path);
    if sha256_file(&codebook_path)? != visual.sh_vq.codebook_sha256
        || sha256_file(&index_path)? != visual.sh_vq.indices_compressed_sha256
    {
        return Err("visual SH VQ artifact SHA-256 mismatch".into());
    }
    let codebook_bytes = fs::read(&codebook_path)?;
    let expected_codebook_bytes =
        visual.sh_vq.codebook_entries * visual.sh_vq.vector_dimensions * 4;
    if codebook_bytes.len() != expected_codebook_bytes {
        return Err("visual SH VQ codebook size does not match its manifest".into());
    }
    let codebook = codebook_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let (decoded, decoded_records) = decoded_anchor_set(decoded_anchors)?;
    if decoded_records != decoded.len() || decoded.len() != manifest.vpcc_points {
        return Err("decoded V-PCC anchors do not match the visual container contract".into());
    }
    let mut anchors = decoded.iter().copied().collect::<Vec<_>>();
    anchors.sort_unstable();
    if sha256_anchor_set(&anchors) != visual.anchor_set_sha256 {
        return Err("decoded V-PCC anchor set differs from the visual sidecar anchor set".into());
    }
    if destination.exists() {
        return Err(format!(
            "refusing to overwrite output PLY: {}",
            destination.display()
        )
        .into());
    }
    if let Some(parent) = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = destination.with_extension("ply.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(
        writer,
        "comment restored by tmc2_adapter visual hybrid V-PCC container"
    )?;
    writeln!(writer, "element vertex {}", visual.records)?;
    for property in &visual.properties {
        writeln!(
            writer,
            "property {} {}",
            property.scalar_type.ply_name(),
            property.name
        )?;
    }
    writeln!(writer, "end_header")?;
    let header = PlyHeader {
        format: PlyFormat::BinaryLittleEndian,
        vertex_count: visual.records,
        properties: visual.properties.clone(),
        record_size: visual.record_size,
    };
    let xyz = xyz_columns(&visual.properties)?;
    let sh_columns = sh_rest_columns(&header)?;
    if visual.semantic_columns.len() != visual.quantization.len() {
        return Err("visual semantic stream metadata is inconsistent".into());
    }
    let mut payload = DeflateDecoder::new(BufReader::new(File::open(&payload_path)?));
    let mut sh_indices = DeflateDecoder::new(BufReader::new(File::open(index_path)?));
    let mut expected = HashSet::with_capacity(manifest.vpcc_points);
    let mut row = vec![0u8; visual.record_size];
    for _ in 0..visual.records {
        row.fill(0);
        let anchor_id = read_compact_index(&mut payload, visual.anchor_index_bytes)?;
        let anchor = anchors
            .get(anchor_id)
            .ok_or("visual anchor index is outside decoded anchor table")?;
        for axis in 0..3 {
            let property = &visual.properties[xyz[axis]];
            let value = anchor_base_coordinate(anchor[axis], axis, &manifest.transform) as f32;
            row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (column, quantization) in visual.semantic_columns.iter().zip(&visual.quantization) {
            let mut bytes = [0u8; 2];
            payload.read_exact(&mut bytes)?;
            let value = dequantize_sh(u16::from_le_bytes(bytes), quantization);
            let property = &visual.properties[*column];
            row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let entry = read_compact_index(&mut sh_indices, visual.sh_vq.index_bytes)?;
        if entry >= visual.sh_vq.codebook_entries {
            return Err("visual SH VQ index is outside the codebook".into());
        }
        let vector = &codebook
            [entry * visual.sh_vq.vector_dimensions..(entry + 1) * visual.sh_vq.vector_dimensions];
        for (column, value) in sh_columns.iter().zip(vector) {
            let property = &visual.properties[*column];
            row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        expected.insert(*anchor);
        writer.write_all(&row)?;
    }
    let mut trailing = [0u8; 1];
    if payload.read(&mut trailing)? != 0 || sh_indices.read(&mut trailing)? != 0 {
        return Err("visual sidecar contains more records than its manifest declares".into());
    }
    writer.flush()?;
    if expected != decoded {
        let _ = fs::remove_file(&temporary);
        return Err(
            "visual sidecar did not reference the complete decoded V-PCC anchor set".into(),
        );
    }
    fs::rename(temporary, destination)?;
    Ok(())
}

fn read_varint(reader: &mut impl Read) -> Result<u32, DynError> {
    let mut value = 0u32;
    for shift in (0..35).step_by(7) {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        value |= ((byte[0] & 0x7f) as u32) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("anchor count varint is too long".into())
}

fn restore_anchor_grouped_ply(
    manifest: &ConversionManifest,
    visual: &VisualSidecarManifest,
    decoded_anchors: &Path,
    destination: &Path,
) -> Result<(), DynError> {
    let groups = visual
        .anchor_groups
        .as_ref()
        .ok_or("missing anchor group metadata")?;
    let dir = decoded_anchors
        .parent()
        .ok_or("decoded anchor path has no parent")?;
    let payload_path = if let Some(grid) = &visual.semantic_webp {
        decode_semantic_webp_grid(dir, grid)?
    } else {
        resolve_artifact(dir, &visual.path)
    };
    let counts_path = resolve_artifact(dir, &groups.counts_path);
    let residual_path = resolve_artifact(dir, &groups.residual_path);
    let index_path = resolve_artifact(dir, &visual.sh_vq.indices_path);
    let codebook_path = resolve_artifact(dir, &visual.sh_vq.codebook_path);
    if (visual.semantic_webp.is_none() && sha256_file(&payload_path)? != visual.compressed_sha256)
        || sha256_file(&counts_path)? != groups.counts_sha256
        || sha256_file(&residual_path)? != groups.residual_sha256
        || sha256_file(&index_path)? != visual.sh_vq.indices_compressed_sha256
        || sha256_file(&codebook_path)? != visual.sh_vq.codebook_sha256
    {
        return Err("anchor-grouped sidecar checksum mismatch".into());
    }
    let (decoded, count) = decoded_anchor_set(decoded_anchors)?;
    if count != decoded.len() || decoded.len() != manifest.vpcc_points {
        return Err("decoded V-PCC anchors do not match anchor-grouped container".into());
    }
    let mut anchors = decoded.into_iter().collect::<Vec<_>>();
    sort_anchors_morton(&mut anchors, manifest.transform.geometry_bitdepth);
    if sha256_anchor_set(&anchors) != visual.anchor_set_sha256 {
        return Err("decoded anchor set differs from anchor-grouped sidecar".into());
    }
    let codebook_bytes = fs::read(&codebook_path)?;
    let codebook = codebook_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    if codebook.len() != visual.sh_vq.codebook_entries * visual.sh_vq.vector_dimensions {
        return Err("anchor-grouped codebook shape mismatch".into());
    }
    if destination.exists() {
        return Err(format!(
            "refusing to overwrite output PLY: {}",
            destination.display()
        )
        .into());
    }
    if let Some(parent) = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = destination.with_extension("ply.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(
        writer,
        "comment restored by tmc2_adapter anchor-grouped hybrid container"
    )?;
    writeln!(writer, "element vertex {}", visual.records)?;
    for property in &visual.properties {
        writeln!(
            writer,
            "property {} {}",
            property.scalar_type.ply_name(),
            property.name
        )?;
    }
    writeln!(writer, "end_header")?;
    let header = PlyHeader {
        format: PlyFormat::BinaryLittleEndian,
        vertex_count: visual.records,
        properties: visual.properties.clone(),
        record_size: visual.record_size,
    };
    let xyz = xyz_columns(&visual.properties)?;
    let sh_columns = sh_rest_columns(&header)?;
    let mut payload = DeflateDecoder::new(BufReader::new(File::open(&payload_path)?));
    let mut counts = DeflateDecoder::new(BufReader::new(File::open(counts_path)?));
    let mut residuals = DeflateDecoder::new(BufReader::new(File::open(residual_path)?));
    let mut packed_residuals = if groups.residuals_bitpacked {
        let mut bytes = Vec::new();
        residuals.read_to_end(&mut bytes)?;
        Some(PackedResidualReader::new(bytes, groups.residual_bits))
    } else {
        None
    };
    let mut indices = DeflateDecoder::new(BufReader::new(File::open(index_path)?));
    let mut written = 0usize;
    let mut residual_records_read = 0usize;
    let mut row = vec![0u8; visual.record_size];
    for anchor in anchors {
        let group_count = read_varint(&mut counts)? as usize;
        let requires_residual = groups.residuals_for_singletons
            || (groups.residuals_for_collisions && group_count > 1);
        let mut previous_low_code = 0u32;
        for _ in 0..group_count {
            row.fill(0);
            let entropy_low = if requires_residual && groups.entropy_low_bits {
                previous_low_code = previous_low_code
                    .checked_add(read_varint(&mut residuals)?)
                    .ok_or("low-bit Morton delta overflow")?;
                Some(anchor_from_morton(previous_low_code, groups.residual_bits))
            } else {
                None
            };
            for axis in 0..3 {
                let value = if requires_residual {
                    let quantized = if let Some(low) = entropy_low {
                        low[axis]
                    } else if let Some(reader) = packed_residuals.as_mut() {
                        reader.read()?
                    } else {
                        let mut bytes = [0u8; 2];
                        residuals.read_exact(&mut bytes)?;
                        u16::from_le_bytes(bytes)
                    };
                    if axis == 0 {
                        residual_records_read += 1;
                    }
                    if groups.high_low_geometry {
                        let full = (anchor[axis] << groups.residual_bits) | u32::from(quantized);
                        dequantize_coordinate(full, axis, &manifest.transform) as f32
                    } else {
                        dequantize_local_residual(
                            quantized,
                            anchor[axis],
                            axis,
                            &manifest.transform,
                            groups.residual_bits,
                        )
                    }
                } else {
                    anchor_base_coordinate(anchor[axis], axis, &manifest.transform)
                };
                let property = &visual.properties[xyz[axis]];
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (column, quant) in visual.semantic_columns.iter().zip(&visual.quantization) {
                let mut bytes = [0u8; 2];
                payload.read_exact(&mut bytes)?;
                let value = dequantize_sh(u16::from_le_bytes(bytes), quant);
                let property = &visual.properties[*column];
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            let entry = read_compact_index(&mut indices, visual.sh_vq.index_bytes)?;
            if entry >= visual.sh_vq.codebook_entries {
                return Err("anchor-grouped SH VQ index is outside codebook".into());
            }
            let vector = &codebook[entry * visual.sh_vq.vector_dimensions
                ..(entry + 1) * visual.sh_vq.vector_dimensions];
            for (column, value) in sh_columns.iter().zip(vector) {
                let property = &visual.properties[*column];
                row[property.offset..property.offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            writer.write_all(&row)?;
            written += 1;
        }
    }
    if written != visual.records {
        return Err(format!(
            "anchor counts produced {written} records, expected {}",
            visual.records
        )
        .into());
    }
    if groups.residual_records != 0 && groups.residual_records != residual_records_read {
        return Err("anchor-grouped residual record count differs from decoded groups".into());
    }
    if let Some(reader) = packed_residuals.as_ref() {
        reader.validate_padding()?;
    }
    let mut trailing = [0u8; 1];
    if payload.read(&mut trailing)? != 0
        || counts.read(&mut trailing)? != 0
        || residuals.read(&mut trailing)? != 0
        || indices.read(&mut trailing)? != 0
    {
        return Err("anchor-grouped sidecar contains more records than its manifest declares".into());
    }
    writer.flush()?;
    fs::rename(temporary, destination)?;
    if visual.semantic_webp.is_some() {
        fs::remove_file(&payload_path)?;
    }
    Ok(())
}

fn decode_full_command(cli: &DecodeCli) -> Result<(), DynError> {
    let manifest_path = cli.container_dir.join("manifest.json");
    let manifest: ConversionManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    if !matches!(
        manifest.mode.as_str(),
        "mpeg_vpcc_3dgs_hybrid_v1"
            | "mpeg_vpcc_3dgs_hybrid_v2_visual"
            | "mpeg_vpcc_3dgs_hybrid_v3_anchor_grouped"
            | "mpeg_vpcc_3dgs_hybrid_v4_collision_residuals"
            | "mpeg_vpcc_3dgs_hybrid_v5_packed_geometry_residuals"
            | "mpeg_vpcc_3dgs_hybrid_v6_vpcc_only_geometry"
            | "mpeg_vpcc_3dgs_hybrid_v9_high_low_16bit_geometry"
            | "mpeg_vpcc_3dgs_hybrid_v10_high_low_entropy_geometry"
            | "mpeg_vpcc_3dgs_hybrid_v11_high_low_12bit_geometry"
            | "mpeg_vpcc_3dgs_hybrid_v12_semantic_webp_grid"
    ) {
        return Err(format!("not a full hybrid container: mode={}", manifest.mode).into());
    }
    let stream_value = manifest
        .artifacts
        .compressed_stream
        .as_deref()
        .ok_or("container manifest has no compressed stream")?;
    let stream = resolve_artifact(&cli.container_dir, stream_value);
    let decoded_pattern = cli.container_dir.join("decoded_for_restore_%04d.ply");
    let decoder_log = cli.container_dir.join("decode_full.log");
    run_decoder_paths(
        &cli.tmc2_root,
        true,
        cli.threads,
        &stream,
        &decoded_pattern,
        &decoder_log,
        None,
    )?;
    restore_full_ply(
        &manifest,
        &cli.container_dir.join("decoded_for_restore_0000.ply"),
        &cli.output_ply,
    )?;
    println!(
        "restored {} records to {} after exact V-PCC anchor validation",
        manifest.selected_points,
        cli.output_ply.display()
    );
    Ok(())
}

fn run_encoder(
    cli: &Cli,
    input: &Path,
    stream: &Path,
    reconstruction: &Path,
    log: &Path,
) -> Result<(), DynError> {
    let cfg = cli.tmc2_root.join("cfg");
    let input_pattern = input.with_file_name("frame_%04d.ply");
    let common = if cli.lossless_geometry {
        "common/ctc-common-lossless-geometry.cfg"
    } else {
        "common/ctc-common.cfg"
    };
    let condition = if cli.lossless_geometry {
        "condition/ctc-all-intra-lossless-geometry.cfg"
    } else {
        "condition/ctc-all-intra.cfg"
    };
    let rate_name = if cli.lossless_geometry {
        "r5-lossless"
    } else {
        &cli.rate
    };
    let rate = cfg.join(format!("rate/ctc-{rate_name}.cfg"));
    let mut args = vec![
        format!("--configurationFolder={}/", cfg.display()),
        format!("--config={}", cfg.join(common).display()),
        format!("--config={}", cfg.join(condition).display()),
        format!("--config={}", rate.display()),
        format!("--uncompressedDataPath={}", input_pattern.display()),
        "--startFrameNumber=0".into(),
        "--frameCount=1".into(),
        "--groupOfFramesSize=1".into(),
        // The pinned lossless HM profile uses QP=-12 and 10-bit image I/O.
        // Smaller integer anchor grids fit exactly in that container; keeping
        // the declared video/coordinate depth >=10 avoids implicit bit-depth
        // conversion and invalid HM QP ranges without changing anchor values.
        format!("--geometry3dCoordinatesBitdepth={}",
            if cli.lossless_geometry { cli.geometry_bitdepth.max(10) } else { cli.geometry_bitdepth }),
        format!(
            "--geometryNominal2dBitdepth={}",
            if cli.lossless_geometry {
                10
            } else {
                8
            }
        ),
        format!("--nbThread={}", cli.threads),
        "--keepIntermediateFiles=0".into(),
        format!("--reconstructedDataPath={}", reconstruction.display()),
        format!("--compressedStreamPath={}", stream.display()),
    ];
    if let Some(width) = cli.vpcc_atlas_width {
        args.push(format!("--minimumImageWidth={width}"));
    }
    if cli.vpcc_disable_patch_splitting {
        args.push("--enablePatchSplitting=0".into());
    }
    if cli.lossless_geometry {
        let hm_encoder = cli
            .tmc2_root
            .join("dependencies/HM-16.20+SCM-8.8-app/bin/TAppEncoderStatic");
        args.extend([
            "--apply3dMotionCompensation=0".into(),
            "--usePccRDO=0".into(),
            "--computeChecksum=0".into(),
            "--videoEncoderOccupancyCodecId=HMAPP".into(),
            "--videoEncoderGeometryCodecId=HMAPP".into(),
            "--videoEncoderAttributeCodecId=HMAPP".into(),
            format!("--videoEncoderOccupancyPath={}", hm_encoder.display()),
            format!("--videoEncoderGeometryPath={}", hm_encoder.display()),
            format!("--videoEncoderAttributePath={}", hm_encoder.display()),
        ]);
    } else {
        args.extend([
            "--videoEncoderOccupancyCodecId=HMLIB".into(),
            "--videoEncoderGeometryCodecId=HMLIB".into(),
            "--videoEncoderAttributeCodecId=HMLIB".into(),
        ]);
    }
    run_logged(&cli.tmc2_root.join("bin/PccAppEncoder"), &args, log)
}

fn run_decoder(
    cli: &Cli,
    stream: &Path,
    decoded: &Path,
    log: &Path,
    patch_map: Option<&Path>,
) -> Result<(), DynError> {
    run_decoder_paths(
        &cli.tmc2_root,
        cli.lossless_geometry,
        cli.threads,
        stream,
        decoded,
        log,
        patch_map,
    )
}

fn run_decoder_paths(
    tmc2_root: &Path,
    lossless_geometry: bool,
    threads: usize,
    stream: &Path,
    decoded: &Path,
    log: &Path,
    patch_map: Option<&Path>,
) -> Result<(), DynError> {
    let inverse_color_config = tmc2_root.join("cfg/hdrconvert/yuv420toyuv444_16bit.cfg");
    let mut args = vec![
        format!("--compressedStreamPath={}", stream.display()),
        format!(
            "--inverseColorSpaceConversionConfig={}",
            inverse_color_config.display()
        ),
        "--startFrameNumber=0".into(),
        "--frameCount=1".into(),
        format!("--nbThread={threads}"),
        format!("--reconstructedDataPath={}", decoded.display()),
    ];
    if lossless_geometry {
        let hm_decoder = tmc2_root.join("dependencies/HM-16.20+SCM-8.8-app/bin/TAppDecoderStatic");
        args.extend([
            "--computeChecksum=0".into(),
            format!("--videoDecoderOccupancyPath={}", hm_decoder.display()),
            format!("--videoDecoderGeometryPath={}", hm_decoder.display()),
            format!("--videoDecoderAttributePath={}", hm_decoder.display()),
        ]);
    }
    run_logged_with_env(
        &tmc2_root.join("bin/PccAppDecoder"),
        &args,
        log,
        patch_map.map(|path| ("PCC_EXPORT_PATCH_MAP", path.as_os_str())),
    )
}

fn run_3dgscompare_benchmark(
    cli: &Cli,
    restored_ply: &Path,
    output: &Path,
    log: &Path,
) -> Result<(), DynError> {
    let python = cli
        .benchmark_python
        .as_deref()
        .unwrap_or_else(|| Path::new(DEFAULT_3DGSCOMPARE_PYTHON));
    let backend = cli
        .benchmark_backend
        .as_deref()
        .unwrap_or_else(|| Path::new(DEFAULT_3DGSCOMPARE_BACKEND));
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot resolve repository root for 3dgsCompare benchmark script")?
        .join("evaluation/benchmark_3dgscompare.py");
    let run_name = cli
        .output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("3DGS codec run");
    let args = vec![
        windows_interop_path(&script),
        "--candidate".into(),
        windows_interop_path(restored_ply),
        "--output".into(),
        windows_interop_path(output),
        "--backend".into(),
        windows_interop_path(backend),
        "--name".into(),
        format!("V-PCC 3DGS: {run_name}"),
        "--original-scene-id".into(),
        cli.benchmark_original_scene_id.clone(),
        "--camera-bundle-id".into(),
        cli.benchmark_camera_bundle_id.clone(),
    ];
    run_logged(python, &args, log)?;
    if !output.is_file() {
        return Err(format!(
            "3dgsCompare completed without writing benchmark output: {}",
            output.display()
        )
        .into());
    }
    Ok(())
}

fn windows_interop_path(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let bytes = raw.as_bytes();
    if bytes.len() > 7
        && bytes.starts_with(b"/mnt/")
        && bytes[5].is_ascii_alphabetic()
        && bytes[6] == b'/'
    {
        format!(
            "{}:\\{}",
            (bytes[5] as char).to_ascii_uppercase(),
            raw[7..].replace('/', "\\")
        )
    } else {
        raw.into_owned()
    }
}

fn run_logged(executable: &Path, args: &[String], log_path: &Path) -> Result<(), DynError> {
    run_logged_with_env(executable, args, log_path, None)
}

fn run_logged_with_env(
    executable: &Path,
    args: &[String],
    log_path: &Path,
    environment: Option<(&str, &std::ffi::OsStr)>,
) -> Result<(), DynError> {
    let mut log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(log_path)?;
    writeln!(log, "executable={}", executable.display())?;
    for arg in args {
        writeln!(log, "argument={arg}")?;
    }
    log.flush()?;
    let stdout = log.try_clone()?;
    let mut command = Command::new(executable);
    command
        .args(args)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log));
    if let Some((key, value)) = environment {
        command.env(key, value);
    }
    let status = command.status()?;
    if !status.success() {
        return Err(format!(
            "{} failed with status {}; see {}",
            executable.display(),
            status,
            log_path.display()
        )
        .into());
    }
    Ok(())
}

fn git_commit(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_manifest(output_dir: &Path, manifest: &ConversionManifest) -> Result<(), DynError> {
    let path = output_dir.join("manifest.json");
    let temporary = output_dir.join("manifest.json.tmp");
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), manifest)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_tiled_manifest(
    output_dir: &Path,
    manifest: &TiledContainerManifest,
) -> Result<(), DynError> {
    let path = output_dir.join("manifest.json");
    let temporary = output_dir.join("manifest.json.tmp");
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), manifest)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn finalize_manifest_size(
    output_dir: &Path,
    manifest: &mut ConversionManifest,
) -> Result<(), DynError> {
    for _ in 0..8 {
        write_manifest(output_dir, manifest)?;
        let manifest_bytes = fs::metadata(output_dir.join("manifest.json"))?.len();
        let total = manifest.artifacts.container_payload_bytes.unwrap_or(0) + manifest_bytes;
        if manifest.artifacts.manifest_bytes == Some(manifest_bytes)
            && manifest.artifacts.container_total_bytes == Some(total)
        {
            return Ok(());
        }
        manifest.artifacts.manifest_bytes = Some(manifest_bytes);
        manifest.artifacts.container_total_bytes = Some(total);
    }
    Err("manifest size accounting did not converge".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_low_split_accepts_six_through_twelve_bit_anchors() {
        for bits in [6u8, 7, 8, 9, 10, 11, 12, 15] {
            let bits_value = bits.to_string();
            let args = ["scene.ply", "output", "--high-low-geometry", "--signed-log-geometry",
                        "--geometry-bitdepth", &bits_value];
            let cli = parse_cli_args(args.into_iter().map(OsString::from)).unwrap();
            assert_eq!(cli.geometry_bitdepth, bits);
            assert_eq!(cli.geometry_residual_bits, 16 - bits);
            assert!(cli.lossless_geometry);
            assert!(cli.packed_geometry_residuals);
            if bits <= 12 {
                let path = env::temp_dir().join(format!("tmc2-split-validation-{}-{bits}.ply", std::process::id()));
                let mut file = OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
                file.write_all(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n").unwrap();
                drop(file);
                let mut cli = cli;
                cli.source = path.clone();
                cli.prepare_only = true;
                let result = validate_cli(&cli);
                fs::remove_file(path).unwrap();
                result.unwrap();
            }
        }
    }

    #[test]
    fn dc_zero_maps_to_mid_gray() {
        let mapped = ((0.5 + SH_C0 * 0.0).clamp(0.0, 1.0) * 255.0).round() as u8;
        assert_eq!(mapped, 128);
    }

    #[test]
    fn scalar_types_have_expected_sizes() {
        assert_eq!(ScalarType::U8.size(), 1);
        assert_eq!(ScalarType::F32.size(), 4);
        assert_eq!(ScalarType::F64.size(), 8);
    }

    #[test]
    fn vector_quantizer_selects_the_nearest_codeword() {
        let codebook = [0.0, 0.0, 1.0, 1.0, 4.0, 4.0];
        let (index, squared_distance) = nearest_codebook_entry(&[0.8, 1.1], &codebook, 2);
        assert_eq!(index, 1);
        assert!((squared_distance - 0.05).abs() < 1e-9);
    }

    #[test]
    fn sog_signed_log_coordinates_use_a_shared_range_and_round_trip() {
        // These are the V16 bicycle SOG-web endpoints, derived from the
        // source extrema with sign(x) * log1p(abs(x)).
        let domain_min = -4.416_191_577_911_377;
        let domain_max = 4.584_405_899_047_852;
        let quantized_max = 65_535;
        let transform = TransformManifest {
            method: "sog_signed_log1p_global_range".into(),
            geometry_bitdepth: 16,
            quantized_max,
            source_min: [-65.395_240_783_691_4, -81.780_418_395_996_1, -71.705_932_617_187_5],
            source_max: [54.879_310_607_910_156, 14.772_521_018_981_934, 96.944_984_436_035_16],
            source_longest_extent: 168.650_917_053_222_66,
            scale_to_quantized: quantized_max as f64 / (domain_max - domain_min),
            inverse_scale_to_source: (domain_max - domain_min) / quantized_max as f64,
            inverse_formula: "test".into(),
            signed_log_coordinates: true,
            coordinate_domain_min: Some(domain_min),
            coordinate_domain_max: Some(domain_max),
        };
        assert_eq!(quantize_coordinate(-81.780_418_395_996_1, 1, &transform), 0);
        assert_eq!(quantize_coordinate(96.944_984_436_035_16, 2, &transform), quantized_max);
        for (axis, value) in [(0, -65.395_240_783_691_4), (1, -1.9353), (2, 96.944_984_436_035_16)] {
            let restored = dequantize_coordinate(quantize_coordinate(value, axis, &transform), axis, &transform);
            assert!((restored - value).abs() < 0.01, "axis={axis} value={value} restored={restored}");
        }
    }
}
