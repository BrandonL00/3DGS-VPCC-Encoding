use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use gaussian_packing::QualityPreset;

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub mode: RunMode,
    pub input_path: String,
    pub output_dir: String,
    pub overwrite: bool,
    pub write_decoded_ply: bool,
    pub decoded_ply_path: Option<PathBuf>,
    pub width: u32,
    pub compare_gsz: Option<String>,
    pub log_file: Option<PathBuf>,
    pub log_level: LogLevel,
    pub quality: QualityPreset,
    pub validation: ValidationMode,
    pub max_gaussians: Option<usize>,
    pub sample_rate: Option<f32>,
    pub cache_mode: CacheMode,
    pub sh_rest_sweep: ShRestSweepOverrides,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Dev,
    Bench,
}

impl RunMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Bench => "bench",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    None,
    Light,
    Full,
}

impl ValidationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Light => "light",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    None,
    Scene,
    Packed,
}

impl CacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scene => "scene",
            Self::Packed => "packed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Normal,
    Verbose,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Verbose => "verbose",
        }
    }

    pub fn is_verbose(self) -> bool {
        matches!(self, Self::Verbose)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ShRestSweepOverrides {
    pub enabled: Option<bool>,
    pub uniform_crfs: Option<Vec<u8>>,
    pub uniform_presets: Option<Vec<&'static str>>,
    pub tier_bounds: Option<[usize; 2]>,
    pub tier_crfs: Option<[u8; 3]>,
    pub tier_presets: Option<[&'static str; 3]>,
    pub prune_thresholds: Option<Vec<f32>>,
    pub prune_start_band: Option<usize>,
}

pub fn parse_args() -> Result<CliConfig, Box<dyn std::error::Error>> {
    let mut positionals = Vec::new();
    let mut output_dir = None;
    let mut overwrite = false;
    let mut write_decoded_ply = false;
    let mut decoded_ply_path = None;
    let mut timestamp_output = false;
    let mut log_file = None;
    let mut log_level = LogLevel::Normal;
    let mut quality = QualityPreset::Lossless;
    let mut mode = RunMode::Dev;
    let mut validation = None;
    let mut max_gaussians = None;
    let mut sample_rate = None;
    let mut cache_mode = None;
    let mut sh_rest_sweep = ShRestSweepOverrides::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--output-dir" {
            output_dir = Some(args.next().ok_or("Missing path after --output-dir")?);
        } else if arg == "--overwrite" {
            overwrite = true;
        } else if arg == "--write-decoded-ply" {
            write_decoded_ply = true;
        } else if arg == "--decoded-ply" {
            write_decoded_ply = true;
            decoded_ply_path = Some(PathBuf::from(
                args.next().ok_or("Missing path after --decoded-ply")?,
            ));
        } else if arg == "--timestamp-output" {
            timestamp_output = true;
        } else if arg == "--log-file" {
            log_file = Some(PathBuf::from(
                args.next().ok_or("Missing path after --log-file")?,
            ));
        } else if arg == "--debug" {
            log_level = LogLevel::Verbose;
        } else if let Some(value) = arg.strip_prefix("--log-level=") {
            log_level = parse_log_level(value)?;
        } else if arg == "--log-level" {
            log_level = parse_log_level(&args.next().ok_or("Missing value after --log-level")?)?;
        } else if arg == "--quality" {
            quality = parse_quality_preset(&args.next().ok_or("Missing value after --quality")?)?;
        } else if let Some(value) = arg.strip_prefix("--validate=") {
            validation = Some(parse_validation_mode(value)?);
        } else if arg == "--validate" {
            validation = Some(parse_validation_mode(
                &args.next().ok_or("Missing value after --validate")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--mode=") {
            mode = parse_run_mode(value)?;
        } else if arg == "--mode" {
            mode = parse_run_mode(&args.next().ok_or("Missing value after --mode")?)?;
        } else if let Some(value) = arg.strip_prefix("--cache=") {
            cache_mode = Some(parse_cache_mode(value)?);
        } else if arg == "--cache" {
            cache_mode = Some(parse_cache_mode(
                &args.next().ok_or("Missing value after --cache")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--max-gaussians=") {
            max_gaussians = Some(value.parse::<usize>()?);
        } else if arg == "--max-gaussians" {
            max_gaussians = Some(
                args.next()
                    .ok_or("Missing value after --max-gaussians")?
                    .parse::<usize>()?,
            );
        } else if let Some(value) = arg.strip_prefix("--sample-rate=") {
            sample_rate = Some(parse_sample_rate(value)?);
        } else if arg == "--sample-rate" {
            sample_rate = Some(parse_sample_rate(
                &args.next().ok_or("Missing value after --sample-rate")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-sweep=") {
            sh_rest_sweep.enabled = Some(parse_on_off(value)?);
        } else if arg == "--sh-rest-sweep" {
            sh_rest_sweep.enabled = Some(parse_on_off(
                &args.next().ok_or("Missing value after --sh-rest-sweep")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-sweep-crfs=") {
            sh_rest_sweep.uniform_crfs = Some(parse_csv_u8(value)?);
        } else if arg == "--sh-rest-sweep-crfs" {
            sh_rest_sweep.uniform_crfs = Some(parse_csv_u8(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-sweep-crfs")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-sweep-presets=") {
            sh_rest_sweep.uniform_presets = Some(parse_csv_presets(value)?);
        } else if arg == "--sh-rest-sweep-presets" {
            sh_rest_sweep.uniform_presets = Some(parse_csv_presets(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-sweep-presets")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-tier-bounds=") {
            sh_rest_sweep.tier_bounds = Some(parse_pair_usize(value)?);
        } else if arg == "--sh-rest-tier-bounds" {
            sh_rest_sweep.tier_bounds = Some(parse_pair_usize(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-tier-bounds")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-tier-crfs=") {
            sh_rest_sweep.tier_crfs = Some(parse_triple_u8(value)?);
        } else if arg == "--sh-rest-tier-crfs" {
            sh_rest_sweep.tier_crfs = Some(parse_triple_u8(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-tier-crfs")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-tier-presets=") {
            sh_rest_sweep.tier_presets = Some(parse_triple_presets(value)?);
        } else if arg == "--sh-rest-tier-presets" {
            sh_rest_sweep.tier_presets = Some(parse_triple_presets(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-tier-presets")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-prune-thresholds=") {
            sh_rest_sweep.prune_thresholds = Some(parse_csv_f32(value)?);
        } else if arg == "--sh-rest-prune-thresholds" {
            sh_rest_sweep.prune_thresholds = Some(parse_csv_f32(
                &args
                    .next()
                    .ok_or("Missing value after --sh-rest-prune-thresholds")?,
            )?);
        } else if let Some(value) = arg.strip_prefix("--sh-rest-prune-start-band=") {
            sh_rest_sweep.prune_start_band = Some(value.parse::<usize>()?);
        } else if arg == "--sh-rest-prune-start-band" {
            sh_rest_sweep.prune_start_band = Some(
                args.next()
                    .ok_or("Missing value after --sh-rest-prune-start-band")?
                    .parse::<usize>()?,
            );
        } else {
            positionals.push(arg);
        }
    }

    let input_path = positionals
        .first()
        .cloned()
        .ok_or("Missing input .ply path")?;
    let (positional_output_dir, width_idx) = match positionals.get(1) {
        Some(value) if value.parse::<u32>().is_err() => (Some(value.clone()), 2usize),
        _ => (None, 1usize),
    };
    let width = positionals
        .get(width_idx)
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(1024);
    let compare_gsz = positionals.get(width_idx + 1).cloned();
    let output_dir = resolve_output_dir(
        &input_path,
        output_dir.or(positional_output_dir),
        timestamp_output,
    );
    let validation = validation.unwrap_or(match mode {
        RunMode::Dev => ValidationMode::Light,
        RunMode::Bench => ValidationMode::Full,
    });
    let cache_mode = cache_mode.unwrap_or(match mode {
        RunMode::Dev => CacheMode::Scene,
        RunMode::Bench => CacheMode::None,
    });

    Ok(CliConfig {
        mode,
        input_path,
        output_dir,
        overwrite,
        write_decoded_ply,
        decoded_ply_path,
        width,
        compare_gsz,
        log_file,
        log_level,
        quality,
        validation,
        max_gaussians,
        sample_rate,
        cache_mode,
        sh_rest_sweep,
    })
}

fn resolve_output_dir(
    input_path: &str,
    output_dir: Option<String>,
    timestamp_output: bool,
) -> String {
    let base = output_dir.unwrap_or_else(|| {
        let stem = Path::new(input_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene");
        format!("runs/{stem}")
    });
    if timestamp_output {
        format!("{base}/{}", output_timestamp())
    } else {
        base
    }
}

fn output_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    secs.to_string()
}

fn parse_run_mode(value: &str) -> Result<RunMode, Box<dyn std::error::Error>> {
    match value {
        "dev" => Ok(RunMode::Dev),
        "bench" | "benchmark" => Ok(RunMode::Bench),
        _ => Err(format!("Unsupported mode: {value}. Expected dev or bench").into()),
    }
}

fn parse_validation_mode(value: &str) -> Result<ValidationMode, Box<dyn std::error::Error>> {
    match value {
        "none" => Ok(ValidationMode::None),
        "light" => Ok(ValidationMode::Light),
        "full" => Ok(ValidationMode::Full),
        _ => {
            Err(format!("Unsupported validate mode: {value}. Expected none, light, or full").into())
        }
    }
}

fn parse_cache_mode(value: &str) -> Result<CacheMode, Box<dyn std::error::Error>> {
    match value {
        "none" => Ok(CacheMode::None),
        "scene" => Ok(CacheMode::Scene),
        "packed" => Ok(CacheMode::Packed),
        _ => {
            Err(format!("Unsupported cache mode: {value}. Expected none, scene, or packed").into())
        }
    }
}

fn parse_log_level(value: &str) -> Result<LogLevel, Box<dyn std::error::Error>> {
    match value {
        "normal" => Ok(LogLevel::Normal),
        "verbose" => Ok(LogLevel::Verbose),
        _ => Err(format!("Unsupported log level: {value}. Expected normal or verbose").into()),
    }
}

fn parse_sample_rate(value: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let sample_rate = value.parse::<f32>()?;
    if !(0.0 < sample_rate && sample_rate <= 1.0) {
        return Err(format!("Invalid sample rate: {sample_rate}. Expected 0 < rate <= 1").into());
    }
    Ok(sample_rate)
}

fn parse_on_off(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        _ => Err(format!("Unsupported boolean flag value: {value}. Expected on or off").into()),
    }
}

fn parse_csv_u8(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<u8>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("expected at least one u8 value".into());
    }
    Ok(values)
}

fn parse_csv_f32(value: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<f32>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("expected at least one f32 value".into());
    }
    Ok(values)
}

fn parse_csv_presets(value: &str) -> Result<Vec<&'static str>, Box<dyn std::error::Error>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            parse_ffmpeg_preset(value).ok_or_else(|| format!("Unsupported ffmpeg preset: {value}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("expected at least one ffmpeg preset".into());
    }
    Ok(values)
}

fn parse_pair_usize(value: &str) -> Result<[usize; 2], Box<dyn std::error::Error>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|_: Vec<_>| "expected exactly two usize values".into())
}

fn parse_triple_u8(value: &str) -> Result<[u8; 3], Box<dyn std::error::Error>> {
    let values = parse_csv_u8(value)?;
    values
        .try_into()
        .map_err(|_: Vec<_>| "expected exactly three u8 values".into())
}

fn parse_triple_presets(value: &str) -> Result<[&'static str; 3], Box<dyn std::error::Error>> {
    let values = parse_csv_presets(value)?;
    values
        .try_into()
        .map_err(|_: Vec<_>| "expected exactly three ffmpeg presets".into())
}

fn parse_quality_preset(value: &str) -> Result<QualityPreset, Box<dyn std::error::Error>> {
    match value {
        "lossless" => Ok(QualityPreset::Lossless),
        "very_good" => Ok(QualityPreset::VeryGood),
        "ok" => Ok(QualityPreset::Ok),
        _ => Err(format!(
            "Unsupported quality preset: {value}. Expected one of: lossless, very_good, ok"
        )
        .into()),
    }
}

pub fn parse_ffmpeg_preset(value: &str) -> Option<&'static str> {
    match value {
        "ultrafast" => Some("ultrafast"),
        "superfast" => Some("superfast"),
        "veryfast" => Some("veryfast"),
        "faster" => Some("faster"),
        "fast" => Some("fast"),
        "medium" => Some("medium"),
        "slow" => Some("slow"),
        "slower" => Some("slower"),
        "veryslow" => Some("veryslow"),
        _ => None,
    }
}
