use std::fs::{self};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use discord_bot::{CompletionDetails, DiscordNotifier, RunContext};
use gaussian_packing::{
    GroupStorageMode, PackRasterTimings, PackedContainerByteSummary, PackedRasterStorageConfig,
    ShRestPackTimings, write_byte_split_planes_dir, write_debug_preview_pngs,
    write_packed_raster_scene_dir_with_report, write_scene_to_ply,
};
use gaussian_parser::load_gaussians_from_ply;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::{CacheMode, CliConfig, ValidationMode};
use crate::debug::{
    VerboseReportInput, run_sh_rest_sweep_report, should_run_sh_rest_sweep, source_normals_summary,
    verbose_run_log_lines,
};
use crate::logging::{LogMode, RunLogger};
use crate::metrics::MetricsReport;
use crate::report::{
    concise_summary_lines, coverage_from_packed, coverage_from_scene, final_coded_summary_lines,
    stream_coverage_lines, timing_summary_lines, verbose_summary_lines,
};

#[derive(Debug, Clone)]
struct ValidationArtifacts {
    decoded_packed: Option<gaussian_packing::PackedRasterScene>,
    reconstructed: Option<gaussian_parser::Scene>,
    diagnostics: Option<gaussian_packing::RoundtripDiagnostics>,
    metrics_report: Option<MetricsReport>,
}

#[derive(Debug, Clone)]
struct QuickValidationSummary {
    notes: Vec<String>,
}

pub fn run(config: CliConfig) -> Result<(), Box<dyn std::error::Error>> {
    let notifier = DiscordNotifier::from_env(RunContext::new(
        "encode",
        &config.input_path,
        &config.output_dir,
    ));
    notifier.run_started();
    let total_start = Instant::now();
    let mut stage = "preparing output";
    let result = run_inner(config, &notifier, total_start, &mut stage);
    if let Err(error) = result.as_ref() {
        notifier.run_failed(stage, &error.to_string(), total_start.elapsed());
    }
    result
}

fn run_inner(
    config: CliConfig,
    notifier: &DiscordNotifier,
    total_start: Instant,
    stage: &mut &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    notifier.stage_changed(*stage);
    prepare_output_dir(&config)?;
    let run_sh_rest_sweep = should_run_sh_rest_sweep(&config);
    let export_byte_split = std::env::var("VPCC_EXPORT_BYTE_SPLIT").ok().as_deref() == Some("1");
    let write_previews = std::env::var("VPCC_WRITE_PREVIEWS").ok().as_deref() == Some("1");
    let mut storage_config = PackedRasterStorageConfig::vpcc_video_atlas(config.quality);
    storage_config.sh_dc = sh_dc_storage_from_env(storage_config.sh_dc)?;
    let sh_rest_codec_config = storage_config.sh_rest_codec;
    let mut logger = RunLogger::new(
        Path::new(&config.output_dir),
        config.log_file.as_deref(),
        LogMode::from_env(),
    )?;

    let pb = progress_bar(5)?;
    *stage = "loading scene";
    notifier.stage_changed(*stage);
    pb.set_message("loading scene");
    let load_start = Instant::now();
    let scene = load_scene_with_cache(&config)?;
    let scene = apply_dev_subset(scene, &config);
    let load_scene = load_start.elapsed();
    pb.inc(1);

    *stage = "sorting gaussians";
    notifier.stage_changed(*stage);
    pb.set_message("sorting gaussians");
    let sort_start = Instant::now();
    let sorted = gaussian_packing::sort_gaussians_by_morton(&scene);
    let sort_gaussians = sort_start.elapsed();
    let source_normals = source_normals_summary(&sorted);
    if std::env::var("VPCC_KEEP_NORMALS").ok().as_deref() == Some("0") {
        storage_config.normals.keep_if_present = false;
    } else if source_normals.any_present && source_normals.valid_unitish_count == 0 {
        storage_config.normals.keep_if_present = false;
    }
    pb.inc(1);

    *stage = "packing raster";
    notifier.stage_changed(*stage);
    pb.set_message("packing raster");
    let packed_cache_path = packed_cache_path(&config);
    let (packed, pack_timings) = match config.cache_mode {
        CacheMode::Packed => {
            load_or_pack_cached(&sorted, config.width, packed_cache_path.as_ref())?
        }
        _ => gaussian_packing::pack_sorted_gaussians_to_raster_with_timing(&sorted, config.width),
    };
    pb.inc(1);

    *stage = "validating packed scene";
    notifier.stage_changed(*stage);
    pb.set_message(match config.validation {
        ValidationMode::Full => "validating",
        ValidationMode::Light => "checking",
        ValidationMode::None => "skipping validation",
    });
    let prewrite_validation = validate_prewrite(&sorted, &packed, config.validation)?;
    pb.inc(1);

    *stage = "writing encoded container";
    notifier.stage_changed(*stage);
    pb.set_message("writing output");
    let write_start = Instant::now();
    let write_report =
        write_packed_raster_scene_dir_with_report(&packed, &config.output_dir, storage_config)?;
    if export_byte_split {
        write_byte_split_planes_dir(&packed, &config.output_dir)?;
    }
    if write_previews {
        write_debug_preview_pngs(&packed, &config.output_dir)?;
    }
    let write_container = write_start.elapsed();
    pb.inc(1);
    pb.finish_and_clear();

    let coded_summary = gaussian_packing::measure_packed_raster_scene_dir(&config.output_dir)?;
    let decoded_ply = if config.write_decoded_ply {
        *stage = "writing decoded PLY";
        notifier.stage_changed(*stage);
        Some(write_decoded_ply(&config, None)?)
    } else {
        None
    };
    *stage = "post-write validation";
    notifier.stage_changed(*stage);
    let validation_start = Instant::now();
    let (quick_validation, validation_artifacts) =
        validate_postwrite(&config, &sorted, &packed, &coded_summary)?;
    let unpack_roundtrip_validation = validation_start.elapsed();
    let raw_payload_bytes = gaussian_packing::packed_raster_raw_payload_bytes(&packed);
    let sh_rest_stats = gaussian_packing::sh_rest_payload_stats(&packed);
    let original_ply_size = fs::metadata(&config.input_path)?.len();
    let sh_rest_sweep_report = if run_sh_rest_sweep {
        Some(run_sh_rest_sweep_report(
            &config,
            &sorted,
            &packed,
            &coded_summary,
            raw_payload_bytes,
        )?)
    } else {
        None
    };

    let summary_lines = concise_summary_lines(
        &config,
        &config.output_dir,
        &coded_summary,
        raw_payload_bytes,
        total_start.elapsed(),
        decoded_ply
            .as_ref()
            .map(|(path, size)| (path.as_path(), *size)),
        sh_rest_sweep_report
            .as_ref()
            .and_then(|report| report.best.as_ref()),
    );
    logger.summary_lines(summary_lines)?;

    let should_write_detail_log = config.log_level.is_verbose() || run_sh_rest_sweep;
    if should_write_detail_log {
        let comparison_lines = comparison_lines(&config, raw_payload_bytes);
        let schema_coverage = coverage_from_scene(&scene);
        let realized_coverage = coverage_from_packed(&packed);
        let final_summary_lines = final_coded_summary_lines(
            &coded_summary,
            raw_payload_bytes,
            &sorted,
            validation_artifacts
                .reconstructed
                .as_ref()
                .map(|scene| scene.gaussians.as_slice()),
            &sh_rest_stats,
        );
        let timing_lines = timing_summary_lines(
            &config,
            load_scene,
            sort_gaussians,
            pack_timings.geometry,
            pack_timings.sh_dc,
            pack_timings.sh_rest,
            pack_timings.sh_rest_breakdown.quantization,
            pack_timings.sh_rest_breakdown.buffer_construction,
            pack_timings.sh_rest_breakdown.atlas_construction,
            pack_timings.remaining_streams,
            write_container,
            &write_report,
            unpack_roundtrip_validation,
            total_start.elapsed(),
        );
        let coverage_lines = stream_coverage_lines(
            "vpcc_encoder",
            &schema_coverage,
            &realized_coverage,
            &realized_coverage,
        );
        let validation_notes = quick_validation
            .notes
            .iter()
            .map(|line| format!("Validation: {line}"))
            .chain(
                prewrite_validation
                    .notes
                    .iter()
                    .map(|line| format!("Validation prewrite: {line}")),
            )
            .collect::<Vec<_>>();

        logger.file_lines(verbose_summary_lines(
            &config,
            &config.input_path,
            &config.output_dir,
            logger.path(),
            &packed,
            raw_payload_bytes,
            original_ply_size,
            config.quality,
            storage_config,
            sh_rest_codec_config,
            sh_rest_stats.raw_bytes as u64,
            coded_summary.sh_rest_bytes,
            sh_rest_stats.storage,
            &validation_notes,
            sh_rest_sweep_report.as_ref(),
            &final_summary_lines,
            &timing_lines,
        ))?;
        if config.log_level.is_verbose() {
            logger.file_lines(verbose_run_log_lines(VerboseReportInput {
                input_path: &config.input_path,
                packed: &packed,
                original_sorted: &sorted,
                reconstructed: validation_artifacts.reconstructed.as_ref(),
                diagnostics: validation_artifacts.diagnostics.as_ref(),
                metrics_report: validation_artifacts.metrics_report.as_ref(),
                auxiliary_codec_metadata: validation_artifacts
                    .decoded_packed
                    .as_ref()
                    .and_then(|packed| packed.auxiliary_codec_metadata.as_ref()),
                comparison_lines: &comparison_lines,
                coverage_lines: &coverage_lines,
                sh_rest_sweep_report: sh_rest_sweep_report.as_ref(),
            }))?;
        } else if let Some(report) = sh_rest_sweep_report.as_ref() {
            logger.file_lines(report.lines.clone())?;
        }
    }

    logger.flush()?;
    notifier.run_completed(
        CompletionDetails::new(total_start.elapsed())
            .with_total_bytes(coded_summary.total_coded_bytes)
            .add("geometry bytes", coded_summary.geometry_bytes.to_string())
            .add("SH DC bytes", coded_summary.sh_dc_bytes.to_string())
            .add("SH rest bytes", coded_summary.sh_rest_bytes.to_string())
            .add("opacity bytes", coded_summary.opacity_bytes.to_string())
            .add("scale bytes", coded_summary.scale_bytes.to_string())
            .add("rotation bytes", coded_summary.rotation_bytes.to_string())
            .add("decoded PLY", decoded_ply.as_ref().map(|(path, _)| path.display().to_string()).unwrap_or_else(|| "not requested".to_string())),
    );
    Ok(())
}

fn prepare_output_dir(config: &CliConfig) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = Path::new(&config.output_dir);
    if output_dir.exists() && config.overwrite {
        fs::remove_dir_all(output_dir)?;
    } else if output_dir.exists() && !config.overwrite {
        return Err(format!(
            "output directory already exists: {}. Use --overwrite or --timestamp-output.",
            output_dir.display()
        )
        .into());
    }
    fs::create_dir_all(output_dir)?;
    Ok(())
}

fn comparison_lines(config: &CliConfig, raw_payload_bytes: usize) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(gsz_path) = &config.compare_gsz {
        match fs::metadata(gsz_path) {
            Ok(meta) => {
                let gsz_size = meta.len();
                lines.push(format!("Comparison .gsz size: {}", gsz_size));
                lines.push(format!(
                    "Raw payload / .gsz: {:.6}",
                    raw_payload_bytes as f64 / gsz_size as f64
                ));
            }
            Err(err) => {
                lines.push(format!(
                    "Comparison .gsz unavailable ({}): {}",
                    gsz_path, err
                ));
            }
        }
    }
    lines
}

fn write_decoded_ply(
    config: &CliConfig,
    decoded_packed: Option<&gaussian_packing::PackedRasterScene>,
) -> Result<(PathBuf, u64), Box<dyn std::error::Error>> {
    let decoded_packed_owned;
    let packed = if let Some(packed) = decoded_packed {
        packed
    } else {
        decoded_packed_owned = gaussian_packing::read_packed_raster_scene_dir(&config.output_dir)?;
        &decoded_packed_owned
    };
    let scene = gaussian_packing::unpack_raster_to_scene(packed);
    let output_path = config
        .decoded_ply_path
        .clone()
        .unwrap_or_else(|| Path::new(&config.output_dir).join("decoded.ply"));
    write_scene_to_ply(&output_path, &scene)?;
    let ply_size = fs::metadata(&output_path)?.len();
    Ok((output_path, ply_size))
}

fn load_or_pack_cached(
    sorted: &[gaussian_types::Gaussian],
    width: u32,
    packed_cache_path: Option<&PathBuf>,
) -> Result<(gaussian_packing::PackedRasterScene, PackRasterTimings), Box<dyn std::error::Error>> {
    if let Some(path) = packed_cache_path.filter(|path| path.is_file()) {
        let bytes = fs::read(path)?;
        Ok((
            bincode::deserialize::<gaussian_packing::PackedRasterScene>(&bytes)?,
            PackRasterTimings {
                geometry: Duration::ZERO,
                sh_dc: Duration::ZERO,
                sh_rest: Duration::ZERO,
                sh_rest_breakdown: ShRestPackTimings {
                    quantization: Duration::ZERO,
                    buffer_construction: Duration::ZERO,
                    atlas_construction: Duration::ZERO,
                },
                remaining_streams: Duration::ZERO,
            },
        ))
    } else {
        let (packed, timings) =
            gaussian_packing::pack_sorted_gaussians_to_raster_with_timing(sorted, width);
        if let Some(path) = packed_cache_path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, bincode::serialize(&packed)?)?;
        }
        Ok((packed, timings))
    }
}

fn load_scene_with_cache(
    config: &CliConfig,
) -> Result<gaussian_parser::Scene, Box<dyn std::error::Error>> {
    match config.cache_mode {
        CacheMode::None | CacheMode::Packed => {
            load_gaussians_from_ply(&config.input_path).map_err(Into::into)
        }
        CacheMode::Scene => {
            let cache_path = scene_cache_path(config);
            if cache_path.is_file() {
                let bytes = fs::read(&cache_path)?;
                Ok(bincode::deserialize(&bytes)?)
            } else {
                let scene = load_gaussians_from_ply(&config.input_path)?;
                if let Some(parent) = cache_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(cache_path, bincode::serialize(&scene)?)?;
                Ok(scene)
            }
        }
    }
}

fn apply_dev_subset(
    mut scene: gaussian_parser::Scene,
    config: &CliConfig,
) -> gaussian_parser::Scene {
    if config.sample_rate.is_none() && config.max_gaussians.is_none() {
        return scene;
    }

    let original_len = scene.gaussians.len();
    if let Some(sample_rate) = config.sample_rate {
        let target = ((original_len as f32) * sample_rate).ceil() as usize;
        let target = target.clamp(1, original_len.max(1));
        if target < scene.gaussians.len() {
            let step = (scene.gaussians.len() as f64 / target as f64).max(1.0);
            let mut sampled = Vec::with_capacity(target);
            let mut cursor = 0.0f64;
            while sampled.len() < target && (cursor as usize) < scene.gaussians.len() {
                sampled.push(scene.gaussians[cursor as usize].clone());
                cursor += step;
            }
            if sampled.is_empty() && !scene.gaussians.is_empty() {
                sampled.push(scene.gaussians[0].clone());
            }
            scene.gaussians = sampled;
        }
    }
    if let Some(max_gaussians) = config.max_gaussians {
        if scene.gaussians.len() > max_gaussians {
            scene.gaussians.truncate(max_gaussians);
        }
    }
    let (mins, maxes) = compute_scene_bounds(&scene.gaussians);
    scene.mins = mins;
    scene.maxes = maxes;
    scene
}

fn compute_scene_bounds(gaussians: &[gaussian_types::Gaussian]) -> ([f32; 3], [f32; 3]) {
    if let Some(first) = gaussians.first() {
        let mut mins = first.xyz;
        let mut maxes = first.xyz;
        for gaussian in &gaussians[1..] {
            for axis in 0..3 {
                mins[axis] = mins[axis].min(gaussian.xyz[axis]);
                maxes[axis] = maxes[axis].max(gaussian.xyz[axis]);
            }
        }
        (mins, maxes)
    } else {
        ([0.0; 3], [0.0; 3])
    }
}

fn scene_cache_path(config: &CliConfig) -> PathBuf {
    cache_root(config).join(format!(
        "scene_{}_{}.bin",
        sanitize_cache_stem(&config.input_path),
        input_signature(&config.input_path),
    ))
}

fn packed_cache_path(config: &CliConfig) -> Option<PathBuf> {
    (config.cache_mode == CacheMode::Packed).then(|| {
        cache_root(config).join(format!(
            "packed_{}_{}_{}_w{}.bin",
            sanitize_cache_stem(&config.input_path),
            input_signature(&config.input_path),
            subset_signature(config),
            config.width,
        ))
    })
}

fn cache_root(config: &CliConfig) -> PathBuf {
    Path::new(&config.output_dir).join(".vpcc_cache")
}

fn sanitize_cache_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("scene")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn input_signature(path: &str) -> String {
    match fs::metadata(path).and_then(|meta| meta.modified().map(|mtime| (meta.len(), mtime))) {
        Ok((len, mtime)) => format!(
            "{}_{}",
            len,
            mtime
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
        Err(_) => "unknown".to_string(),
    }
}

fn subset_signature(config: &CliConfig) -> String {
    format!(
        "sr{}_mg{}",
        config
            .sample_rate
            .map(|rate| format!("{rate:.4}"))
            .unwrap_or_else(|| "none".to_string()),
        config
            .max_gaussians
            .map(|count| count.to_string())
            .unwrap_or_else(|| "none".to_string())
    )
}

fn validate_prewrite(
    sorted: &[gaussian_types::Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    mode: ValidationMode,
) -> Result<QuickValidationSummary, Box<dyn std::error::Error>> {
    if matches!(mode, ValidationMode::None) {
        return Ok(QuickValidationSummary {
            notes: vec!["validation skipped by configuration".to_string()],
        });
    }
    let plane_len = (packed.width as usize) * (packed.height as usize);
    let occupied = packed.occupancy.iter().filter(|&&value| value != 0).count();
    let expected_payload_bytes =
        packed.num_gaussians as usize * packed.sh_rest_len as usize * std::mem::size_of::<u16>();
    let notes = vec![
        format!(
            "packed invariants: plane_len={} occupancy_len={} occupied={} gaussians={}",
            plane_len,
            packed.occupancy.len(),
            occupied,
            packed.num_gaussians
        ),
        format!(
            "sh_rest payload bytes: expected={} actual={}",
            expected_payload_bytes,
            packed.sh_rest_payload.len()
        ),
        format!(
            "sorted input count matches packed: {}",
            sorted.len() == packed.num_gaussians as usize
        ),
    ];
    if packed.occupancy.len() != plane_len
        || occupied != packed.num_gaussians as usize
        || packed.sh_rest_payload.len() != expected_payload_bytes
    {
        return Err("light validation failed on packed invariants".into());
    }
    Ok(QuickValidationSummary { notes })
}

fn validate_postwrite(
    config: &CliConfig,
    sorted: &[gaussian_types::Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
) -> Result<(QuickValidationSummary, ValidationArtifacts), Box<dyn std::error::Error>> {
    match config.validation {
        ValidationMode::None => Ok((
            QuickValidationSummary {
                notes: vec!["validation skipped by configuration".to_string()],
            },
            ValidationArtifacts {
                decoded_packed: None,
                reconstructed: None,
                diagnostics: None,
                metrics_report: None,
            },
        )),
        ValidationMode::Light => Ok((
            QuickValidationSummary {
                notes: vec![
                    format!("coded bytes measured: {}", coded_summary.total_coded_bytes),
                    format!("occupancy bytes written: {}", coded_summary.occupancy_bytes),
                    format!("metadata bytes written: {}", coded_summary.metadata_bytes),
                ],
            },
            ValidationArtifacts {
                decoded_packed: None,
                reconstructed: None,
                diagnostics: None,
                metrics_report: None,
            },
        )),
        ValidationMode::Full => {
            let decoded_packed =
                gaussian_packing::read_packed_raster_scene_dir(&config.output_dir)?;
            let reconstructed = gaussian_packing::unpack_raster_to_scene(&decoded_packed);
            let diagnostics = gaussian_packing::compute_roundtrip_diagnostics(
                sorted,
                &reconstructed.gaussians,
                &decoded_packed,
            );
            let sh_rest_stats = gaussian_packing::sh_rest_payload_stats(packed);
            let metrics_report = crate::metrics::evaluate_metrics(
                sorted,
                &reconstructed.gaussians,
                packed,
                &decoded_packed,
                coded_summary,
                &sh_rest_stats,
            );
            Ok((
                QuickValidationSummary {
                    notes: vec![
                        format!(
                            "decoded packed scene gaussians: {}",
                            decoded_packed.num_gaussians
                        ),
                        "full decode/unpack roundtrip validation completed".to_string(),
                    ],
                },
                ValidationArtifacts {
                    decoded_packed: Some(decoded_packed),
                    reconstructed: Some(reconstructed),
                    diagnostics: Some(diagnostics),
                    metrics_report: Some(metrics_report),
                },
            ))
        }
    }
}

fn sh_dc_storage_from_env(
    default: GroupStorageMode,
) -> Result<GroupStorageMode, Box<dyn std::error::Error>> {
    match std::env::var("VPCC_SH_DC_STORAGE").ok().as_deref() {
        None | Some("") => Ok(default),
        Some("byte_split_deflate") => Ok(GroupStorageMode::ByteSplitDeflate),
        Some("video_atlas") => Ok(GroupStorageMode::CodedVideoAtlas),
        Some("raw") => Ok(GroupStorageMode::Raw),
        Some(other) => Err(format!(
            "unsupported VPCC_SH_DC_STORAGE={other}; expected byte_split_deflate, video_atlas, or raw"
        )
        .into()),
    }
}

fn progress_bar(len: u64) -> Result<ProgressBar, Box<dyn std::error::Error>> {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )?
        .progress_chars("=>-"),
    );
    Ok(pb)
}
