use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use gaussian_packing::{
    AuxiliaryCodecMetadata, PackedContainerByteSummary, PackedRasterScene, ShRestPayloadStats,
    ShRestTierSpec, ShRestVideoAtlasCodecConfig, decode_sh_rest_payload_values,
    encode_sh_rest_payload_from_values, packed_scene_with_sh_rest_streams,
    read_sh_rest_tiered_video_atlas_streams, read_sh_rest_video_atlas_streams,
    sh_rest_streams_from_packed, unpack_raster_to_scene, write_sh_rest_tiered_video_atlas_streams,
    write_sh_rest_video_atlas_streams,
};

use crate::cli::{CliConfig, ShRestSweepOverrides, parse_ffmpeg_preset};
use crate::metrics::{MetricsReport, group_metric_lines, stream_metric_lines};
use crate::report::{ratio_u64, reduction_percent_u64, sh_rest_error_stats};
use gaussian_packing::QualityPreset;

#[derive(Debug, Clone)]
pub struct ShRestSweepReport {
    pub lines: Vec<String>,
    pub best: Option<ShRestExperimentResult>,
}

#[derive(Debug, Clone)]
pub struct ShRestExperimentResult {
    pub name: String,
    pub sh_rest_bytes: u64,
    pub total_bytes: u64,
    pub total_reduction_percent: f64,
    pub rmse: f64,
    pub maxabs: f64,
    pub value_psnr_mean: f64,
    pub atlas_psnr_mean: Option<f64>,
    pub atlas_ssim_mean: Option<f64>,
    pub pruned_coefficients: usize,
    pub pruned_fraction_percent: f64,
    pub note: String,
}

#[derive(Debug, Clone, Copy)]
pub struct SourceNormalsSummary {
    pub any_present: bool,
    pub valid_unitish_count: usize,
}

#[derive(Debug, Clone)]
pub struct VerboseReportInput<'a> {
    pub input_path: &'a str,
    pub packed: &'a PackedRasterScene,
    pub original_sorted: &'a [gaussian_types::Gaussian],
    pub reconstructed: Option<&'a gaussian_parser::Scene>,
    pub diagnostics: Option<&'a gaussian_packing::RoundtripDiagnostics>,
    pub metrics_report: Option<&'a MetricsReport>,
    pub auxiliary_codec_metadata: Option<&'a AuxiliaryCodecMetadata>,
    pub comparison_lines: &'a [String],
    pub coverage_lines: &'a [String],
    pub sh_rest_sweep_report: Option<&'a ShRestSweepReport>,
}

pub fn should_run_sh_rest_sweep(config: &CliConfig) -> bool {
    config
        .sh_rest_sweep
        .enabled
        .unwrap_or(matches!(config.mode, crate::cli::RunMode::Bench))
}

pub fn source_normals_summary(
    original_sorted: &[gaussian_types::Gaussian],
) -> SourceNormalsSummary {
    let mut valid_unitish_count = 0usize;
    let any_present = original_sorted.iter().any(|g| g.normals.is_some());
    for gaussian in original_sorted {
        let normal = gaussian.normals.unwrap_or([0.0; 3]);
        let norm = vec3_norm_f64(normal);
        if (norm - 1.0).abs() <= 1e-3 {
            valid_unitish_count += 1;
        }
    }
    SourceNormalsSummary {
        any_present,
        valid_unitish_count,
    }
}

pub fn verbose_run_log_lines(input: VerboseReportInput<'_>) -> Vec<String> {
    let mut lines = vec!["Verbose Diagnostics:".to_string()];
    lines.extend(input.comparison_lines.iter().cloned());
    lines.extend(source_ply_investigation_lines(
        input.input_path,
        input.original_sorted,
        input.packed,
    ));
    if let Some(diagnostics) = input.diagnostics {
        lines.extend(diagnostic_detail_lines(diagnostics));
    } else {
        lines.push("Roundtrip Diagnostics: skipped".to_string());
    }
    if let (Some(reconstructed), Some(metrics_report)) = (input.reconstructed, input.metrics_report)
    {
        lines.extend(experimental_group_log_lines(
            input.original_sorted,
            &reconstructed.gaussians,
            input.auxiliary_codec_metadata,
        ));
        lines.extend(group_metric_lines(&metrics_report.groups));
        lines.extend(stream_metric_lines(&metrics_report.streams));
    } else {
        lines.push("Metrics: skipped".to_string());
    }
    if let Some(report) = input.sh_rest_sweep_report {
        lines.extend(report.lines.iter().cloned());
    }
    lines.extend(input.coverage_lines.iter().cloned());
    lines
}

pub fn run_sh_rest_sweep_report(
    config: &CliConfig,
    original_sorted: &[gaussian_types::Gaussian],
    packed: &PackedRasterScene,
    baseline_coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
) -> Result<ShRestSweepReport, Box<dyn std::error::Error>> {
    let stream_count = packed.sh_rest_len as usize;
    if stream_count == 0 {
        return Ok(ShRestSweepReport {
            lines: vec!["SH Rest Optimization Sweep: skipped (no sh_rest streams)".to_string()],
            best: None,
        });
    }

    let config = sh_rest_sweep_config(stream_count, &config.sh_rest_sweep);
    let mut lines = vec![
        "SH Rest Optimization Sweep:".to_string(),
        format!(
            "- config: uniform_crfs={:?} uniform_presets={:?} tier_bounds=[{},{}] tier_crfs={:?} tier_presets={:?} prune_thresholds={:?} prune_start_band={}",
            config.uniform_crfs,
            config.uniform_presets,
            config.tier_bounds[0],
            config.tier_bounds[1],
            config.tier_crfs,
            config.tier_presets,
            config.prune_thresholds,
            config.prune_start_band,
        ),
    ];

    let mut successful = Vec::new();
    for &preset in &config.uniform_presets {
        for &crf in &config.uniform_crfs {
            match run_sh_rest_uniform_experiment(
                original_sorted,
                packed,
                baseline_coded_summary,
                raw_payload_bytes,
                sh_rest_codec_with_crf(preset, crf),
            ) {
                Ok(result) => {
                    lines.push(format_experiment_line(&result));
                    successful.push(result);
                }
                Err(err) => lines.push(format!("- experiment failed: {err}")),
            }
        }
    }
    for prune_threshold in
        std::iter::once(None).chain(config.prune_thresholds.iter().copied().map(Some))
    {
        match run_sh_rest_tiered_experiment(
            original_sorted,
            packed,
            baseline_coded_summary,
            raw_payload_bytes,
            &config,
            prune_threshold,
        ) {
            Ok(result) => {
                lines.push(format_experiment_line(&result));
                successful.push(result);
            }
            Err(err) => lines.push(format!("- experiment failed: {err}")),
        }
    }

    let best = successful
        .into_iter()
        .min_by_key(|result| result.total_bytes);
    if let Some(best_result) = best.as_ref() {
        lines.push(format!(
            "- best_by_total_bytes: {} sh_rest_bytes={} total_bytes={} total_reduction={}% rmse={} maxabs={} value_psnr_mean={} atlas_psnr_mean={} atlas_ssim_mean={} pruned_coeffs={} pruned_fraction={:.2}% note={}",
            best_result.name,
            best_result.sh_rest_bytes,
            best_result.total_bytes,
            crate::metrics::format_metric(best_result.total_reduction_percent),
            crate::metrics::format_metric(best_result.rmse),
            crate::metrics::format_metric(best_result.maxabs),
            crate::metrics::format_metric(best_result.value_psnr_mean),
            crate::metrics::format_optional_metric(best_result.atlas_psnr_mean),
            crate::metrics::format_optional_metric(best_result.atlas_ssim_mean),
            best_result.pruned_coefficients,
            best_result.pruned_fraction_percent,
            best_result.note,
        ));
    }

    Ok(ShRestSweepReport { lines, best })
}

fn format_experiment_line(result: &ShRestExperimentResult) -> String {
    format!(
        "- {}: sh_rest_bytes={} total_bytes={} total_reduction={}% rmse={} maxabs={} value_psnr_mean={} atlas_psnr_mean={} atlas_ssim_mean={} pruned_coeffs={} pruned_fraction={}% note={}",
        result.name,
        result.sh_rest_bytes,
        result.total_bytes,
        crate::metrics::format_metric(result.total_reduction_percent),
        crate::metrics::format_metric(result.rmse),
        crate::metrics::format_metric(result.maxabs),
        crate::metrics::format_metric(result.value_psnr_mean),
        crate::metrics::format_optional_metric(result.atlas_psnr_mean),
        crate::metrics::format_optional_metric(result.atlas_ssim_mean),
        result.pruned_coefficients,
        crate::metrics::format_metric(result.pruned_fraction_percent),
        result.note,
    )
}

#[derive(Debug, Clone)]
struct ShRestSweepConfig {
    uniform_crfs: Vec<u8>,
    uniform_presets: Vec<&'static str>,
    tier_bounds: [usize; 2],
    tier_crfs: [u8; 3],
    tier_presets: [&'static str; 3],
    prune_thresholds: Vec<f32>,
    prune_start_band: usize,
}

fn sh_rest_sweep_config(
    stream_count: usize,
    overrides: &ShRestSweepOverrides,
) -> ShRestSweepConfig {
    let tier_bounds = overrides
        .tier_bounds
        .or_else(|| {
            parse_usize_list_env("VPCC_SH_REST_TIER_BOUNDS")
                .and_then(|values| (values.len() == 2).then_some([values[0], values[1]]))
        })
        .filter(|bounds| bounds[0] < bounds[1] && bounds[1] <= stream_count)
        .unwrap_or([
            (stream_count / 3).max(1),
            ((stream_count * 2) / 3).max(2).min(stream_count),
        ]);
    let prune_start_band = overrides
        .prune_start_band
        .or_else(|| {
            std::env::var("VPCC_SH_REST_PRUNE_START_BAND")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(tier_bounds[1]);

    ShRestSweepConfig {
        uniform_crfs: overrides
            .uniform_crfs
            .clone()
            .or_else(|| parse_u8_list_env("VPCC_SH_REST_SWEEP_CRFS"))
            .unwrap_or_else(|| vec![12, 16, 20]),
        uniform_presets: overrides
            .uniform_presets
            .clone()
            .or_else(|| parse_preset_list_env("VPCC_SH_REST_SWEEP_PRESETS"))
            .unwrap_or_else(|| vec!["medium", "slow"]),
        tier_bounds,
        tier_crfs: overrides
            .tier_crfs
            .or_else(|| {
                parse_u8_list_env("VPCC_SH_REST_TIER_CRFS").and_then(|values| {
                    (values.len() == 3).then_some([values[0], values[1], values[2]])
                })
            })
            .unwrap_or([12, 18, 24]),
        tier_presets: overrides
            .tier_presets
            .or_else(|| {
                parse_preset_list_env("VPCC_SH_REST_TIER_PRESETS").and_then(|values| {
                    (values.len() == 3).then_some([values[0], values[1], values[2]])
                })
            })
            .unwrap_or(["medium", "medium", "slow"]),
        prune_thresholds: overrides
            .prune_thresholds
            .clone()
            .or_else(|| parse_f32_list_env("VPCC_SH_REST_PRUNE_THRESHOLDS"))
            .unwrap_or_else(|| vec![0.0025]),
        prune_start_band,
    }
}

fn run_sh_rest_uniform_experiment(
    original_sorted: &[gaussian_types::Gaussian],
    packed: &PackedRasterScene,
    baseline_coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
    codec: ShRestVideoAtlasCodecConfig,
) -> Result<ShRestExperimentResult, Box<dyn std::error::Error>> {
    let streams = sh_rest_streams_from_packed(packed);
    let temp_dir = sh_rest_temp_dir(&format!(
        "uniform_{}_crf{}",
        codec.ffmpeg_preset.unwrap_or("none"),
        codec.crf.unwrap_or(0)
    ));
    std::fs::create_dir_all(&temp_dir)?;
    write_sh_rest_video_atlas_streams(
        &streams,
        packed.width,
        packed.height,
        packed.num_gaussians as usize,
        &temp_dir,
        codec,
    )?;
    let decoded_streams =
        read_sh_rest_video_atlas_streams(&temp_dir, packed.width, packed.height, streams.len())?;
    let sh_rest_bytes = sh_rest_uniform_video_bytes(&temp_dir)?;
    std::fs::remove_dir_all(&temp_dir)?;
    sh_rest_result_from_streams(
        original_sorted,
        packed,
        packed,
        baseline_coded_summary,
        raw_payload_bytes,
        decoded_streams,
        sh_rest_bytes,
        format!(
            "uniform[preset={},crf={}]",
            codec.ffmpeg_preset.unwrap_or("n/a"),
            codec
                .crf
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ),
        0,
        "single atlas for all sh_rest streams".to_string(),
    )
}

fn run_sh_rest_tiered_experiment(
    original_sorted: &[gaussian_types::Gaussian],
    packed: &PackedRasterScene,
    baseline_coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
    config: &ShRestSweepConfig,
    prune_threshold: Option<f32>,
) -> Result<ShRestExperimentResult, Box<dyn std::error::Error>> {
    let tiers = vec![
        ShRestTierSpec {
            label: "low".to_string(),
            start_stream: 0,
            end_stream: config.tier_bounds[0],
            codec_config: sh_rest_codec_with_crf(config.tier_presets[0], config.tier_crfs[0]),
        },
        ShRestTierSpec {
            label: "mid".to_string(),
            start_stream: config.tier_bounds[0],
            end_stream: config.tier_bounds[1],
            codec_config: sh_rest_codec_with_crf(config.tier_presets[1], config.tier_crfs[1]),
        },
        ShRestTierSpec {
            label: "high".to_string(),
            start_stream: config.tier_bounds[1],
            end_stream: packed.sh_rest_len as usize,
            codec_config: sh_rest_codec_with_crf(config.tier_presets[2], config.tier_crfs[2]),
        },
    ];
    let (candidate_packed, pruned_coefficients, pruned_fraction_percent) =
        sh_rest_pruned_candidate(packed, prune_threshold, config.prune_start_band)?;
    let streams = sh_rest_streams_from_packed(&candidate_packed);
    let temp_label = match prune_threshold {
        Some(threshold) => format!("tiered_prune_{threshold:.4}"),
        None => "tiered".to_string(),
    };
    let temp_dir = sh_rest_temp_dir(&temp_label);
    std::fs::create_dir_all(&temp_dir)?;
    write_sh_rest_tiered_video_atlas_streams(
        &streams,
        packed.width,
        packed.height,
        packed.num_gaussians as usize,
        &temp_dir,
        &tiers,
    )?;
    let decoded_streams =
        read_sh_rest_tiered_video_atlas_streams(&temp_dir, packed.width, packed.height, &tiers)?;
    let sh_rest_bytes = sh_rest_tiered_video_bytes(&temp_dir, tiers.len())?;
    std::fs::remove_dir_all(&temp_dir)?;
    let name = match prune_threshold {
        Some(threshold) => format!(
            "tiered[{}|{}|{}]+prune[start={},threshold={:.6}]",
            tier_name(&tiers[0]),
            tier_name(&tiers[1]),
            tier_name(&tiers[2]),
            config.prune_start_band,
            threshold,
        ),
        None => format!(
            "tiered[{}|{}|{}]",
            tier_name(&tiers[0]),
            tier_name(&tiers[1]),
            tier_name(&tiers[2]),
        ),
    };
    sh_rest_result_from_streams(
        original_sorted,
        packed,
        &candidate_packed,
        baseline_coded_summary,
        raw_payload_bytes,
        decoded_streams,
        sh_rest_bytes,
        name,
        pruned_coefficients,
        format!(
            "three-tier atlas with bounds [{}, {}) and [{}, {})",
            tiers[0].start_stream,
            tiers[0].end_stream,
            tiers[1].start_stream,
            tiers[2].start_stream
        ),
    )
    .map(|mut result| {
        result.pruned_fraction_percent = pruned_fraction_percent;
        result
    })
}

fn sh_rest_result_from_streams(
    original_sorted: &[gaussian_types::Gaussian],
    original_packed: &PackedRasterScene,
    packed_for_candidate: &PackedRasterScene,
    baseline_coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
    decoded_streams: Vec<Vec<u16>>,
    sh_rest_bytes: u64,
    name: String,
    pruned_coefficients: usize,
    note: String,
) -> Result<ShRestExperimentResult, Box<dyn std::error::Error>> {
    let decoded_packed = packed_scene_with_sh_rest_streams(packed_for_candidate, &decoded_streams)?;
    let reconstructed = unpack_raster_to_scene(&decoded_packed);
    let mut candidate_summary = baseline_coded_summary.clone();
    candidate_summary.sh_rest_bytes = sh_rest_bytes;
    candidate_summary.total_coded_bytes = baseline_coded_summary.total_coded_bytes
        - baseline_coded_summary.sh_rest_bytes
        + sh_rest_bytes;
    let sh_rest_stats = ShRestPayloadStats {
        coded_bytes: sh_rest_bytes as usize,
        raw_bytes: packed_for_candidate.num_gaussians as usize
            * packed_for_candidate.sh_rest_len as usize
            * std::mem::size_of::<f32>(),
        storage: "sh_rest_experiment",
    };
    let metrics_report = crate::metrics::evaluate_metrics(
        original_sorted,
        &reconstructed.gaussians,
        original_packed,
        &decoded_packed,
        &candidate_summary,
        &sh_rest_stats,
    );
    let sh_rest_group = metrics_report
        .groups
        .iter()
        .find(|group| group.name == "sh_rest")
        .ok_or("missing sh_rest metrics group")?;
    let (rmse, maxabs) = sh_rest_error_stats(original_sorted, &reconstructed.gaussians);
    Ok(ShRestExperimentResult {
        name,
        sh_rest_bytes,
        total_bytes: candidate_summary.total_coded_bytes,
        total_reduction_percent: reduction_percent_u64(
            raw_payload_bytes as u64,
            candidate_summary.total_coded_bytes,
        ),
        rmse,
        maxabs,
        value_psnr_mean: sh_rest_group.value_psnr_mean,
        atlas_psnr_mean: sh_rest_group.atlas_psnr_mean,
        atlas_ssim_mean: sh_rest_group.atlas_ssim_mean,
        pruned_coefficients,
        pruned_fraction_percent: 0.0,
        note,
    })
}

fn sh_rest_pruned_candidate(
    packed: &PackedRasterScene,
    prune_threshold: Option<f32>,
    prune_start_band: usize,
) -> Result<(PackedRasterScene, usize, f64), Box<dyn std::error::Error>> {
    let Some(threshold) = prune_threshold else {
        return Ok((packed.clone(), 0, 0.0));
    };
    let gaussian_len = packed.num_gaussians as usize;
    let sh_rest_len = packed.sh_rest_len as usize;
    let mut values = decode_sh_rest_payload_values(
        &packed.sh_rest_payload,
        &packed.sh_rest_quant,
        gaussian_len,
    )?;
    let mut pruned_coefficients = 0usize;
    let mut eligible_coefficients = 0usize;
    for gaussian_idx in 0..gaussian_len {
        for stream_idx in prune_start_band.min(sh_rest_len)..sh_rest_len {
            eligible_coefficients += 1;
            let value_idx = gaussian_idx * sh_rest_len + stream_idx;
            if values[value_idx].abs() < threshold {
                values[value_idx] = 0.0;
                pruned_coefficients += 1;
            }
        }
    }
    let payload = encode_sh_rest_payload_from_values(&values, &packed.sh_rest_quant, gaussian_len)?;
    let mut candidate = packed.clone();
    candidate.sh_rest_payload = payload;
    let pruned_fraction_percent = if eligible_coefficients == 0 {
        0.0
    } else {
        100.0 * pruned_coefficients as f64 / eligible_coefficients as f64
    };
    Ok((candidate, pruned_coefficients, pruned_fraction_percent))
}

fn sh_rest_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vpcc_sh_rest_sweep_{}_{}_{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn sh_rest_uniform_video_bytes(dir: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(std::fs::metadata(dir.join("sh_rest_hi.mkv"))?.len()
        + std::fs::metadata(dir.join("sh_rest_lo.mkv"))?.len())
}

fn sh_rest_tiered_video_bytes(
    dir: &Path,
    tier_count: usize,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total = 0u64;
    for tier_idx in 0..tier_count {
        total += std::fs::metadata(dir.join(format!("sh_rest_tier_{tier_idx}_hi.mkv")))?.len();
        total += std::fs::metadata(dir.join(format!("sh_rest_tier_{tier_idx}_lo.mkv")))?.len();
    }
    Ok(total)
}

fn tier_name(tier: &ShRestTierSpec) -> String {
    format!(
        "{}:{}-{}@{}:{}",
        tier.label,
        tier.start_stream,
        tier.end_stream,
        tier.codec_config.ffmpeg_preset.unwrap_or("n/a"),
        tier.codec_config
            .crf
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
    )
}

fn sh_rest_codec_with_crf(preset: &'static str, crf: u8) -> ShRestVideoAtlasCodecConfig {
    ShRestVideoAtlasCodecConfig {
        quality_preset: QualityPreset::VeryGood,
        codec: "libx264rgb",
        pixel_format: "rgb24",
        ffmpeg_preset: Some(preset),
        crf: Some(crf),
        lossless: false,
    }
}

fn parse_u8_list_env(key: &str) -> Option<Vec<u8>> {
    let values = std::env::var(key)
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<u8>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then_some(values)
}

fn parse_usize_list_env(key: &str) -> Option<Vec<usize>> {
    let values = std::env::var(key)
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then_some(values)
}

fn parse_f32_list_env(key: &str) -> Option<Vec<f32>> {
    let values = std::env::var(key)
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<f32>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then_some(values)
}

fn parse_preset_list_env(key: &str) -> Option<Vec<&'static str>> {
    let values = std::env::var(key)
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(parse_ffmpeg_preset)
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then_some(values)
}

fn diagnostic_detail_lines(diagnostics: &gaussian_packing::RoundtripDiagnostics) -> Vec<String> {
    let mut lines = vec![
        "Roundtrip Diagnostics:".to_string(),
        format!(
            "- xyz rmse: x={:.9} y={:.9} z={:.9}",
            diagnostics.xyz_rmse[0], diagnostics.xyz_rmse[1], diagnostics.xyz_rmse[2]
        ),
        format!(
            "- sh_dc rmse: r={:.9} g={:.9} b={:.9}",
            diagnostics.sh_dc_rmse[0], diagnostics.sh_dc_rmse[1], diagnostics.sh_dc_rmse[2]
        ),
    ];
    lines.push("Plane stats: verbose diagnostic output retained behind --debug".to_string());
    lines
}

fn experimental_group_log_lines(
    original_sorted: &[gaussian_types::Gaussian],
    reconstructed: &[gaussian_types::Gaussian],
    auxiliary_codec_metadata: Option<&AuxiliaryCodecMetadata>,
) -> Vec<String> {
    let Some(metadata) = auxiliary_codec_metadata else {
        return vec!["Auxiliary Group Diagnostics: unavailable".to_string()];
    };

    let mut lines = vec!["Auxiliary Group Diagnostics:".to_string()];
    let opacity_original = original_sorted
        .iter()
        .map(|g| g.opacity)
        .collect::<Vec<_>>();
    let opacity_recon = reconstructed.iter().map(|g| g.opacity).collect::<Vec<_>>();
    let (opacity_rmse, opacity_maxabs) = scalar_error_stats(&opacity_original, &opacity_recon);
    lines.push(format!(
        "- opacity: strategy={:?} raw_bytes={} coded_bytes={} ratio={} reduction={}% rmse={} maxabs={}",
        metadata.opacity.strategy,
        metadata.opacity.raw_stream_bytes,
        payload_bytes(&metadata.opacity.payload_streams),
        crate::metrics::format_metric(ratio_u64(
            metadata.opacity.raw_stream_bytes,
            payload_bytes(&metadata.opacity.payload_streams)
        )),
        crate::metrics::format_metric(reduction_percent_u64(
            metadata.opacity.raw_stream_bytes,
            payload_bytes(&metadata.opacity.payload_streams)
        )),
        crate::metrics::format_metric(opacity_rmse),
        crate::metrics::format_metric(opacity_maxabs),
    ));
    lines.push(format!(
        "- rotation angle rmse/max deg: {} / {}",
        crate::metrics::format_metric(quaternion_angle_error_deg(original_sorted, reconstructed).0),
        crate::metrics::format_metric(quaternion_angle_error_deg(original_sorted, reconstructed).1),
    ));
    lines
}

fn payload_bytes(streams: &[gaussian_packing::StreamPayloadLayout]) -> u64 {
    streams.iter().map(|stream| stream.coded_bytes).sum()
}

fn scalar_error_stats(original: &[f32], reconstructed: &[f32]) -> (f64, f64) {
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&lhs, &rhs) in original.iter().zip(reconstructed.iter()) {
        let err = rhs as f64 - lhs as f64;
        sum_sq += err * err;
        max_abs = max_abs.max(err.abs());
    }
    let rmse = if original.is_empty() {
        0.0
    } else {
        (sum_sq / original.len() as f64).sqrt()
    };
    (rmse, max_abs)
}

fn quaternion_angle_error_deg(
    original_sorted: &[gaussian_types::Gaussian],
    reconstructed: &[gaussian_types::Gaussian],
) -> (f64, f64) {
    let mut sum_sq = 0.0f64;
    let mut max_deg = 0.0f64;
    let mut count = 0usize;
    for (lhs, rhs) in original_sorted.iter().zip(reconstructed.iter()) {
        let dot = (lhs.rot[0] * rhs.rot[0]
            + lhs.rot[1] * rhs.rot[1]
            + lhs.rot[2] * rhs.rot[2]
            + lhs.rot[3] * rhs.rot[3])
            .abs()
            .clamp(0.0, 1.0) as f64;
        let angle_deg = (2.0 * dot.acos()).to_degrees();
        sum_sq += angle_deg * angle_deg;
        max_deg = max_deg.max(angle_deg);
        count += 1;
    }
    if count == 0 {
        (0.0, 0.0)
    } else {
        ((sum_sq / count as f64).sqrt(), max_deg)
    }
}

fn source_ply_investigation_lines(
    input_path: &str,
    original_sorted: &[gaussian_types::Gaussian],
    packed: &PackedRasterScene,
) -> Vec<String> {
    let source_diag = source_normals_diagnostics(original_sorted);
    let mut lines = vec!["Source PLY Investigation:".to_string()];
    match read_ply_header_lines(input_path) {
        Ok(header_lines) => {
            let properties = header_lines
                .iter()
                .filter(|line| line.starts_with("property "))
                .cloned()
                .collect::<Vec<_>>();
            lines.push(format!("- header property count={}", properties.len()));
            lines.push(format!(
                "- normal properties present: nx={} ny={} nz={}",
                properties.iter().any(|line| line.ends_with(" nx")),
                properties.iter().any(|line| line.ends_with(" ny")),
                properties.iter().any(|line| line.ends_with(" nz")),
            ));
        }
        Err(err) => lines.push(format!("- header inspection failed: {err}")),
    }
    lines.push(format!(
        "- parsed normals: any_present={} zero_vector_count={} valid_unitish_count={}",
        source_diag.any_present, source_diag.zero_vector_count, source_diag.valid_unitish_count
    ));
    if let Ok(raw_probe) = raw_ply_normal_probe(input_path, 3) {
        lines.push(format!(
            "- raw file probe: format={} vertex_count={} stride_bytes={}",
            raw_probe.format, raw_probe.vertex_count, raw_probe.stride_bytes
        ));
    }
    for idx in 0..packed.normals_x.len().min(3) {
        lines.push(format!(
            "  packed normals sample {}: [{:.6},{:.6},{:.6}]",
            idx, packed.normals_x[idx], packed.normals_y[idx], packed.normals_z[idx],
        ));
    }
    lines
}

#[derive(Debug, Clone)]
struct SourceNormalsDiagnostics {
    any_present: bool,
    zero_vector_count: usize,
    valid_unitish_count: usize,
}

#[derive(Debug, Clone)]
struct RawPlyNormalProbe {
    format: String,
    vertex_count: usize,
    stride_bytes: usize,
}

fn source_normals_diagnostics(
    original_sorted: &[gaussian_types::Gaussian],
) -> SourceNormalsDiagnostics {
    let any_present = original_sorted.iter().any(|g| g.normals.is_some());
    let mut zero_vector_count = 0usize;
    let mut valid_unitish_count = 0usize;
    for gaussian in original_sorted {
        let normal = gaussian.normals.unwrap_or([0.0; 3]);
        let norm = vec3_norm_f64(normal);
        if norm <= 1e-12 {
            zero_vector_count += 1;
        }
        if (norm - 1.0).abs() <= 1e-3 {
            valid_unitish_count += 1;
        }
    }
    SourceNormalsDiagnostics {
        any_present,
        zero_vector_count,
        valid_unitish_count,
    }
}

fn read_ply_header_lines(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end().to_string();
        lines.push(trimmed.clone());
        if trimmed == "end_header" {
            break;
        }
    }
    Ok(lines)
}

fn raw_ply_normal_probe(
    path: &str,
    _sample_count: usize,
) -> Result<RawPlyNormalProbe, Box<dyn std::error::Error>> {
    let header_lines = read_ply_header_lines(path)?;
    let mut format = "unknown".to_string();
    let mut vertex_count = 0usize;
    let mut properties = 0usize;
    let mut in_vertex = false;
    for trimmed in header_lines {
        if trimmed.starts_with("format ") {
            format = trimmed
                .split_whitespace()
                .nth(1)
                .unwrap_or("unknown")
                .to_string();
        }
        if trimmed.starts_with("element ") {
            in_vertex = trimmed.starts_with("element vertex ");
            if in_vertex {
                vertex_count = trimmed
                    .split_whitespace()
                    .last()
                    .unwrap_or("0")
                    .parse::<usize>()?;
            }
        } else if in_vertex && trimmed.starts_with("property ") {
            properties += 1;
        }
    }
    Ok(RawPlyNormalProbe {
        format,
        vertex_count,
        stride_bytes: properties * 4,
    })
}

fn vec3_norm_f64(v: [f32; 3]) -> f64 {
    ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)).sqrt()
}
