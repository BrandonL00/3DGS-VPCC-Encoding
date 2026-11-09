use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use gaussian_packing::{
    ContainerWriteTimings, PackedContainerByteSummary, PackedRasterScene,
    PackedRasterStorageConfig, QualityPreset, ShRestPayloadStats, ShRestVideoAtlasCodecConfig,
};

use crate::cli::CliConfig;
use crate::debug::ShRestSweepReport;

pub fn concise_summary_lines(
    config: &CliConfig,
    output_dir: &str,
    coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
    total_runtime: Duration,
    decoded_ply: Option<(&Path, u64)>,
    sh_rest_best: Option<&crate::debug::ShRestExperimentResult>,
) -> Vec<String> {
    let raw_bytes = raw_payload_bytes as u64;
    let coded_bytes = coded_summary.total_coded_bytes;
    let ratio = ratio_u64(raw_bytes, coded_bytes);
    let reduction = reduction_percent_u64(raw_bytes, coded_bytes);
    let mut lines = vec![
        format!("Output directory: {output_dir}"),
        format!(
            "Coded bytes: total={} geometry={} sh_dc={} sh_rest={} opacity={} scale={} rotation={} normals={}",
            coded_summary.total_coded_bytes,
            coded_summary.geometry_bytes,
            coded_summary.sh_dc_bytes,
            coded_summary.sh_rest_bytes,
            coded_summary.opacity_bytes,
            coded_summary.scale_bytes,
            coded_summary.rotation_bytes,
            coded_summary.normals_bytes,
        ),
        format!("Compression: ratio={ratio:.4} reduction={reduction:.2}%"),
        format!("Validation: {}", config.validation.as_str()),
        format!("Runtime: {}", format_duration(total_runtime)),
    ];
    if let Some((path, size)) = decoded_ply {
        lines.push(format!("Decoded PLY: {} ({size} bytes)", path.display()));
    }
    if let Some(best) = sh_rest_best {
        lines.push(format!(
            "Best sh_rest: {} sh_rest_bytes={} total_bytes={} reduction={:.2}% rmse={} psnr={} ssim={}",
            best.name,
            best.sh_rest_bytes,
            best.total_bytes,
            best.total_reduction_percent,
            crate::metrics::format_metric(best.rmse),
            crate::metrics::format_metric(best.value_psnr_mean),
            crate::metrics::format_optional_metric(best.atlas_ssim_mean),
        ));
    }
    lines
}

pub fn coverage_from_scene(scene: &gaussian_parser::Scene) -> Vec<String> {
    let sh_rest_len = scene
        .gaussians
        .iter()
        .map(|g| g.sh_rest.len())
        .max()
        .unwrap_or(0);
    let normals_present = scene.gaussians.iter().any(|g| g.normals.is_some());
    let mut streams = vec![
        "xyz_x".to_string(),
        "xyz_y".to_string(),
        "xyz_z".to_string(),
    ];
    if normals_present {
        streams.extend(
            ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .map(String::from),
        );
    }
    streams.extend(
        ["sh_dc_r", "sh_dc_g", "sh_dc_b"]
            .into_iter()
            .map(String::from),
    );
    for i in 0..sh_rest_len {
        streams.push(format!("sh_rest_{i}"));
    }
    streams.extend(
        [
            "opacity", "scale_x", "scale_y", "scale_z", "rot_x", "rot_y", "rot_z", "rot_w",
        ]
        .into_iter()
        .map(String::from),
    );
    streams
}

pub fn coverage_from_packed(packed: &PackedRasterScene) -> Vec<String> {
    let sh_rest_len = packed.sh_rest_len as usize;
    let normals_present = packed.normals_present;
    let mut streams = vec![
        "xyz_x".to_string(),
        "xyz_y".to_string(),
        "xyz_z".to_string(),
    ];
    if normals_present {
        streams.extend(
            ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .map(String::from),
        );
    }
    streams.extend(
        ["sh_dc_r", "sh_dc_g", "sh_dc_b"]
            .into_iter()
            .map(String::from),
    );
    for i in 0..sh_rest_len {
        streams.push(format!("sh_rest_{i}"));
    }
    streams.extend(
        [
            "opacity", "scale_x", "scale_y", "scale_z", "rot_x", "rot_y", "rot_z", "rot_w",
        ]
        .into_iter()
        .map(String::from),
    );
    streams
}

pub fn stream_coverage_lines(
    mode: &str,
    packed_streams: &[String],
    compressed_streams: &[String],
    decoded_streams: &[String],
) -> Vec<String> {
    let missing_packed = missing_streams(packed_streams, packed_streams);
    let missing_compressed = missing_streams(packed_streams, compressed_streams);
    let missing_decoded = missing_streams(packed_streams, decoded_streams);
    vec![
        format!("Stream Coverage [{mode}]:"),
        format!(
            "- counts: schema={} packed={} compressed={} decoded={} reconstructed={}",
            packed_streams.len(),
            packed_streams.len(),
            compressed_streams.len(),
            decoded_streams.len(),
            decoded_streams.len()
        ),
        format!("- missing packed: {}", format_stream_list(&missing_packed)),
        format!(
            "- missing compressed: {}",
            format_stream_list(&missing_compressed)
        ),
        format!(
            "- missing decoded: {}",
            format_stream_list(&missing_decoded)
        ),
    ]
}

pub fn final_coded_summary_lines(
    coded_summary: &PackedContainerByteSummary,
    raw_payload_bytes: usize,
    original_sorted: &[gaussian_types::Gaussian],
    reconstructed: Option<&[gaussian_types::Gaussian]>,
    sh_rest_stats: &ShRestPayloadStats,
) -> Vec<String> {
    let raw_bytes = raw_payload_bytes as u64;
    let coded_bytes = coded_summary.total_coded_bytes;
    let compression_ratio = ratio_u64(raw_bytes, coded_bytes);
    let reduction_percent = reduction_percent_u64(raw_bytes, coded_bytes);
    let (sh_rest_rmse, sh_rest_max_abs) = reconstructed
        .map(|reconstructed| sh_rest_error_stats(original_sorted, reconstructed))
        .unwrap_or((f64::NAN, f64::NAN));
    let overhead_bytes = coded_summary.occupancy_bytes + coded_summary.metadata_bytes;

    vec![
        "Final Coded Summary:".to_string(),
        format!(
            "- bytes: geometry={} sh_dc={} sh_rest={} opacity={} scale={} rotation={} normals={} overhead={} total={} raw={}",
            coded_summary.geometry_bytes,
            coded_summary.sh_dc_bytes,
            coded_summary.sh_rest_bytes,
            coded_summary.opacity_bytes,
            coded_summary.scale_bytes,
            coded_summary.rotation_bytes,
            coded_summary.normals_bytes,
            overhead_bytes,
            coded_bytes,
            raw_bytes
        ),
        format!("- coding: ratio={compression_ratio:.4} reduction={reduction_percent:.2}%"),
        format!(
            "- sh_rest: rmse={sh_rest_rmse:.9} maxabs={sh_rest_max_abs:.9} coded_bytes={} storage={}",
            coded_summary.sh_rest_bytes, sh_rest_stats.storage
        ),
    ]
}

pub fn timing_summary_lines(
    config: &CliConfig,
    load_scene: Duration,
    sort_gaussians: Duration,
    pack_geometry: Duration,
    pack_sh_dc: Duration,
    pack_sh_rest: Duration,
    pack_sh_rest_quantization: Duration,
    pack_sh_rest_buffer_construction: Duration,
    pack_sh_rest_atlas_construction: Duration,
    pack_remaining_streams: Duration,
    write_container: Duration,
    write_report: &ContainerWriteTimings,
    unpack_roundtrip_validation: Duration,
    total_runtime: Duration,
) -> Vec<String> {
    let mut lines = vec![
        "Timing Summary:".to_string(),
        format!("- load scene: {}", format_duration(load_scene)),
        format!("- sort gaussians: {}", format_duration(sort_gaussians)),
        format!("- pack geometry: {}", format_duration(pack_geometry)),
        format!("- pack sh_dc: {}", format_duration(pack_sh_dc)),
        format!("- pack sh_rest: {}", format_duration(pack_sh_rest)),
        format!(
            "- pack sh_rest quantization: {}",
            format_duration(pack_sh_rest_quantization)
        ),
        format!(
            "- pack sh_rest buffer construction: {}",
            format_duration(pack_sh_rest_buffer_construction)
        ),
        format!(
            "- pack sh_rest atlas construction: {}",
            format_duration(pack_sh_rest_atlas_construction)
        ),
        format!(
            "- pack remaining streams: {}",
            format_duration(pack_remaining_streams)
        ),
        format!("- write container: {}", format_duration(write_container)),
        format!(
            "- write container assembly / metadata: {}",
            format_duration(write_report.container_assembly)
        ),
        format!(
            "- unpack / roundtrip validation: {}",
            format_duration(unpack_roundtrip_validation)
        ),
        format!("- total runtime: {}", format_duration(total_runtime)),
        format!("- run mode: {}", config.mode.as_str()),
        format!("- validation mode: {}", config.validation.as_str()),
        format!("- cache mode: {}", config.cache_mode.as_str()),
        format!("- log level: {}", config.log_level.as_str()),
    ];
    for group in &write_report.coded_groups {
        lines.push(format!(
            "- write container {} ffmpeg encode: {}",
            group.group_name,
            format_duration(group.ffmpeg_encode)
        ));
        lines.push(format!(
            "- write container {} temp file / file I/O: {}",
            group.group_name,
            format_duration(group.temp_file_io)
        ));
    }
    lines
}

pub fn verbose_summary_lines(
    config: &CliConfig,
    input_path: &str,
    output_dir: &str,
    log_path: Option<&Path>,
    packed: &PackedRasterScene,
    raw_payload_bytes: usize,
    original_ply_size: u64,
    quality_preset: QualityPreset,
    storage_config: PackedRasterStorageConfig,
    _sh_rest_codec_config: ShRestVideoAtlasCodecConfig,
    sh_rest_raw_bytes: u64,
    sh_rest_coded_bytes: u64,
    sh_rest_storage: &str,
    quick_validation: &[String],
    sh_rest_sweep_report: Option<&ShRestSweepReport>,
    final_summary_lines: &[String],
    timing_lines: &[String],
) -> Vec<String> {
    let mut lines = vec![
        format!("Input PLY: {input_path}"),
        format!("Output directory: {output_dir}"),
        format!("Run mode: {}", config.mode.as_str()),
        format!("Validation mode: {}", config.validation.as_str()),
        format!("Cache mode: {}", config.cache_mode.as_str()),
        format!("Log level: {}", config.log_level.as_str()),
        format!("Atlas dimensions: {} x {}", packed.width, packed.height),
        format!("Gaussians packed: {}", packed.num_gaussians),
        format!("Quality preset: {}", quality_preset.as_str()),
        format!(
            "Storage modes: geometry={} sh_dc={} sh_rest={}",
            storage_config.geometry.as_str(),
            storage_config.sh_dc.as_str(),
            storage_config.sh_rest.as_str()
        ),
        format!("Total raw payload bytes: {}", raw_payload_bytes),
        format!(
            "SH rest payload: raw_bytes={} coded_bytes={} storage={}",
            sh_rest_raw_bytes, sh_rest_coded_bytes, sh_rest_storage
        ),
        format!("Original PLY size: {}", original_ply_size),
    ];
    if let Some(log_path) = log_path {
        lines.push(format!("Detailed log: {}", log_path.display()));
    }
    lines.extend(quick_validation.iter().cloned());
    if let Some(best) = sh_rest_sweep_report.and_then(|report| report.best.as_ref()) {
        lines.push(format!(
            "Best sh_rest config: {} total_bytes={} total_reduction={:.2}% sh_rest_bytes={}",
            best.name, best.total_bytes, best.total_reduction_percent, best.sh_rest_bytes
        ));
    }
    lines.extend(final_summary_lines.iter().cloned());
    lines.extend(timing_lines.iter().cloned());
    lines
}

pub fn sh_rest_error_stats(
    original_sorted: &[gaussian_types::Gaussian],
    reconstructed: &[gaussian_types::Gaussian],
) -> (f64, f64) {
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    let mut max_abs = 0.0f64;

    for (original, recon) in original_sorted.iter().zip(reconstructed.iter()) {
        for (original_value, recon_value) in original.sh_rest.iter().zip(recon.sh_rest.iter()) {
            let err = *recon_value as f64 - *original_value as f64;
            sum_sq += err * err;
            count += 1;
            max_abs = max_abs.max(err.abs());
        }
    }

    let rmse = if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    };
    (rmse, max_abs)
}

pub fn ratio_u64(raw_bytes: u64, coded_bytes: u64) -> f64 {
    if coded_bytes == 0 {
        0.0
    } else {
        raw_bytes as f64 / coded_bytes as f64
    }
}

pub fn reduction_percent_u64(raw_bytes: u64, coded_bytes: u64) -> f64 {
    if raw_bytes == 0 {
        0.0
    } else {
        100.0 * (1.0 - coded_bytes as f64 / raw_bytes as f64)
    }
}

pub fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

fn missing_streams(schema_streams: &[String], covered_streams: &[String]) -> Vec<String> {
    let covered: BTreeSet<&str> = covered_streams.iter().map(|name| name.as_str()).collect();
    schema_streams
        .iter()
        .filter(|name| !covered.contains(name.as_str()))
        .cloned()
        .collect()
}

fn format_stream_list(streams: &[String]) -> String {
    if streams.is_empty() {
        "none".to_string()
    } else {
        streams.join(", ")
    }
}
