use std::collections::BTreeMap;

use gaussian_packing::{
    AuxiliaryCodecMetadata, NormalsStrategy, PackedContainerByteSummary, PackedRasterScene,
    RotationStrategy, ShRestPayloadStats,
};
use gaussian_types::Gaussian;

#[derive(Debug, Clone)]
pub struct ValueMetrics {
    pub rmse: f64,
    pub max_abs: f64,
    pub psnr: f64,
}

#[derive(Debug, Clone)]
pub struct AtlasMetrics {
    pub psnr: f64,
    pub ssim: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub group: &'static str,
    pub name: String,
    pub value: ValueMetrics,
    pub atlas: Option<AtlasMetrics>,
    pub compression: CompressionStats,
}

#[derive(Debug, Clone)]
pub struct GroupMetricsSummary {
    pub name: &'static str,
    pub stream_count: usize,
    pub coded_bytes: u64,
    pub raw_bytes: u64,
    pub compression_ratio: f64,
    pub reduction_percent: f64,
    pub rmse_mean: f64,
    pub rmse_max: f64,
    pub max_abs_max: f64,
    pub value_psnr_mean: f64,
    pub value_psnr_min: f64,
    pub value_psnr_max: f64,
    pub atlas_psnr_mean: Option<f64>,
    pub atlas_psnr_min: Option<f64>,
    pub atlas_psnr_max: Option<f64>,
    pub atlas_ssim_mean: Option<f64>,
    pub atlas_ssim_min: Option<f64>,
    pub atlas_ssim_max: Option<f64>,
    pub atlas_ssim_count: usize,
}

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub raw_bytes: u64,
    pub coded_bytes: u64,
    pub compression_ratio: f64,
    pub reduction_percent: f64,
    pub attribution: &'static str,
}

#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub streams: Vec<StreamMetrics>,
    pub groups: Vec<GroupMetricsSummary>,
}

pub fn evaluate_metrics(
    original_sorted: &[Gaussian],
    reconstructed: &[Gaussian],
    original_packed: &PackedRasterScene,
    decoded_packed: &PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
    sh_rest_stats: &ShRestPayloadStats,
) -> MetricsReport {
    let mut streams = collect_value_metrics(original_sorted, reconstructed);
    let compression = per_stream_compression_stats(
        original_packed,
        coded_summary,
        sh_rest_stats,
        decoded_packed.auxiliary_codec_metadata.as_ref(),
    );
    let atlas_metrics = collect_atlas_metrics(original_packed, decoded_packed);
    for stream in &mut streams {
        stream.atlas = atlas_metrics
            .get(&stream_key(stream.group, &stream.name))
            .cloned();
        stream.compression = compression
            .get(&stream_key(stream.group, &stream.name))
            .cloned()
            .unwrap_or_else(zero_compression_stats);
    }

    let groups = summarize_groups(&streams, original_packed, coded_summary, sh_rest_stats);
    MetricsReport { streams, groups }
}

pub fn group_metric_lines(groups: &[GroupMetricsSummary]) -> Vec<String> {
    let mut lines = vec!["Per-Group Quality:".to_string()];
    for group in groups {
        lines.push(format!(
            "- {}: streams={} coded_bytes={} raw_bytes={} ratio={} reduction={}% rmse_mean={} rmse_max={} maxabs_max={} value_psnr_mean={} value_psnr_min={} value_psnr_max={} atlas_psnr_mean={} atlas_psnr_min={} atlas_psnr_max={} atlas_ssim_mean={} atlas_ssim_min={} atlas_ssim_max={} atlas_ssim_count={}",
            group.name,
            group.stream_count,
            group.coded_bytes,
            group.raw_bytes,
            format_metric(group.compression_ratio),
            format_metric(group.reduction_percent),
            format_metric(group.rmse_mean),
            format_metric(group.rmse_max),
            format_metric(group.max_abs_max),
            format_metric(group.value_psnr_mean),
            format_metric(group.value_psnr_min),
            format_metric(group.value_psnr_max),
            format_optional_metric(group.atlas_psnr_mean),
            format_optional_metric(group.atlas_psnr_min),
            format_optional_metric(group.atlas_psnr_max),
            format_optional_metric(group.atlas_ssim_mean),
            format_optional_metric(group.atlas_ssim_min),
            format_optional_metric(group.atlas_ssim_max),
            group.atlas_ssim_count,
        ));
    }
    lines
}

pub fn stream_metric_lines(streams: &[StreamMetrics]) -> Vec<String> {
    let mut lines = vec![
        "Per-Stream Quality:".to_string(),
        "- note: transformed semantic streams may report quality without exact storage attribution; see Auxiliary Group Diagnostics for exact representation-stream bytes.".to_string(),
    ];
    for stream in streams {
        let (atlas_psnr, atlas_ssim) = match &stream.atlas {
            Some(atlas) => (
                format_metric(atlas.psnr),
                format_optional_metric(atlas.ssim),
            ),
            None => ("n/a".to_string(), "n/a".to_string()),
        };
        lines.push(format!(
            "- {} / {}: raw_bytes={} coded_bytes={} ratio={} reduction={}% attribution={} rmse={} maxabs={} value_psnr={} atlas_psnr={} atlas_ssim={}",
            stream.group,
            stream.name,
            stream.compression.raw_bytes,
            stream.compression.coded_bytes,
            format_metric(stream.compression.compression_ratio),
            format_metric(stream.compression.reduction_percent),
            stream.compression.attribution,
            format_metric(stream.value.rmse),
            format_metric(stream.value.max_abs),
            format_metric(stream.value.psnr),
            atlas_psnr,
            atlas_ssim,
        ));
    }
    lines
}

pub fn format_metric(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else if value.is_infinite() && value.is_sign_positive() {
        "inf".to_string()
    } else if value.is_infinite() {
        "-inf".to_string()
    } else {
        "n/a".to_string()
    }
}

pub fn format_optional_metric(value: Option<f64>) -> String {
    value
        .map(format_metric)
        .unwrap_or_else(|| "n/a".to_string())
}

fn collect_value_metrics(
    original_sorted: &[Gaussian],
    reconstructed: &[Gaussian],
) -> Vec<StreamMetrics> {
    let mut streams = Vec::new();

    push_component_streams(
        &mut streams,
        "geometry",
        ["xyz_x", "xyz_y", "xyz_z"],
        original_sorted,
        reconstructed,
        |g| g.xyz,
    );
    push_component_streams(
        &mut streams,
        "sh_dc",
        ["sh_dc_r", "sh_dc_g", "sh_dc_b"],
        original_sorted,
        reconstructed,
        |g| g.sh_dc,
    );

    if original_sorted.iter().any(|g| g.normals.is_some()) {
        push_component_streams(
            &mut streams,
            "normals",
            ["normals_x", "normals_y", "normals_z"],
            original_sorted,
            reconstructed,
            |g| g.normals.unwrap_or([0.0; 3]),
        );
    }

    let sh_rest_len = original_sorted
        .iter()
        .map(|g| g.sh_rest.len())
        .max()
        .unwrap_or(0);
    for stream_idx in 0..sh_rest_len {
        push_scalar_stream(
            &mut streams,
            "sh_rest",
            &format!("sh_rest_{stream_idx}"),
            original_sorted
                .iter()
                .map(|g| g.sh_rest[stream_idx])
                .collect(),
            reconstructed
                .iter()
                .map(|g| g.sh_rest[stream_idx])
                .collect(),
        );
    }

    push_scalar_stream(
        &mut streams,
        "opacity",
        "opacity",
        original_sorted.iter().map(|g| g.opacity).collect(),
        reconstructed.iter().map(|g| g.opacity).collect(),
    );
    push_component_streams(
        &mut streams,
        "scale",
        ["scale_x", "scale_y", "scale_z"],
        original_sorted,
        reconstructed,
        |g| g.scale,
    );
    push_component_streams(
        &mut streams,
        "rotation",
        ["rot_x", "rot_y", "rot_z", "rot_w"],
        original_sorted,
        reconstructed,
        |g| g.rot,
    );

    streams
}

fn push_component_streams<const N: usize, F>(
    streams: &mut Vec<StreamMetrics>,
    group: &'static str,
    names: [&str; N],
    original_sorted: &[Gaussian],
    reconstructed: &[Gaussian],
    extract: F,
) where
    F: Fn(&Gaussian) -> [f32; N],
{
    for component_idx in 0..N {
        let original_values = original_sorted
            .iter()
            .map(|g| extract(g)[component_idx])
            .collect::<Vec<_>>();
        let reconstructed_values = reconstructed
            .iter()
            .map(|g| extract(g)[component_idx])
            .collect::<Vec<_>>();
        push_scalar_stream(
            streams,
            group,
            names[component_idx],
            original_values,
            reconstructed_values,
        );
    }
}

fn push_scalar_stream(
    streams: &mut Vec<StreamMetrics>,
    group: &'static str,
    name: &str,
    original_values: Vec<f32>,
    reconstructed_values: Vec<f32>,
) {
    streams.push(StreamMetrics {
        group,
        name: name.to_string(),
        value: compute_value_metrics(&original_values, &reconstructed_values),
        atlas: None,
        compression: zero_compression_stats(),
    });
}

fn compute_value_metrics(original: &[f32], reconstructed: &[f32]) -> ValueMetrics {
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for (&original_value, &reconstructed_value) in original.iter().zip(reconstructed.iter()) {
        let original_value = original_value as f64;
        let reconstructed_value = reconstructed_value as f64;
        let err = reconstructed_value - original_value;
        sum_sq += err * err;
        max_abs = max_abs.max(err.abs());
        min_value = min_value.min(original_value);
        max_value = max_value.max(original_value);
    }

    let rmse = if original.is_empty() {
        0.0
    } else {
        (sum_sq / original.len() as f64).sqrt()
    };
    let psnr = psnr_from_rmse(rmse, min_value, max_value);
    ValueMetrics {
        rmse,
        max_abs,
        psnr,
    }
}

fn collect_atlas_metrics(
    original_packed: &PackedRasterScene,
    decoded_packed: &PackedRasterScene,
) -> BTreeMap<String, AtlasMetrics> {
    let mut atlas_metrics = BTreeMap::new();

    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "geometry",
        "xyz_x",
        &original_packed.geom_x,
        &decoded_packed.geom_x,
    );
    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "geometry",
        "xyz_y",
        &original_packed.geom_y,
        &decoded_packed.geom_y,
    );
    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "geometry",
        "xyz_z",
        &original_packed.geom_z,
        &decoded_packed.geom_z,
    );
    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "sh_dc",
        "sh_dc_r",
        &original_packed.sh_dc_r,
        &decoded_packed.sh_dc_r,
    );
    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "sh_dc",
        "sh_dc_g",
        &original_packed.sh_dc_g,
        &decoded_packed.sh_dc_g,
    );
    push_atlas_plane_metrics(
        &mut atlas_metrics,
        "sh_dc",
        "sh_dc_b",
        &original_packed.sh_dc_b,
        &decoded_packed.sh_dc_b,
    );

    for stream_idx in 0..original_packed.sh_rest_len as usize {
        let original_plane = sh_rest_plane(original_packed, stream_idx);
        let decoded_plane = sh_rest_plane(decoded_packed, stream_idx);
        push_atlas_plane_metrics(
            &mut atlas_metrics,
            "sh_rest",
            &format!("sh_rest_{stream_idx}"),
            &original_plane,
            &decoded_plane,
        );
    }

    atlas_metrics
}

fn push_atlas_plane_metrics(
    atlas_metrics: &mut BTreeMap<String, AtlasMetrics>,
    group: &'static str,
    name: &str,
    original_plane: &[u16],
    decoded_plane: &[u16],
) {
    let (psnr, ssim) = compute_atlas_metrics(original_plane, decoded_plane);
    atlas_metrics.insert(stream_key(group, name), AtlasMetrics { psnr, ssim });
}

fn sh_rest_plane(packed: &PackedRasterScene, stream_idx: usize) -> Vec<u16> {
    let plane_len = (packed.width as usize) * (packed.height as usize);
    let gaussian_len = packed.num_gaussians as usize;
    let mut plane = vec![0u16; plane_len];
    for gaussian_idx in 0..gaussian_len {
        plane[gaussian_idx] = sh_rest_value_at(
            &packed.sh_rest_payload,
            stream_idx,
            gaussian_idx,
            gaussian_len,
        );
    }
    plane
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

fn compute_atlas_metrics(original: &[u16], decoded: &[u16]) -> (f64, Option<f64>) {
    let mut sum_sq = 0.0f64;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut original_sum = 0.0f64;
    let mut decoded_sum = 0.0f64;

    for (&original_value, &decoded_value) in original.iter().zip(decoded.iter()) {
        let original_value = original_value as f64;
        let decoded_value = decoded_value as f64;
        let err = decoded_value - original_value;
        sum_sq += err * err;
        min_value = min_value.min(original_value);
        max_value = max_value.max(original_value);
        original_sum += original_value;
        decoded_sum += decoded_value;
    }

    let rmse = if original.is_empty() {
        0.0
    } else {
        (sum_sq / original.len() as f64).sqrt()
    };
    let psnr = psnr_from_rmse(rmse, min_value, max_value);
    let ssim = ssim_u16(original, decoded, original_sum, decoded_sum);
    (psnr, ssim)
}

fn ssim_u16(original: &[u16], decoded: &[u16], original_sum: f64, decoded_sum: f64) -> Option<f64> {
    if original.len() != decoded.len() || original.is_empty() {
        return None;
    }

    let len = original.len() as f64;
    let mean_original = original_sum / len;
    let mean_decoded = decoded_sum / len;

    let mut var_original = 0.0f64;
    let mut var_decoded = 0.0f64;
    let mut covariance = 0.0f64;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for (&original_value, &decoded_value) in original.iter().zip(decoded.iter()) {
        let original_value = original_value as f64;
        let decoded_value = decoded_value as f64;
        let centered_original = original_value - mean_original;
        let centered_decoded = decoded_value - mean_decoded;
        var_original += centered_original * centered_original;
        var_decoded += centered_decoded * centered_decoded;
        covariance += centered_original * centered_decoded;
        min_value = min_value.min(original_value.min(decoded_value));
        max_value = max_value.max(original_value.max(decoded_value));
    }

    let denom = (len - 1.0).max(1.0);
    var_original /= denom;
    var_decoded /= denom;
    covariance /= denom;

    let dynamic_range = max_value - min_value;
    if dynamic_range == 0.0 {
        return Some(if original == decoded { 1.0 } else { 0.0 });
    }

    let c1 = (0.01 * dynamic_range).powi(2);
    let c2 = (0.03 * dynamic_range).powi(2);
    let numerator = (2.0 * mean_original * mean_decoded + c1) * (2.0 * covariance + c2);
    let denominator =
        (mean_original.powi(2) + mean_decoded.powi(2) + c1) * (var_original + var_decoded + c2);
    if denominator == 0.0 {
        Some(1.0)
    } else {
        Some(numerator / denominator)
    }
}

fn summarize_groups(
    streams: &[StreamMetrics],
    packed: &PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
    sh_rest_stats: &ShRestPayloadStats,
) -> Vec<GroupMetricsSummary> {
    let groups = group_byte_stats(packed, coded_summary, sh_rest_stats);
    let mut summaries = Vec::new();
    for (name, coded_bytes, raw_bytes) in groups {
        let members: Vec<&StreamMetrics> = streams
            .iter()
            .filter(|stream| stream.group == name)
            .collect();
        if members.is_empty() {
            continue;
        }
        let stream_count = members.len();
        let rmse_mean =
            members.iter().map(|stream| stream.value.rmse).sum::<f64>() / stream_count as f64;
        let rmse_max = members
            .iter()
            .map(|stream| stream.value.rmse)
            .fold(0.0, f64::max);
        let max_abs_max = members
            .iter()
            .map(|stream| stream.value.max_abs)
            .fold(0.0, f64::max);
        let value_psnr_mean =
            members.iter().map(|stream| stream.value.psnr).sum::<f64>() / stream_count as f64;
        let value_psnr_min = members
            .iter()
            .map(|stream| stream.value.psnr)
            .fold(f64::INFINITY, f64::min);
        let value_psnr_max = members
            .iter()
            .map(|stream| stream.value.psnr)
            .fold(f64::NEG_INFINITY, f64::max);

        let atlas_members: Vec<&AtlasMetrics> = members
            .iter()
            .filter_map(|stream| stream.atlas.as_ref())
            .collect();
        let atlas_psnr_mean = mean_optional(atlas_members.iter().map(|atlas| atlas.psnr).collect());
        let atlas_psnr_min = min_optional(atlas_members.iter().map(|atlas| atlas.psnr).collect());
        let atlas_psnr_max = max_optional(atlas_members.iter().map(|atlas| atlas.psnr).collect());
        let atlas_ssim_values: Vec<f64> = atlas_members
            .iter()
            .filter_map(|atlas| atlas.ssim)
            .collect();
        let atlas_ssim_mean = mean_optional(atlas_ssim_values.clone());
        let atlas_ssim_min = min_optional(atlas_ssim_values.clone());
        let atlas_ssim_max = max_optional(atlas_ssim_values.clone());

        summaries.push(GroupMetricsSummary {
            name,
            stream_count,
            coded_bytes,
            raw_bytes,
            compression_ratio: ratio(raw_bytes, coded_bytes),
            reduction_percent: reduction_percent(raw_bytes, coded_bytes),
            rmse_mean,
            rmse_max,
            max_abs_max,
            value_psnr_mean,
            value_psnr_min,
            value_psnr_max,
            atlas_psnr_mean,
            atlas_psnr_min,
            atlas_psnr_max,
            atlas_ssim_mean,
            atlas_ssim_min,
            atlas_ssim_max,
            atlas_ssim_count: atlas_ssim_values.len(),
        });
    }
    summaries
}

fn per_stream_compression_stats(
    packed: &PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
    sh_rest_stats: &ShRestPayloadStats,
    auxiliary: Option<&AuxiliaryCodecMetadata>,
) -> BTreeMap<String, CompressionStats> {
    let mut stats = BTreeMap::new();
    insert_even_split_group(
        &mut stats,
        "geometry",
        &["xyz_x", "xyz_y", "xyz_z"],
        (packed.geom_x.len() * std::mem::size_of::<u16>()) as u64,
        coded_summary.geometry_bytes,
    );
    insert_even_split_group(
        &mut stats,
        "sh_dc",
        &["sh_dc_r", "sh_dc_g", "sh_dc_b"],
        (packed.sh_dc_r.len() * std::mem::size_of::<u16>()) as u64,
        coded_summary.sh_dc_bytes,
    );
    let sh_rest_stream_count = packed.sh_rest_len as usize;
    if sh_rest_stream_count > 0 {
        let names = (0..sh_rest_stream_count)
            .map(|idx| format!("sh_rest_{idx}"))
            .collect::<Vec<_>>();
        let refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
        insert_even_split_group(
            &mut stats,
            "sh_rest",
            &refs,
            (packed.num_gaussians as usize * std::mem::size_of::<f32>()) as u64,
            coded_summary.sh_rest_bytes,
        );
    }
    insert_auxiliary_group_stats(&mut stats, auxiliary, packed, coded_summary);
    let _ = sh_rest_stats;
    stats
}

fn insert_auxiliary_group_stats(
    stats: &mut BTreeMap<String, CompressionStats>,
    auxiliary: Option<&AuxiliaryCodecMetadata>,
    packed: &PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
) {
    let Some(auxiliary) = auxiliary else {
        insert_even_split_group(
            stats,
            "opacity",
            &["opacity"],
            (packed.opacity.len() * std::mem::size_of::<f32>()) as u64,
            coded_summary.opacity_bytes,
        );
        insert_even_split_group(
            stats,
            "scale",
            &["scale_x", "scale_y", "scale_z"],
            (packed.scale_x.len() * std::mem::size_of::<f32>()) as u64,
            coded_summary.scale_bytes,
        );
        insert_even_split_group(
            stats,
            "rotation",
            &["rot_x", "rot_y", "rot_z", "rot_w"],
            (packed.rot_x.len() * std::mem::size_of::<f32>()) as u64,
            coded_summary.rotation_bytes,
        );
        if !packed.normals_x.is_empty() {
            insert_even_split_group(
                stats,
                "normals",
                &["normals_x", "normals_y", "normals_z"],
                (packed.normals_x.len() * std::mem::size_of::<f32>()) as u64,
                coded_summary.normals_bytes,
            );
        }
        return;
    };

    for stream in &auxiliary.opacity.payload_streams {
        if stream.name == "opacity" {
            insert_exact_stream(
                stats,
                "opacity",
                "opacity",
                auxiliary.opacity.raw_stream_bytes,
                stream.coded_bytes,
                "exact_representation_stream",
            );
        }
    }
    for (idx, name) in ["scale_x", "scale_y", "scale_z"].into_iter().enumerate() {
        let coded_bytes = auxiliary
            .scale
            .payload_streams
            .iter()
            .find(|stream| stream.name == name)
            .map(|stream| stream.coded_bytes)
            .unwrap_or(0);
        insert_exact_stream(
            stats,
            "scale",
            name,
            auxiliary.scale.raw_stream_bytes[idx],
            coded_bytes,
            "exact_representation_stream",
        );
    }
    match auxiliary.rotation.strategy {
        RotationStrategy::RawF32 => {
            for (idx, name) in ["rot_x", "rot_y", "rot_z", "rot_w"].into_iter().enumerate() {
                let coded_bytes = auxiliary
                    .rotation
                    .payload_streams
                    .iter()
                    .find(|stream| stream.name == name)
                    .map(|stream| stream.coded_bytes)
                    .unwrap_or(0);
                insert_exact_stream(
                    stats,
                    "rotation",
                    name,
                    auxiliary.rotation.raw_stream_bytes[idx],
                    coded_bytes,
                    "exact_representation_stream",
                );
            }
        }
        _ => {
            for (idx, name) in ["rot_x", "rot_y", "rot_z", "rot_w"].into_iter().enumerate() {
                insert_exact_stream(
                    stats,
                    "rotation",
                    name,
                    auxiliary.rotation.raw_stream_bytes[idx],
                    0,
                    "semantic_quality_only_transformed_storage",
                );
            }
        }
    }
    match auxiliary.normals.strategy {
        NormalsStrategy::RawF32 => {
            for (idx, name) in ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .enumerate()
            {
                let coded_bytes = auxiliary
                    .normals
                    .payload_streams
                    .iter()
                    .find(|stream| stream.name == name)
                    .map(|stream| stream.coded_bytes)
                    .unwrap_or(0);
                insert_exact_stream(
                    stats,
                    "normals",
                    name,
                    auxiliary.normals.raw_stream_bytes[idx],
                    coded_bytes,
                    "exact_representation_stream",
                );
            }
        }
        NormalsStrategy::Omitted => {
            for (idx, name) in ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .enumerate()
            {
                insert_exact_stream(
                    stats,
                    "normals",
                    name,
                    auxiliary
                        .normals
                        .raw_stream_bytes
                        .get(idx)
                        .copied()
                        .unwrap_or(0),
                    0,
                    "omitted_no_semantic_storage",
                );
            }
        }
        _ => {
            for (idx, name) in ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .enumerate()
            {
                insert_exact_stream(
                    stats,
                    "normals",
                    name,
                    auxiliary.normals.raw_stream_bytes[idx],
                    0,
                    "semantic_quality_only_transformed_storage",
                );
            }
        }
    }
}

fn insert_exact_stream(
    stats: &mut BTreeMap<String, CompressionStats>,
    group: &'static str,
    name: &str,
    raw_bytes: u64,
    coded_bytes: u64,
    attribution: &'static str,
) {
    stats.insert(
        stream_key(group, name),
        CompressionStats {
            raw_bytes,
            coded_bytes,
            compression_ratio: ratio(raw_bytes, coded_bytes),
            reduction_percent: reduction_percent(raw_bytes, coded_bytes),
            attribution,
        },
    );
}

fn insert_even_split_group(
    stats: &mut BTreeMap<String, CompressionStats>,
    group: &'static str,
    stream_names: &[impl AsRef<str>],
    raw_bytes_per_stream: u64,
    total_coded_bytes: u64,
) {
    if stream_names.is_empty() {
        return;
    }
    let stream_count = stream_names.len() as u64;
    let base_coded = total_coded_bytes / stream_count;
    let remainder = total_coded_bytes % stream_count;
    for (idx, name) in stream_names.iter().enumerate() {
        let coded_bytes = base_coded + u64::from((idx as u64) < remainder);
        stats.insert(
            stream_key(group, name.as_ref()),
            CompressionStats {
                raw_bytes: raw_bytes_per_stream,
                coded_bytes,
                compression_ratio: ratio(raw_bytes_per_stream, coded_bytes),
                reduction_percent: reduction_percent(raw_bytes_per_stream, coded_bytes),
                attribution: "shared_group_even_split",
            },
        );
    }
}

fn zero_compression_stats() -> CompressionStats {
    CompressionStats {
        raw_bytes: 0,
        coded_bytes: 0,
        compression_ratio: 0.0,
        reduction_percent: 0.0,
        attribution: "n/a",
    }
}

fn ratio(raw_bytes: u64, coded_bytes: u64) -> f64 {
    if coded_bytes == 0 {
        0.0
    } else {
        raw_bytes as f64 / coded_bytes as f64
    }
}

fn reduction_percent(raw_bytes: u64, coded_bytes: u64) -> f64 {
    if raw_bytes == 0 {
        0.0
    } else {
        100.0 * (1.0 - coded_bytes as f64 / raw_bytes as f64)
    }
}

fn group_byte_stats(
    packed: &PackedRasterScene,
    coded_summary: &PackedContainerByteSummary,
    sh_rest_stats: &ShRestPayloadStats,
) -> [(&'static str, u64, u64); 7] {
    [
        (
            "geometry",
            coded_summary.geometry_bytes,
            ((packed.geom_x.len() + packed.geom_y.len() + packed.geom_z.len())
                * std::mem::size_of::<u16>()) as u64,
        ),
        (
            "sh_dc",
            coded_summary.sh_dc_bytes,
            ((packed.sh_dc_r.len() + packed.sh_dc_g.len() + packed.sh_dc_b.len())
                * std::mem::size_of::<u16>()) as u64,
        ),
        (
            "sh_rest",
            coded_summary.sh_rest_bytes,
            sh_rest_stats.raw_bytes as u64,
        ),
        (
            "opacity",
            coded_summary.opacity_bytes,
            (packed.opacity.len() * std::mem::size_of::<f32>()) as u64,
        ),
        (
            "scale",
            coded_summary.scale_bytes,
            ((packed.scale_x.len() + packed.scale_y.len() + packed.scale_z.len())
                * std::mem::size_of::<f32>()) as u64,
        ),
        (
            "rotation",
            coded_summary.rotation_bytes,
            ((packed.rot_x.len() + packed.rot_y.len() + packed.rot_z.len() + packed.rot_w.len())
                * std::mem::size_of::<f32>()) as u64,
        ),
        (
            "normals",
            coded_summary.normals_bytes,
            ((packed.normals_x.len() + packed.normals_y.len() + packed.normals_z.len())
                * std::mem::size_of::<f32>()) as u64,
        ),
    ]
}

fn psnr_from_rmse(rmse: f64, min_value: f64, max_value: f64) -> f64 {
    let range = if min_value.is_finite() && max_value.is_finite() {
        max_value - min_value
    } else {
        0.0
    };
    if rmse == 0.0 {
        f64::INFINITY
    } else if range <= 0.0 {
        0.0
    } else {
        20.0 * (range / rmse).log10()
    }
}

fn mean_optional(values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn min_optional(values: Vec<f64>) -> Option<f64> {
    values.into_iter().reduce(f64::min)
}

fn max_optional(values: Vec<f64>) -> Option<f64> {
    values.into_iter().reduce(f64::max)
}

fn stream_key(group: &str, name: &str) -> String {
    format!("{group}:{name}")
}
