//! eparser: Kerbl-style 3DGS PLY parser (binary LE + ASCII), no deps.
//! Returns an array of `Gaussian`, each with its own fields.

use gaussian_types::Gaussian;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};

/// Full scene: array of Gaussians + some metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub gaussians: Vec<Gaussian>,
    pub meta: HashMap<String, String>,
    pub mins: [f32; 3],
    pub maxes: [f32; 3],
}

/// Errors for PLY parsing. All message-carrying variants own their String to avoid `'static` lifetimes.
#[derive(Debug)]
pub enum PlyError {
    Io(io::Error),
    Header(String),
    Eof,
    MissingProp(String),
    UnsupportedFormat(String),
    BadNumber,
}

impl Display for PlyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PlyError::Io(e) => write!(f, "io error: {e}"),
            PlyError::Header(s) => write!(f, "invalid/unsupported PLY header: {s}"),
            PlyError::Eof => write!(f, "unexpected EOF while reading vertex data"),
            PlyError::MissingProp(s) => write!(f, "required property missing: {s}"),
            PlyError::UnsupportedFormat(s) => write!(f, "unsupported PLY format: {s}"),
            PlyError::BadNumber => write!(f, "invalid numeric field"),
        }
    }
}
impl StdError for PlyError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        if let PlyError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<io::Error> for PlyError {
    fn from(e: io::Error) -> Self {
        PlyError::Io(e)
    }
}

// Small helpers to build string-carrying errors concisely
fn header<S: Into<String>>(s: S) -> PlyError {
    PlyError::Header(s.into())
}
fn missing<S: Into<String>>(s: S) -> PlyError {
    PlyError::MissingProp(s.into())
}

#[derive(Debug)]
struct PlyMeta {
    format: PlyFormat,
    vertex_count: usize,
    properties: Vec<(String, String)>, // (name, dtype)
    header_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    BinaryLittleEndian,
    Ascii,
}

impl PlyFormat {
    fn from_header_line(line: &str) -> Result<Self, PlyError> {
        if line.contains("binary_little_endian") {
            Ok(Self::BinaryLittleEndian)
        } else if line.contains("ascii") {
            Ok(Self::Ascii)
        } else {
            Err(PlyError::UnsupportedFormat(line.to_string()))
        }
    }
}

/// Load a Kerbl-style 3DGS PLY file, returning an array of `Gaussian`s.
pub fn load_gaussians_from_ply(path: &str) -> Result<Scene, PlyError> {
    let file = File::open(path)?;
    let mut br = BufReader::new(file);
    let meta = parse_header(&mut br)?;

    // Read vertex rows as Vec<Vec<f32>> aligned with meta.properties.
    let rows: Vec<Vec<f32>> = match meta.format {
        PlyFormat::BinaryLittleEndian => {
            let mut f = br.into_inner();
            read_vertices_binary(&mut f, &meta)?
        }
        PlyFormat::Ascii => read_vertices_ascii(&mut br, &meta)?,
    };
    let names: Vec<String> = meta.properties.iter().map(|(n, _)| n.clone()).collect();

    // Indices for required/optional fields (by name).
    let i_x = idx(&names, "x")?;
    let i_y = idx(&names, "y")?;
    let i_z = idx(&names, "z")?;

    let i_opacity = idx_any(&names, &["opacity", "alpha"])?;

    // normals optional
    let i_nx = names.iter().position(|n| n == "nx");
    let i_ny = names.iter().position(|n| n == "ny");
    let i_nz = names.iter().position(|n| n == "nz");
    let normals_present = i_nx.is_some() && i_ny.is_some() && i_nz.is_some();

    // scale_0..2 (accept "scale_0" or "scale0")
    let [i_s0, i_s1, i_s2] = prefix_triple(&names, &["scale_", "scale", "s", "sc"])?;

    // quaternion: rot_0..3 or qx,qy,qz,qw
    let (i_qx, i_qy, i_qz, i_qw) = if ["rot_0", "rot_1", "rot_2", "rot_3"]
        .iter()
        .all(|k| names.contains(&k.to_string()))
    {
        (
            idx(&names, "rot_0")?,
            idx(&names, "rot_1")?,
            idx(&names, "rot_2")?,
            idx(&names, "rot_3")?,
        )
    } else if ["qx", "qy", "qz", "qw"]
        .iter()
        .all(|k| names.contains(&k.to_string()))
    {
        (
            idx(&names, "qx")?,
            idx(&names, "qy")?,
            idx(&names, "qz")?,
            idx(&names, "qw")?,
        )
    } else {
        return Err(missing("rot quaternion"));
    };

    // SH DC
    let i_dc0 = idx_any(&names, &["f_dc_0", "sh_dc_0", "dc_0"])?;
    let i_dc1 = idx_any(&names, &["f_dc_1", "sh_dc_1", "dc_1"])?;
    let i_dc2 = idx_any(&names, &["f_dc_2", "sh_dc_2", "dc_2"])?;

    // SH rest indices (prefer f_rest_*)
    let mut rest_cols: Vec<(usize, usize)> = Vec::new(); // (col_index, order)
    for (i, n) in names.iter().enumerate() {
        if let Some(sfx) = n
            .strip_prefix("f_rest_")
            .or_else(|| n.strip_prefix("sh_rest_"))
            .or_else(|| n.strip_prefix("rest_"))
        {
            if let Ok(k) = sfx.parse::<usize>() {
                rest_cols.push((i, k));
            }
        }
    }
    rest_cols.sort_by_key(|&(_, k)| k);

    // Fallback: f_0.. (first 3 are DC)
    let fallback_rest: Option<Vec<usize>> = if rest_cols.is_empty() {
        let mut fcols: Vec<(usize, usize)> = Vec::new();
        for (i, n) in names.iter().enumerate() {
            if let Some(k) = n
                .strip_prefix("f_")
                .or_else(|| n.strip_prefix("sh_"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                fcols.push((i, k));
            }
        }
        fcols.sort_by_key(|&(_, k)| k);
        if fcols.len() >= 3 {
            Some(
                fcols
                    .into_iter()
                    .filter(|&(_, k)| k >= 3)
                    .map(|(i, _)| i)
                    .collect(),
            )
        } else {
            None
        }
    } else {
        None
    };

    // Build array-of-structs

    //find mins
    let mut mins: [f32; 3] = [0f32; 3];
    let mut maxes: [f32; 3] = [0f32; 3];

    //initialize mins
    if let Some(init_row) = rows.first() {
        mins = [init_row[i_x], init_row[i_y], init_row[i_z]];
        maxes = mins;
    }

    let mut gaussians = Vec::with_capacity(rows.len());
    for r in &rows {
        let xyz = [r[i_x], r[i_y], r[i_z]];
        let normals = if normals_present {
            Some([r[i_nx.unwrap()], r[i_ny.unwrap()], r[i_nz.unwrap()]])
        } else {
            None
        };
        let sh_dc = [r[i_dc0], r[i_dc1], r[i_dc2]];

        let sh_rest: Vec<f32> = if !rest_cols.is_empty() {
            rest_cols.iter().map(|(i, _)| r[*i]).collect()
        } else if let Some(idxv) = &fallback_rest {
            idxv.iter().map(|i| r[*i]).collect()
        } else {
            Vec::new()
        };

        let opacity = r[i_opacity];

        let scale = [r[i_s0], r[i_s1], r[i_s2]];

        // normalize quaternion
        let mut rot = [r[i_qx], r[i_qy], r[i_qz], r[i_qw]];
        let n = (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3])
            .sqrt()
            .max(1e-8);
        rot[0] /= n;
        rot[1] /= n;
        rot[2] /= n;
        rot[3] /= n;

        for (m, v) in mins.iter_mut().zip(xyz.iter()) {
            *m = m.min(*v);
        }

        for (m, v) in maxes.iter_mut().zip(xyz.iter()) {
            *m = m.max(*v);
        }

        gaussians.push(Gaussian {
            xyz,
            normals,
            sh_dc,
            sh_rest,
            opacity,
            scale,
            rot,
        });
    }

    let mut meta_map = HashMap::new();
    meta_map.insert(
        "format".into(),
        match meta.format {
            PlyFormat::BinaryLittleEndian => "binary_little_endian",
            PlyFormat::Ascii => "ascii",
        }
        .to_string(),
    );
    meta_map.insert("vertex_count".into(), meta.vertex_count.to_string());

    Ok(Scene {
        gaussians,
        meta: meta_map,
        mins,
        maxes,
    })
}

// ---------------- internal helpers ----------------

fn parse_header(r: &mut BufReader<File>) -> Result<PlyMeta, PlyError> {
    let mut header_len: u64 = 0;
    let mut header_vec: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            return Err(header("EOF in header"));
        }
        header_len += n as u64;
        let trimmed = line.trim_end().to_string();
        if header_vec.is_empty() && !trimmed.starts_with("ply") {
            return Err(header("missing 'ply' signature"));
        }
        header_vec.push(trimmed);
        if header_vec.last().unwrap() == "end_header" {
            break;
        }
    }
    let fmt_line = header_vec
        .iter()
        .find(|l| l.starts_with("format "))
        .ok_or_else(|| header("missing 'format'"))?;
    let format = PlyFormat::from_header_line(fmt_line)?;

    let mut vertex_count: Option<usize> = None;
    let mut in_vertex = false;
    let mut properties: Vec<(String, String)> = Vec::new();
    for l in &header_vec {
        if l.starts_with("element ") {
            in_vertex = l.starts_with("element vertex");
            if in_vertex {
                let parts: Vec<&str> = l.split_whitespace().collect();
                let count = parts
                    .last()
                    .ok_or_else(|| header("bad 'element vertex'"))?
                    .parse::<usize>()
                    .map_err(|_| header("bad vertex count"))?;
                vertex_count = Some(count);
            }
        } else if in_vertex && l.starts_with("property ") {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.get(1) == Some(&"list") {
                continue;
            } // skip lists
            let dtype = parts.get(1).ok_or_else(|| header("bad property dtype"))?;
            let name = parts.get(2).ok_or_else(|| header("bad property name"))?;
            properties.push(((*name).to_string(), (*dtype).to_string()));
        }
    }
    Ok(PlyMeta {
        format,
        vertex_count: vertex_count.ok_or_else(|| header("no vertex element"))?,
        properties,
        header_len,
    })
}

fn dtype_size(t: &str) -> usize {
    match t {
        "char" | "uchar" => 1,
        "short" | "ushort" => 2,
        "int" | "uint" | "float" => 4,
        "double" => 8,
        _ => 4,
    }
}

fn read_vertices_binary(file: &mut File, meta: &PlyMeta) -> Result<Vec<Vec<f32>>, PlyError> {
    file.seek(SeekFrom::Start(meta.header_len))?;
    let stride: usize = meta.properties.iter().map(|(_, dt)| dtype_size(dt)).sum();
    let mut buf = vec![0u8; meta.vertex_count * stride];
    file.read_exact(&mut buf).map_err(|_| PlyError::Eof)?;

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(meta.vertex_count);
    let mut offset = 0usize;
    for _ in 0..meta.vertex_count {
        let mut row: Vec<f32> = Vec::with_capacity(meta.properties.len());
        let mut off = offset;
        for (_, dt) in &meta.properties {
            let f = match dt.as_str() {
                "char" => {
                    let v = buf[off] as i8;
                    off += 1;
                    v as f32
                }
                "uchar" => {
                    let v = buf[off] as u8;
                    off += 1;
                    v as f32
                }
                "short" => {
                    let v = i16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
                    off += 2;
                    v as f32
                }
                "ushort" => {
                    let v = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
                    off += 2;
                    v as f32
                }
                "int" => {
                    let v = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                    off += 4;
                    v as f32
                }
                "uint" => {
                    let v = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                    off += 4;
                    v as f32
                }
                "double" => {
                    let v = f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
                    off += 8;
                    v as f32
                }
                _ => {
                    let v = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                    off += 4;
                    v
                }
            };
            row.push(f);
        }
        rows.push(row);
        offset += stride;
    }
    Ok(rows)
}

fn read_vertices_ascii(
    reader: &mut BufReader<File>,
    meta: &PlyMeta,
) -> Result<Vec<Vec<f32>>, PlyError> {
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(meta.vertex_count);
    for _ in 0..meta.vertex_count {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let mut vals: Vec<f32> = Vec::with_capacity(meta.properties.len());
        for tok in line.split_whitespace().take(meta.properties.len()) {
            vals.push(tok.parse::<f32>().map_err(|_| PlyError::BadNumber)?);
        }
        if vals.len() != meta.properties.len() {
            vals.resize(meta.properties.len(), 0.0);
        }
        rows.push(vals);
    }
    Ok(rows)
}

fn idx(names: &[String], needle: &str) -> Result<usize, PlyError> {
    names
        .iter()
        .position(|n| n == needle)
        .ok_or_else(|| missing(needle))
}

fn idx_any(names: &[String], candidates: &[&str]) -> Result<usize, PlyError> {
    for c in candidates {
        if let Some(i) = names.iter().position(|n| n == *c) {
            return Ok(i);
        }
    }
    let msg = if !candidates.is_empty() {
        format!("one of [{}]", candidates.join(", "))
    } else {
        "column".to_string()
    };
    Err(missing(msg))
}

fn prefix_triple(names: &[String], prefixes: &[&str]) -> Result<[usize; 3], PlyError> {
    for pref in prefixes {
        let f1 = |i: usize| format!("{pref}{i}");
        let base = pref.trim_end_matches('_');
        let f2 = |i: usize| format!("{base}_{i}");
        let mut idxs: [Option<usize>; 3] = [None, None, None];
        for i in 0..3 {
            idxs[i] = names.iter().position(|n| n == &f1(i) || n == &f2(i));
        }
        if idxs.iter().all(|o| o.is_some()) {
            return Ok([idxs[0].unwrap(), idxs[1].unwrap(), idxs[2].unwrap()]);
        }
    }
    Err(missing("scale_0..2"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quaternion_normalization() {
        let mut q = [0.2f32, -0.3, 0.4, 0.5];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
            .sqrt()
            .max(1e-8);
        q[0] /= n;
        q[1] /= n;
        q[2] /= n;
        q[3] /= n;
        let one = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        assert!((one - 1.0).abs() < 1e-5);
    }
}
