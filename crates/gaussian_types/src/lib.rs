/// One Gaussian (per-vertex) record.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Gaussian {
    pub xyz: [f32; 3],
    pub normals: Option<[f32; 3]>,
    pub sh_dc: [f32; 3],
    pub sh_rest: Vec<f32>,     // e.g., 45 for Kerbl (L=3 per color)
    pub opacity: f32,
    pub scale: [f32; 3],       // log-scales
    pub rot: [f32; 4],         // quaternion xyzw (normalized)
}