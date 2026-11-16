//! Backend aliases.

use burn::backend::{Autodiff, NdArray};

/// CPU inference backend (always available).
pub type Cpu = NdArray<f32>;
/// CPU training backend.
pub type CpuAutodiff = Autodiff<Cpu>;

/// Vulkan (wgpu + SPIR-V) inference backend.
#[cfg(feature = "vulkan")]
pub type Gpu = burn::backend::Vulkan<f32, i32>;
/// Vulkan training backend.
#[cfg(feature = "vulkan")]
pub type GpuAutodiff = Autodiff<Gpu>;

/// Whether a Vulkan adapter is visible (always `false` without the `vulkan` feature).
pub fn gpu_available() -> bool {
    #[cfg(feature = "vulkan")]
    {
        threestudio_kernels::gpu::gpu_available()
    }
    #[cfg(not(feature = "vulkan"))]
    {
        false
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::models::encoding::FrequencyEncoding;
    use burn::prelude::*;

    #[test]
    fn vulkan_backend_matches_cpu() {
        if !gpu_available() {
            eprintln!("no Vulkan adapter found; skipping Burn Vulkan backend test");
            return;
        }
        let pts: Vec<f32> = (0..30).map(|i| i as f32 * 0.05 - 0.7).collect();
        let enc = FrequencyEncoding::new(3, 5);
        let cpu = enc
            .forward(Tensor::<Cpu, 2>::from_data(
                TensorData::new(pts.clone(), [10, 3]),
                &Default::default(),
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gpu = enc
            .forward(Tensor::<Gpu, 2>::from_data(
                TensorData::new(pts, [10, 3]),
                &Default::default(),
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in cpu.iter().zip(&gpu) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
