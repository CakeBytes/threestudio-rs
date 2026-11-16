//! CubeCL kernels on the wgpu runtime.

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

const WORKGROUP: u32 = 64;

/// One invocation per ray: front-to-back alpha compositing along the ray.
#[cube(launch_unchecked)]
fn alpha_composite_kernel<F: Float>(
    sigma_delta: &Array<F>,
    weights: &mut Array<F>,
    n_rays: u32,
    n_samples: u32,
) {
    let ray = ABSOLUTE_POS;
    if ray < n_rays {
        let base = ray * n_samples;
        let mut transmittance = F::new(1.0);
        for s in 0..n_samples {
            let sd = F::max(sigma_delta[base + s], F::new(0.0));
            let alpha = F::new(1.0) - F::exp(F::new(0.0) - sd);
            weights[base + s] = transmittance * alpha;
            transmittance *= F::new(1.0) - alpha;
        }
    }
}

/// One invocation per (point, frequency) pair.
#[cube(launch_unchecked)]
fn positional_encoding_kernel<F: Float>(
    points: &Array<F>,
    out: &mut Array<F>,
    n_points: u32,
    n_freqs: u32,
) {
    // dim is fixed to 3 (xyz) for this kernel.
    let idx = ABSOLUTE_POS;
    if idx < n_points * (n_freqs + 1) {
        let p = idx / (n_freqs + 1);
        let slot = idx % (n_freqs + 1);
        let out_dim = 3 * (1 + 2 * n_freqs);
        let in_base = p * 3;
        let out_base = p * out_dim;
        if slot == 0 {
            for d in 0..3u32 {
                out[out_base + d] = points[in_base + d];
            }
        } else {
            let f = slot - 1;
            let scale = F::powf(F::new(2.0), F::cast_from(f));
            let o = out_base + 3 + f * 6;
            for d in 0..3u32 {
                let x = points[in_base + d] * scale;
                out[o + d] = F::sin(x);
                out[o + 3 + d] = F::cos(x);
            }
        }
    }
}

/// Returns `true` when wgpu can see at least one Vulkan adapter on this machine.
///
/// The runtime is compiled with the SPIR-V compiler, so only Vulkan adapters
/// count. Used by tests and the CLI to skip GPU paths gracefully on headless
/// machines (such as CI runners).
pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    !instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .is_empty()
}

fn cube_count(n: usize) -> CubeCount {
    CubeCount::Static(Ord::max((n as u32).div_ceil(WORKGROUP), 1), 1, 1)
}

/// GPU version of [`crate::cpu::alpha_composite_weights`].
pub fn alpha_composite_weights(sigma_delta: &[f32], n_samples: usize) -> Vec<f32> {
    assert!(n_samples > 0);
    assert_eq!(sigma_delta.len() % n_samples, 0);
    let n_rays = sigma_delta.len() / n_samples;
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let input = client.create(f32::as_bytes(sigma_delta));
    let output = client.empty(std::mem::size_of_val(sigma_delta));
    unsafe {
        alpha_composite_kernel::launch_unchecked::<f32, WgpuRuntime>(
            &client,
            cube_count(n_rays),
            CubeDim::new(WORKGROUP, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, sigma_delta.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, sigma_delta.len(), 1),
            ScalarArg::new(n_rays as u32),
            ScalarArg::new(n_samples as u32),
        );
    }
    let bytes = client.read_one(output.binding());
    f32::from_bytes(&bytes).to_vec()
}

/// GPU version of [`crate::cpu::positional_encoding`] for 3-D points.
pub fn positional_encoding(points: &[f32], n_freqs: usize) -> Vec<f32> {
    assert_eq!(points.len() % 3, 0);
    let n_points = points.len() / 3;
    let out_len = n_points * 3 * (1 + 2 * n_freqs);
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let input = client.create(f32::as_bytes(points));
    let output = client.empty(out_len * std::mem::size_of::<f32>());
    unsafe {
        positional_encoding_kernel::launch_unchecked::<f32, WgpuRuntime>(
            &client,
            cube_count(n_points * (n_freqs + 1)),
            CubeDim::new(WORKGROUP, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, points.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, out_len, 1),
            ScalarArg::new(n_points as u32),
            ScalarArg::new(n_freqs as u32),
        );
    }
    let bytes = client.read_one(output.binding());
    f32::from_bytes(&bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn alpha_composite_matches_cpu() {
        if !gpu_available() {
            eprintln!("no wgpu adapter found; skipping GPU kernel test");
            return;
        }
        let sd: Vec<f32> = (0..32 * 24)
            .map(|i| ((i * 37) % 11) as f32 * 0.05)
            .collect();
        close(
            &alpha_composite_weights(&sd, 24),
            &crate::cpu::alpha_composite_weights(&sd, 24),
            1e-5,
        );
    }

    #[test]
    fn positional_encoding_matches_cpu() {
        if !gpu_available() {
            eprintln!("no wgpu adapter found; skipping GPU kernel test");
            return;
        }
        let pts: Vec<f32> = (0..30).map(|i| i as f32 * 0.07 - 1.0).collect();
        close(
            &positional_encoding(&pts, 6),
            &crate::cpu::positional_encoding(&pts, 3, 6),
            1e-4,
        );
    }
}
