//! CPU reference implementations.

/// NeRF-style alpha compositing.
///
/// `sigma_delta` is a row-major `[n_rays, n_samples]` buffer holding
/// `sigma_i * delta_i` for every sample along every ray. Returns the
/// compositing weights `w_i = T_i * (1 - exp(-sigma_i * delta_i))` with
/// `T_i = prod_{j<i} exp(-sigma_j * delta_j)`, same layout as the input.
///
/// This mirrors `nerfacc.render_weight_from_density` as used by threestudio's
/// `NeRFVolumeRenderer`.
pub fn alpha_composite_weights(sigma_delta: &[f32], n_samples: usize) -> Vec<f32> {
    assert!(n_samples > 0, "n_samples must be > 0");
    assert_eq!(
        sigma_delta.len() % n_samples,
        0,
        "buffer length must be a multiple of n_samples"
    );
    let mut weights = vec![0.0f32; sigma_delta.len()];
    for (ray_in, ray_out) in sigma_delta
        .chunks_exact(n_samples)
        .zip(weights.chunks_exact_mut(n_samples))
    {
        let mut transmittance = 1.0f32;
        for (sd, w) in ray_in.iter().zip(ray_out.iter_mut()) {
            let alpha = 1.0 - (-sd.max(0.0)).exp();
            *w = transmittance * alpha;
            transmittance *= 1.0 - alpha;
        }
    }
    weights
}

/// NeRF frequency positional encoding of `[n, dim]` points.
///
/// Output layout per point: `[x, sin(2^0 x), cos(2^0 x), ..., sin(2^{F-1} x), cos(2^{F-1} x)]`
/// where each block has `dim` entries, giving `dim * (1 + 2F)` features.
pub fn positional_encoding(points: &[f32], dim: usize, n_freqs: usize) -> Vec<f32> {
    assert!(dim > 0);
    assert_eq!(points.len() % dim, 0);
    let out_dim = dim * (1 + 2 * n_freqs);
    let n = points.len() / dim;
    let mut out = Vec::with_capacity(n * out_dim);
    for p in points.chunks_exact(dim) {
        out.extend_from_slice(p);
        for f in 0..n_freqs {
            let scale = (1u32 << f) as f32;
            out.extend(p.iter().map(|x| (x * scale).sin()));
            out.extend(p.iter().map(|x| (x * scale).cos()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_at_most_one() {
        let sd: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.3).collect();
        let w = alpha_composite_weights(&sd, 16);
        for ray in w.chunks(16) {
            let s: f32 = ray.iter().sum();
            assert!(s <= 1.0 + 1e-6, "sum {s}");
            assert!(ray.iter().all(|x| *x >= 0.0));
        }
    }

    #[test]
    fn empty_space_has_zero_weight() {
        let w = alpha_composite_weights(&[0.0; 8], 8);
        assert!(w.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn opaque_first_sample_takes_everything() {
        let mut sd = vec![0.0; 8];
        sd[0] = 1e4;
        let w = alpha_composite_weights(&sd, 8);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!(w[1..].iter().all(|x| x.abs() < 1e-6));
    }

    #[test]
    fn pe_shape_and_values() {
        let out = positional_encoding(&[0.0, 1.0, 2.0], 3, 4);
        assert_eq!(out.len(), 3 * 9);
        assert_eq!(&out[..3], &[0.0, 1.0, 2.0]);
        // sin(0) = 0, cos(0) = 1 for the first coordinate
        assert_eq!(out[3], 0.0);
        assert_eq!(out[6], 1.0);
    }
}
