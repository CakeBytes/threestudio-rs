//! Input encodings (`threestudio/models/networks.py`).
//!
//! threestudio uses tiny-cuda-nn's `HashGrid` by default; this port currently
//! provides the NeRF `Frequency` encoding (also supported by threestudio via
//! `pos_encoding_config.otype: Frequency`). A hash grid is on the roadmap.

use burn::prelude::*;

/// NeRF frequency encoding: `[x, sin(2^k x), cos(2^k x)]_{k < n_frequencies}`.
#[derive(Module, Debug, Clone)]
pub struct FrequencyEncoding {
    pub in_dim: usize,
    pub n_frequencies: usize,
}

impl FrequencyEncoding {
    pub fn new(in_dim: usize, n_frequencies: usize) -> Self {
        Self {
            in_dim,
            n_frequencies,
        }
    }

    pub fn out_dim(&self) -> usize {
        self.in_dim * (1 + 2 * self.n_frequencies)
    }

    /// `[n, in_dim]` -> `[n, out_dim]`, same layout as
    /// [`threestudio_kernels::positional_encoding`].
    pub fn forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        assert_eq!(d, self.in_dim, "encoding input dim mismatch");
        if self.n_frequencies == 0 {
            return x;
        }
        let f = self.n_frequencies;
        let freqs: Vec<f32> = (0..f).map(|k| (1u32 << k) as f32).collect();
        let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), &x.device()).reshape([1, f, 1]);
        let scaled = x.clone().reshape([n, 1, d]) * freqs; // [n, f, d]
        let sc = Tensor::cat(vec![scaled.clone().sin(), scaled.cos()], 2); // [n, f, 2d]
        Tensor::cat(vec![x, sc.reshape([n, f * 2 * d])], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    #[test]
    fn shape_is_correct() {
        let enc = FrequencyEncoding::new(3, 6);
        let x = Tensor::<Cpu, 2>::zeros([10, 3], &Default::default());
        assert_eq!(enc.forward(x).dims(), [10, 39]);
        assert_eq!(enc.out_dim(), 39);
    }

    #[test]
    fn matches_cpu_reference_kernel() {
        let pts: Vec<f32> = (0..24).map(|i| i as f32 * 0.11 - 1.3).collect();
        let enc = FrequencyEncoding::new(3, 4);
        let x =
            Tensor::<Cpu, 2>::from_data(TensorData::new(pts.clone(), [8, 3]), &Default::default());
        let got = enc.forward(x).into_data().to_vec::<f32>().unwrap();
        let want = threestudio_kernels::positional_encoding(&pts, 3, 4);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
