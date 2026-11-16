//! Implicit volume geometry (`threestudio/models/geometry/implicit_volume.py`).

use super::encoding::FrequencyEncoding;
use crate::config::GeometryConfig;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, softplus};

/// Density + albedo field: `x -> (sigma, rgb)`.
#[derive(Module, Debug)]
pub struct ImplicitVolume<B: Backend> {
    encoding: FrequencyEncoding,
    hidden: Vec<Linear<B>>,
    density_head: Linear<B>,
    feature_head: Linear<B>,
    radius: f32,
    blob_scale: f32,
    blob_std: f32,
}

/// Output of [`ImplicitVolume::forward`].
pub struct GeometryOutput<B: Backend> {
    /// `[n, 1]`, non-negative.
    pub density: Tensor<B, 2>,
    /// `[n, 3]`, in `[0, 1]`.
    pub features: Tensor<B, 2>,
}

impl<B: Backend> ImplicitVolume<B> {
    pub fn new(cfg: &GeometryConfig, device: &B::Device) -> Self {
        let encoding = FrequencyEncoding::new(3, cfg.n_frequencies);
        let mut hidden = Vec::new();
        let mut in_dim = encoding.out_dim();
        for _ in 0..cfg.n_hidden_layers.max(1) {
            hidden.push(LinearConfig::new(in_dim, cfg.n_neurons).init(device));
            in_dim = cfg.n_neurons;
        }
        let (blob_scale, blob_std) = match cfg.density_bias.as_str() {
            "blob_magic3d" | "blob" => (cfg.density_blob_scale, cfg.density_blob_std),
            _ => (0.0, 1.0),
        };
        Self {
            encoding,
            hidden,
            density_head: LinearConfig::new(in_dim, 1).init(device),
            feature_head: LinearConfig::new(in_dim, 3).init(device),
            radius: cfg.radius,
            blob_scale,
            blob_std,
        }
    }

    /// Magic3D-style density blob centred at the origin:
    /// `scale * (1 - |x| / std)`.
    fn density_bias(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let r = points.powf_scalar(2.0).sum_dim(1).sqrt();
        (r.div_scalar(self.blob_std).neg().add_scalar(1.0)).mul_scalar(self.blob_scale)
    }

    /// `points`: `[n, 3]` world-space positions.
    pub fn forward(&self, points: Tensor<B, 2>) -> GeometryOutput<B> {
        let mut h = self
            .encoding
            .forward(points.clone().div_scalar(self.radius));
        for layer in &self.hidden {
            h = relu(layer.forward(h));
        }
        let raw_density = self.density_head.forward(h.clone()) + self.density_bias(points);
        GeometryOutput {
            // threestudio defaults to `trunc_exp`; softplus is the numerically
            // safer alternative it also supports.
            density: softplus(raw_density, 1.0),
            features: sigmoid(self.feature_head.forward(h)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    #[test]
    fn output_shapes_and_ranges() {
        let device = Default::default();
        let geo = ImplicitVolume::<Cpu>::new(&GeometryConfig::default(), &device);
        let pts = Tensor::<Cpu, 2>::random(
            [50, 3],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let out = geo.forward(pts);
        assert_eq!(out.density.dims(), [50, 1]);
        assert_eq!(out.features.dims(), [50, 3]);
        let d = out.density.into_data().to_vec::<f32>().unwrap();
        assert!(d.iter().all(|x| *x >= 0.0 && x.is_finite()));
        let f = out.features.into_data().to_vec::<f32>().unwrap();
        assert!(f.iter().all(|x| (0.0..=1.0).contains(x)));
    }
}
