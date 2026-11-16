//! Low-level kernels used by `threestudio-rs`.
//!
//! Every kernel has a plain-Rust CPU reference implementation (always compiled)
//! and, behind the `gpu` feature, a CubeCL implementation that runs on the
//! wgpu runtime (Vulkan / SPIR-V on Linux).

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{alpha_composite_weights, positional_encoding};
