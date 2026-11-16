# threestudio-rs

A Rust port scaffold of [threestudio](https://github.com/threestudio-project/threestudio)
— the unified framework for 3D content generation (DreamFusion, Magic3D, ProlificDreamer, ...)
— built on [Burn](https://burn.dev) `0.18` and [CubeCL](https://github.com/tracel-ai/cubecl) `0.6`,
with a CPU (`ndarray`) backend and a GPU backend on **wgpu → Vulkan (SPIR-V)**.

> **Status: first milestone.** The full pipeline (config → random cameras → implicit volume →
> volume rendering → score-distillation loss → Adam) runs end to end and trains, but the
> diffusion prior is a mathematically exact *mock* — there is no Stable Diffusion in Rust yet.
> See [What is ported](#what-is-ported-vs-stubbed).

## Quick start

```bash
# CPU (ndarray + autodiff)
cargo test
cargo run --release --bin launch -- --config configs/dreamfusion-toy.yaml --train --test

# GPU: Burn's wgpu backend compiled for Vulkan + custom CubeCL kernels
cargo build --release --features vulkan
cargo run --release --features vulkan --bin launch -- --config configs/dreamfusion-toy.yaml --train --gpu

# threestudio-style overrides
cargo run --release --bin launch -- --config configs/dreamfusion-toy.yaml --train \
    trainer.max_steps=500 system.prompt_processor.prompt="a blue ball" \
    system.guidance.mock_target_color=[0.1,0.2,1.0]
```

`--test` writes an 8-view evaluation orbit as PPM images to `outputs/<name>/`.

Toolchain: Rust **1.85** (MSRV, pinned in CI). Builds use the committed `Cargo.lock`
(`--locked`); it pins `bytemuck` to 1.23.x because newer releases enable AVX-512 types that
are only stable from Rust 1.89.

## Layout and mapping to threestudio

| threestudio (Python)                                     | threestudio-rs                                         |
|----------------------------------------------------------|--------------------------------------------------------|
| `launch.py`                                              | `crates/threestudio/src/bin/launch.rs`                 |
| `configs/*.yaml` + `utils/config.py` (OmegaConf)         | `configs/*.yaml` + `config.rs` (serde, `a.b=c` overrides) |
| `data/uncond.py` (`RandomCameraDataModule`)              | `data.rs` (`RandomCameraDataset`, eval orbit)          |
| `utils/ops.py` (`get_ray_directions`, `get_rays`)        | `data.rs` (`ray_directions`, `RayBatch`)               |
| `models/networks.py` (encodings)                         | `models/encoding.rs` (`FrequencyEncoding`)             |
| `models/geometry/implicit_volume.py`                     | `models/geometry.rs` (`ImplicitVolume`)                |
| `models/renderers/nerf_volume_renderer.py`               | `models/renderer.rs` (`NeRFVolumeRenderer`)            |
| `models/prompt_processors/base.py`                       | `models/prompt.rs` (view-dependent prompting)          |
| `models/guidance/stable_diffusion_guidance.py`           | `models/guidance.rs` (`ScoreDistillationGuidance`)     |
| `systems/dreamfusion.py`                                 | `systems/dreamfusion.rs` (`DreamFusion`)               |
| CUDA extensions (nerfacc / tiny-cuda-nn)                 | `crates/threestudio-kernels` (CubeCL + CPU reference)  |

## What is ported vs stubbed

**Implemented and tested**

- **Config**: threestudio-shaped YAML schema (`name`, `data_type`, `data`, `system_type`,
  `system.{geometry,renderer,background,prompt_processor,guidance,loss,optimizer}`, `trainer`),
  defaults for every field, dotted CLI overrides.
- **Data**: random spherical cameras over elevation/azimuth/distance/FoV ranges, +z-up look-at
  frames, OpenGL-convention pixel rays, deterministic 360° eval orbit.
- **Geometry**: `ImplicitVolume` — frequency encoding → ReLU MLP → softplus density with
  Magic3D's `blob_magic3d` density bias, sigmoid albedo.
- **Renderer**: stratified (jittered) ray marching, alpha compositing via an exclusive
  cumulative sum, opacity / depth / weights outputs, background compositing. Fully
  differentiable on any Burn autodiff backend.
- **Guidance**: the SDS update of `StableDiffusionGuidance` — SD `scaled_linear` noise
  schedule, timestep sampling in `[min_step_percent, max_step_percent]`, `w(t) = 1 - ᾱ_t`,
  optional grad clipping, and the reparameterised loss `0.5·‖x − sg(x − grad)‖²`.
  The UNet is behind a `NoisePredictor` trait.
- **System**: `DreamFusion` training step (render random view → guidance → optional sparsity
  loss → Adam), `fit`, evaluation and PPM export.
- **CubeCL kernels** (`threestudio-kernels`, feature `gpu`): front-to-back alpha compositing
  and NeRF positional encoding, each with a CPU reference implementation. GPU tests compare
  against the reference and skip cleanly when no Vulkan adapter is present.

**Stubbed / not yet ported**

- **Diffusion prior**: `MockDenoiser` is the exact ε-predictor for a data distribution
  concentrated on one colour, so SDS provably pulls renders toward that colour. A real
  Stable Diffusion / DeepFloyd guidance needs a UNet + VAE + CLIP text encoder in Burn and
  weight loading; `stable-diffusion-guidance` currently returns an error.
- **Text encoder**: `PromptEmbedding.text_embeddings` is always `None`.
- **Hash-grid encoding** (tiny-cuda-nn), **occupancy-grid acceleration** (nerfacc),
  normals/shading, mesh geometries (DMTet), other systems (Magic3D, ProlificDreamer, Zero-1-to-3),
  checkpointing, and the Lightning-style trainer/callback machinery.
- The CubeCL kernels are forward-only and not yet fused into the Burn autodiff graph; the
  renderer uses Burn tensor ops on every backend.

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test                 # CPU: rays, encodings, weights ≤ 1, SDS gradients, training reduces error
cargo test --all-features  # + CubeCL kernel and Burn-Vulkan checks (skipped without a Vulkan adapter)
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy, the all-features build, CPU tests and a launch
smoke run, plus a second job that installs Mesa's lavapipe software Vulkan driver so the CubeCL
kernels run against their CPU references on a real Vulkan device.

## License and credit

Licensed under the [Apache License 2.0](LICENSE), the same license as the original project.

This is a derivative port of **threestudio** by Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao,
Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao
and Song-Hai Zhang (<https://github.com/threestudio-project/threestudio>); see [NOTICE](NOTICE).
If you use this work, please cite threestudio:

```bibtex
@Misc{threestudio2023,
  author =       {Yuan-Chen Guo and Ying-Tian Liu and Ruizhi Shao and Christian Laforte and Vikram Voleti and Guan Luo and Chia-Hao Chen and Zi-Xin Zou and Chen Wang and Yan-Pei Cao and Song-Hai Zhang},
  title =        {threestudio: A unified framework for 3D content generation},
  howpublished = {\url{https://github.com/threestudio-project/threestudio}},
  year =         {2023}
}
```
