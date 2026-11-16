# threestudio-rs

A Rust port scaffold of [threestudio](https://github.com/threestudio-project/threestudio)
— the unified framework for 3D content generation (DreamFusion, Magic3D, ProlificDreamer, ...)
— built on [Burn](https://burn.dev) `0.18` and [CubeCL](https://github.com/tracel-ai/cubecl) `0.6`,
with a CPU (`ndarray`) backend and a GPU backend on **wgpu → Vulkan (SPIR-V)**.

> **Status.** The full pipeline (config → random cameras → implicit volume → volume rendering →
> score-distillation loss → Adam) runs end to end. Guidance can be a cheap exact *mock* or a real
> **SDXL** prior implemented in Burn (UNet, CLIP ViT-L + OpenCLIP ViT-bigG, VAE encoder) that loads
> single-file checkpoints such as **Pony Diffusion V6 XL** directly.
> See [What is ported](#what-is-ported-vs-stubbed) and [SDXL / Pony guidance](#sdxl--pony-diffusion-guidance).

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

## SDXL / Pony Diffusion guidance

`guidance_type: stable-diffusion-xl-guidance` (alias `pony-diffusion-guidance`) runs score
distillation with a real SDXL model, written from scratch in Burn (`crates/threestudio/src/models/sdxl`):

| component | checkpoint prefix | notes |
|-----------|-------------------|-------|
| UNet (320/640/1280 ch, transformer depth 0/2/10, mid 10, 64-d heads, cross-attn 2048, `text_time` added cond 2816) | `model.diffusion_model.*` | LDM key names; block plan in `unet.rs` |
| CLIP ViT-L/14 text encoder (penultimate layer, quick-GELU) | `conditioner.embedders.0.transformer.*` | HF layout |
| OpenCLIP ViT-bigG/14 text encoder (penultimate layer + pooled `text_projection`) | `conditioner.embedders.1.model.*` | fused `in_proj` split into q/k/v |
| VAE encoder + `quant_conv`, latent scale 0.13025 | `first_stage_model.*` | differentiable; `latent_encoder: linear` is a documented approximate fallback |
| CLIP BPE tokenizer | `assets/clip/merges.txt` | byte-level BPE port of CLIP's `simple_tokenizer`; vocab verified identical to `openai/clip-vit-large-patch14` |

Weights are read from the `.safetensors` file through a memory map (F16/BF16/F32 → f32), with no
conversion step. Classifier-free guidance batches the negative and positive prompts; an empty negative
prompt uses zero embeddings (SDXL's `force_zeros_for_empty_prompt`). Pony's quality tags go in
`prompt_processor.prompt_prefix` / `negative_prompt_prefix`. See `configs/dreamfusion-pony.yaml`.

Get the checkpoint (Civitai model 257749, version 290640, file `ponyDiffusionV6XL_v6.safetensors`,
SHA256 `67ab2fd8ec439a89b3fedb15cc65f54336af163c7eb5e4f2acc98f090a29b0b3`):

```bash
mkdir -p ~/.cache/threestudio-rs/pony
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" \
  -o ~/.cache/threestudio-rs/pony/ponyDiffusionV6XL_v6.safetensors \
  "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor"
```

Sanity probe: one real denoising step, UNet streamed block by block from disk:

```bash
cargo run --release --bin sdxl-probe -- --resolution 256
```

Measured on an 8-vCPU CPU-only box (Burn ndarray, f32, 256 px → 32×32 latent, t = 500):

```
VAE encode 256x256 -> [1, 4, 32, 32] in 4.8s (latent mean 0.132, std 1.490)
UNet (CFG batch of 2, t=500, streamed from disk) incl. text encoding: 62.0s
cos(eps_cond, true noise)   = 0.9878
x0 MSE: predicted 0.0630 vs naive (eps=0) 2.6080
```

Memory: the UNet is ~2.6 B parameters, i.e. ~10.3 GB in f32 (all weights resident, `low_memory: false`).
`low_memory: true` streams UNet blocks and OpenCLIP layers from the memory-mapped file (~1.7 GB peak for
the largest block) at the cost of re-reading weights on every step. Practical training at 512–1024 px
needs a GPU; f16 execution on the wgpu backend is not wired up yet.

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
- **SDXL guidance**: see [above](#sdxl--pony-diffusion-guidance); key-mapping coverage is tested
  against the real Pony checkpoint (2373 of 2515 tensors used; the rest are the VAE decoder,
  `post_quant_conv`, `position_ids`, `logit_scale`).
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

- **Diffusion priors other than SDXL**: SD 1.x/2.x (`stable-diffusion-guidance`), DeepFloyd IF,
  Zero-1-to-3 and VSD/LoRA guidance (ProlificDreamer) are not ported. `MockDenoiser` remains as a
  cheap exact ε-predictor for tests.
- **SDXL performance**: f32 only, no flash attention, no f16 on GPU yet; CPU is far too slow for
  real training (one streamed UNet step at 32×32 latents takes about a minute).
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
