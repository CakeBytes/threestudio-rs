//! Minimal single-file `.safetensors` reader and a candle-style `VarBuilder`.
//!
//! The reader memory-maps the file and converts tensors (F16 / BF16 / F32) to
//! f32 on demand, so a 6.9 GB checkpoint can be consumed block by block.

use anyhow::{bail, Context, Result};
use burn::prelude::*;
use burn::tensor::Distribution;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// One entry of the safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// A memory-mapped safetensors file.
pub struct SafeTensorsFile {
    mmap: memmap2::Mmap,
    data_start: usize,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is opened read-only and not modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        anyhow::ensure!(mmap.len() >= 8, "file too small for safetensors");
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        anyhow::ensure!(8 + n <= mmap.len(), "corrupt safetensors header");
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n])?;
        let obj = header.as_object().context("header is not an object")?;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (mk, mv) in m {
                        metadata.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().context("dtype")?.to_string();
            let shape = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"].as_array().context("data_offsets")?;
            let (start, end) = (
                off[0].as_u64().unwrap_or(0) as usize,
                off[1].as_u64().unwrap_or(0) as usize,
            );
            tensors.insert(
                k.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            mmap,
            data_start: 8 + n,
            tensors,
            metadata,
        })
    }

    /// Read a tensor as f32 values (row-major) plus its shape.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let bytes = &self.mmap[self.data_start + info.start..self.data_start + info.end];
        let values = match info.dtype.as_str() {
            "F16" => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => bail!("unsupported dtype {other} for {name}"),
        };
        Ok((values, info.shape.clone()))
    }
}

/// Initialisation hint used when weights are generated randomly.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    Zeros,
    Ones,
    /// Normal(0, std).
    Normal(f64),
}

enum Source {
    /// Random weights (for shape tests without a checkpoint).
    Random,
    /// Real weights from a checkpoint.
    File(Arc<SafeTensorsFile>),
    /// Only validate names and shapes against the checkpoint header; returns
    /// 1-element placeholder tensors. Used by the key-mapping coverage test.
    Meta(Arc<SafeTensorsFile>),
}

struct Shared {
    source: Source,
    used: Mutex<BTreeSet<String>>,
}

/// Hierarchical weight accessor: `vb.pp("input_blocks").pp(1).get(...)`.
pub struct VarBuilder<B: Backend> {
    shared: Arc<Shared>,
    prefix: String,
    pub device: B::Device,
}

impl<B: Backend> Clone for VarBuilder<B> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            prefix: self.prefix.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B: Backend> VarBuilder<B> {
    fn new(source: Source, device: &B::Device) -> Self {
        Self {
            shared: Arc::new(Shared {
                source,
                used: Mutex::new(BTreeSet::new()),
            }),
            prefix: String::new(),
            device: device.clone(),
        }
    }

    pub fn random(device: &B::Device) -> Self {
        Self::new(Source::Random, device)
    }

    pub fn from_file(file: Arc<SafeTensorsFile>, device: &B::Device) -> Self {
        Self::new(Source::File(file), device)
    }

    pub fn meta(file: Arc<SafeTensorsFile>, device: &B::Device) -> Self {
        Self::new(Source::Meta(file), device)
    }

    /// Push a path segment.
    pub fn pp(&self, seg: impl std::fmt::Display) -> Self {
        let prefix = if self.prefix.is_empty() {
            seg.to_string()
        } else {
            format!("{}.{seg}", self.prefix)
        };
        Self {
            shared: self.shared.clone(),
            prefix,
            device: self.device.clone(),
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn is_meta(&self) -> bool {
        matches!(self.shared.source, Source::Meta(_))
    }

    /// Names of all tensors requested so far (full keys).
    pub fn used_keys(&self) -> BTreeSet<String> {
        self.shared.used.lock().unwrap().clone()
    }

    pub fn contains(&self, name: &str) -> bool {
        let key = self.pp(name).prefix;
        match &self.shared.source {
            Source::Random => true,
            Source::File(f) | Source::Meta(f) => f.tensors.contains_key(&key),
        }
    }

    /// Fetch a tensor with an expected shape.
    pub fn get<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        init: Init,
    ) -> Result<Tensor<B, D>> {
        let key = self.pp(name).prefix;
        self.shared.used.lock().unwrap().insert(key.clone());
        match &self.shared.source {
            Source::Random => Ok(match init {
                Init::Zeros => Tensor::zeros(shape, &self.device),
                Init::Ones => Tensor::ones(shape, &self.device),
                Init::Normal(std) => {
                    Tensor::random(shape, Distribution::Normal(0.0, std), &self.device)
                }
            }),
            Source::Meta(f) => {
                let info = f
                    .tensors
                    .get(&key)
                    .with_context(|| format!("missing tensor {key}"))?;
                anyhow::ensure!(
                    info.shape == shape,
                    "shape mismatch for {key}: file {:?}, expected {:?}",
                    info.shape,
                    shape
                );
                Ok(Tensor::zeros([1; D], &self.device))
            }
            Source::File(f) => {
                let (values, file_shape) = f.read_f32(&key)?;
                anyhow::ensure!(
                    file_shape == shape,
                    "shape mismatch for {key}: file {:?}, expected {:?}",
                    file_shape,
                    shape
                );
                Ok(Tensor::from_data(
                    TensorData::new(values, shape),
                    &self.device,
                ))
            }
        }
    }

    /// Fetch a PyTorch `[out, in]` linear weight and return it transposed to `[in, out]`.
    pub fn get_linear_weight(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Tensor<B, 2>> {
        let key = self.pp(name).prefix;
        match &self.shared.source {
            Source::File(f) => {
                self.shared.used.lock().unwrap().insert(key.clone());
                let (values, file_shape) = f.read_f32(&key)?;
                anyhow::ensure!(
                    file_shape == [out_dim, in_dim],
                    "shape mismatch for {key}: file {:?}, expected {:?}",
                    file_shape,
                    [out_dim, in_dim]
                );
                let mut t = vec![0.0f32; values.len()];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        t[i * out_dim + o] = values[o * in_dim + i];
                    }
                }
                Ok(Tensor::from_data(
                    TensorData::new(t, [in_dim, out_dim]),
                    &self.device,
                ))
            }
            _ => Ok(self
                .get(name, [out_dim, in_dim], Init::Normal(0.02))?
                .transpose()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use std::io::Write;

    fn write_test_file(path: &Path) {
        let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect();
        let header = serde_json::json!({
            "__metadata__": {"k": "v"},
            "lin.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, a.len()]},
        })
        .to_string();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&a).unwrap();
    }

    #[test]
    fn reads_f16_and_transposes_linear() {
        let path =
            std::env::temp_dir().join(format!("ts-rs-test-{}.safetensors", std::process::id()));
        write_test_file(&path);
        let file = Arc::new(SafeTensorsFile::open(&path).unwrap());
        assert_eq!(file.metadata["k"], "v");
        let vb = VarBuilder::<Cpu>::from_file(file.clone(), &Default::default());
        let w = vb.pp("lin").get_linear_weight("weight", 2, 3).unwrap();
        assert_eq!(w.dims(), [3, 2]);
        assert_eq!(
            w.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert!(vb
            .pp("lin")
            .get::<2>("weight", [3, 2], Init::Zeros)
            .is_err());
        let meta = VarBuilder::<Cpu>::meta(file, &Default::default());
        assert!(meta.get::<2>("lin.weight", [2, 3], Init::Zeros).is_ok());
        assert!(meta.get::<1>("nope", [1], Init::Zeros).is_err());
        assert!(meta.used_keys().contains("lin.weight"));
        std::fs::remove_file(path).ok();
    }
}
