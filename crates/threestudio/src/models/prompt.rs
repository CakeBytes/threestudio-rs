//! Prompt processing (`threestudio/models/prompt_processors/base.py`).
//!
//! Text encoders (CLIP/T5) are not ported yet; this module keeps the prompt
//! strings and implements threestudio's view-dependent prompting.

use crate::config::PromptProcessorConfig;

/// Encoded prompt. `text_embeddings` is `None` until a text encoder is ported.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptEmbedding {
    pub prompt: String,
    pub negative_prompt: String,
    pub text_embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct PromptProcessor {
    pub cfg: PromptProcessorConfig,
    pub front_threshold: f32,
    pub back_threshold: f32,
    pub overhead_threshold: f32,
}

impl PromptProcessor {
    pub fn new(cfg: PromptProcessorConfig) -> Self {
        Self {
            cfg,
            front_threshold: 45.0,
            back_threshold: 45.0,
            overhead_threshold: 60.0,
        }
    }

    /// View suffix used by threestudio's `view_dependent_prompting`.
    pub fn view_suffix(&self, elevation_deg: f32, azimuth_deg: f32) -> &'static str {
        // shift azimuth into [-180, 180)
        let azim = (azimuth_deg + 180.0).rem_euclid(360.0) - 180.0;
        if elevation_deg > self.overhead_threshold {
            "overhead view"
        } else if azim.abs() < self.front_threshold {
            "front view"
        } else if azim.abs() > 180.0 - self.back_threshold {
            "back view"
        } else {
            "side view"
        }
    }

    pub fn embed(&self, elevation_deg: f32, azimuth_deg: f32) -> PromptEmbedding {
        PromptEmbedding {
            prompt: format!(
                "{}, {}",
                self.cfg.prompt,
                self.view_suffix(elevation_deg, azimuth_deg)
            ),
            negative_prompt: self.cfg.negative_prompt.clone(),
            text_embeddings: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_dependent_suffixes() {
        let p = PromptProcessor::new(Default::default());
        assert_eq!(p.view_suffix(0.0, 0.0), "front view");
        assert_eq!(p.view_suffix(0.0, 90.0), "side view");
        assert_eq!(p.view_suffix(0.0, 179.0), "back view");
        assert_eq!(p.view_suffix(0.0, -179.0), "back view");
        assert_eq!(p.view_suffix(80.0, 0.0), "overhead view");
        assert!(p.embed(0.0, 0.0).prompt.ends_with("front view"));
    }
}
