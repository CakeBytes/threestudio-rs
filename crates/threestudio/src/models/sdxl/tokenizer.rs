//! CLIP byte-level BPE tokenizer (port of OpenAI CLIP's `simple_tokenizer.py`,
//! which OpenCLIP also uses). The merges list is the one shipped with
//! `openai/clip-vit-large-patch14` and is embedded in the binary.

use regex::Regex;
use std::collections::HashMap;

pub const BOS: u32 = 49406;
pub const EOS: u32 = 49407;
pub const CONTEXT_LENGTH: usize = 77;

static MERGES: &str = include_str!("../../../../../assets/clip/merges.txt");

pub struct ClipTokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    pattern: Regex,
}

fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b as u8, char::from_u32(c).unwrap()))
        .collect()
}

impl Default for ClipTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipTokenizer {
    pub fn new() -> Self {
        let b2u = bytes_to_unicode();
        let mut byte_encoder = [' '; 256];
        for (b, c) in &b2u {
            byte_encoder[*b as usize] = *c;
        }
        let merges: Vec<(String, String)> = MERGES
            .lines()
            .skip(1)
            .take(49152 - 256 - 2)
            .filter_map(|l| {
                l.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
            })
            .collect();
        let mut vocab: Vec<String> = b2u.iter().map(|(_, c)| c.to_string()).collect();
        vocab.extend(b2u.iter().map(|(_, c)| format!("{c}</w>")));
        vocab.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
        vocab.push("<|startoftext|>".into());
        vocab.push("<|endoftext|>".into());
        let encoder = vocab
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t, i as u32))
            .collect();
        let ranks = merges
            .into_iter()
            .enumerate()
            .map(|(i, m)| (m, i))
            .collect();
        let pattern = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )
        .expect("valid regex");
        Self {
            encoder,
            ranks,
            byte_encoder,
            pattern,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        loop {
            if word.len() < 2 {
                break;
            }
            let best = word
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.ranks
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            let Some((_, _)) = best else { break };
            let (first, second) = {
                let i = best.unwrap().1;
                (word[i].clone(), word[i + 1].clone())
            };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    /// Token ids without BOS/EOS.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let text = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let mut out = Vec::new();
        for m in self.pattern.find_iter(&text) {
            let tok: String = m
                .as_str()
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            for piece in self.bpe(&tok) {
                if let Some(id) = self.encoder.get(&piece) {
                    out.push(*id);
                }
            }
        }
        out
    }

    /// `[BOS, tokens.., EOS, pad..]` truncated/padded to 77.
    pub fn encode(&self, text: &str, pad_id: u32) -> Vec<u32> {
        let mut ids = vec![BOS];
        let raw = self.encode_raw(text);
        ids.extend(raw.into_iter().take(CONTEXT_LENGTH - 2));
        ids.push(EOS);
        ids.resize(CONTEXT_LENGTH, pad_id);
        ids
    }
}

/// Index of the first EOS token (OpenCLIP pools at `argmax(ids)`).
pub fn eos_position(ids: &[u32]) -> usize {
    ids.iter().position(|&t| t == EOS).unwrap_or(ids.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_has_expected_size() {
        let t = ClipTokenizer::new();
        assert_eq!(t.vocab_size(), 49408);
        assert_eq!(t.token_id("<|startoftext|>"), Some(BOS));
        assert_eq!(t.token_id("<|endoftext|>"), Some(EOS));
    }

    #[test]
    fn matches_reference_ids() {
        let t = ClipTokenizer::new();
        // reference: transformers CLIPTokenizer("openai/clip-vit-large-patch14")
        assert_eq!(
            t.encode_raw("a photo of a cat"),
            vec![320, 1125, 539, 320, 2368]
        );
        assert_eq!(
            t.encode_raw("A  Photo of a CAT"),
            vec![320, 1125, 539, 320, 2368]
        );
        let ids = t.encode("a photo of a cat", EOS);
        assert_eq!(&ids[..7], &[BOS, 320, 1125, 539, 320, 2368, EOS]);
        assert_eq!(ids.len(), 77);
        assert_eq!(eos_position(&ids), 6);
        let ids0 = t.encode("a photo of a cat", 0);
        assert_eq!(ids0[7], 0);
    }

    #[test]
    fn pony_score_tags_tokenize() {
        let t = ClipTokenizer::new();
        let ids = t.encode_raw("score_9, score_8_up");
        assert!(!ids.is_empty());
        let long = "word ".repeat(200);
        let ids = t.encode(&long, 0);
        assert_eq!(ids.len(), 77);
        assert_eq!(ids[76], EOS);
    }
}
