//! Deterministic inference for the compact `RMPOS001` UPOS model format.
//!
//! Feature strings are deliberately part of this module's wire contract.  A
//! trainer must emit precisely these UTF-8 strings before FNV-1a hashing:
//! `W=<form>`, `L=<ASCII-lower form>`, `S=<shape>`, `P1=` through `P4=`,
//! `U1=` through `U4=`, `PL=`, `PS=`, `NL=`, `NS=`, `PC=`, `CN=`, and each
//! true `F=<flag>`. Prefixes and suffixes are Unicode-scalar slices of the
//! ASCII-lowered form. Shape
//! maps uppercase letters to `X`, lowercase letters to `x`, numeric characters
//! to `d`, and retains every other scalar. Context pairs use a literal `|`
//! separator; boundary values are the literal `<BOS>` and `<EOS>`.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 8] = b"RMPOS001";
const VERSION: u16 = 1;
const TAG_COUNT: usize = 17;
const FEATURE_LIMIT: u16 = 22;
const TAG_MASK: u32 = (1 << TAG_COUNT) - 1;
const TAGS: [&str; TAG_COUNT] = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X",
];
const PAIR_SEPARATOR: char = '|';

pub struct LinearPosModel {
    model_id: String,
    tokenizer_id: String,
    artifact_bytes: usize,
    artifact_sha256: String,
    bucket_mask: usize,
    biases: [i16; TAG_COUNT],
    direct: Vec<(u64, u8)>,
    candidates: Vec<(u64, u32)>,
    weights: Vec<i8>,
}

impl LinearPosModel {
    pub fn from_artifact(bytes: Vec<u8>) -> Result<Self, String> {
        let artifact_bytes = bytes.len();
        let artifact_sha256 = sha256_hex(&bytes);
        let mut reader = Reader::new(&bytes);

        if reader.take(8)? != MAGIC {
            return Err("invalid POS model magic; expected RMPOS001".to_string());
        }
        if reader.u16()? != VERSION {
            return Err("unsupported POS model version; expected 1".to_string());
        }
        if reader.u16()? != TAG_COUNT as u16 {
            return Err("invalid POS model tag count; expected 17".to_string());
        }
        let bucket_count = reader.u32()? as usize;
        if bucket_count == 0 || !bucket_count.is_power_of_two() {
            return Err("POS model bucket_count must be a non-zero power of two".to_string());
        }
        if reader.u16()? != FEATURE_LIMIT {
            return Err("invalid POS model feature_limit; expected 22".to_string());
        }
        if reader.u16()? != 0 {
            return Err("POS model reserved header field must be zero".to_string());
        }
        let model_id_len = reader.u16()? as usize;
        let tokenizer_id_len = reader.u16()? as usize;
        let direct_count = reader.u32()? as usize;
        let candidate_count = reader.u32()? as usize;
        let weights_len = reader.u32()? as usize;
        let expected_weights_len = bucket_count
            .checked_mul(TAG_COUNT)
            .ok_or_else(|| "POS model bucket_count is too large".to_string())?;
        if weights_len != expected_weights_len {
            return Err(format!(
                "invalid POS model weights_len; expected {expected_weights_len}, got {weights_len}"
            ));
        }

        let model_id = utf8_field(reader.take(model_id_len)?, "model_id")?;
        let tokenizer_id = utf8_field(reader.take(tokenizer_id_len)?, "tokenizer_id")?;
        if model_id.is_empty() || tokenizer_id.is_empty() {
            return Err("POS model model_id and tokenizer_id must be non-empty".to_string());
        }

        let mut biases = [0_i16; TAG_COUNT];
        for bias in &mut biases {
            *bias = reader.i16()?;
        }

        if direct_count > reader.remaining() / 9 {
            return Err("POS model direct table is truncated".to_string());
        }
        let mut direct = Vec::with_capacity(direct_count);
        let mut previous = None;
        for _ in 0..direct_count {
            let form_hash = reader.u64()?;
            let tag = reader.u8()?;
            if tag as usize >= TAG_COUNT {
                return Err("POS model direct entry has an invalid tag index".to_string());
            }
            require_strictly_sorted(previous, form_hash, "direct")?;
            previous = Some(form_hash);
            direct.push((form_hash, tag));
        }

        if candidate_count > reader.remaining() / 12 {
            return Err("POS model candidate table is truncated".to_string());
        }
        let mut candidates = Vec::with_capacity(candidate_count);
        previous = None;
        for _ in 0..candidate_count {
            let form_hash = reader.u64()?;
            let mask = reader.u32()?;
            if mask == 0 || mask & !TAG_MASK != 0 {
                return Err("POS model candidate entry has an invalid tag mask".to_string());
            }
            require_strictly_sorted(previous, form_hash, "candidate")?;
            previous = Some(form_hash);
            candidates.push((form_hash, mask));
        }
        for (form_hash, _) in &direct {
            if candidates
                .binary_search_by_key(form_hash, |(candidate_hash, _)| *candidate_hash)
                .is_ok()
            {
                return Err(
                    "POS model direct and candidate tables must not share a form hash".to_string(),
                );
            }
        }

        let weights = reader
            .take(weights_len)?
            .iter()
            .map(|byte| *byte as i8)
            .collect();
        if !reader.is_empty() {
            return Err("POS model has trailing bytes".to_string());
        }

        Ok(Self {
            model_id,
            tokenizer_id,
            artifact_bytes,
            artifact_sha256,
            bucket_mask: bucket_count - 1,
            biases,
            direct,
            candidates,
            weights,
        })
    }

    pub fn tag_documents(
        &self,
        documents: &[Vec<Vec<String>>],
    ) -> Result<Vec<Vec<Vec<String>>>, String> {
        documents
            .iter()
            .enumerate()
            .map(|(document_index, document)| {
                document
                    .iter()
                    .enumerate()
                    .map(|(sentence_index, sentence)| {
                        if sentence.iter().any(String::is_empty) {
                            return Err(format!(
                                "empty form at document {document_index}, sentence {sentence_index}"
                            ));
                        }
                        Ok(sentence
                            .iter()
                            .enumerate()
                            .map(|(index, _)| self.tag_token(sentence, index).to_string())
                            .collect())
                    })
                    .collect()
            })
            .collect()
    }

    fn tag_token(&self, sentence: &[String], index: usize) -> &'static str {
        let form = &sentence[index];
        if let Some(tag) = rule_tag(form) {
            return TAGS[tag];
        }

        let form_hash = fnv1a(form.as_bytes());
        if let Ok(index) = self
            .direct
            .binary_search_by_key(&form_hash, |(entry_hash, _)| *entry_hash)
        {
            return TAGS[self.direct[index].1 as usize];
        }
        let mask = self
            .candidates
            .binary_search_by_key(&form_hash, |(entry_hash, _)| *entry_hash)
            .map(|index| self.candidates[index].1)
            .unwrap_or(TAG_MASK);

        let mut scores = self.biases.map(i32::from);
        for feature in feature_strings(sentence, index) {
            let bucket = (fnv1a(feature.as_bytes()) as usize) & self.bucket_mask;
            let weights = &self.weights[bucket * TAG_COUNT..(bucket + 1) * TAG_COUNT];
            for (tag, weight) in weights.iter().enumerate() {
                if mask & (1 << tag) != 0 {
                    scores[tag] += i32::from(*weight);
                }
            }
        }

        let mut best = mask.trailing_zeros() as usize;
        for tag in best + 1..TAG_COUNT {
            if mask & (1 << tag) != 0 && scores[tag] > scores[best] {
                best = tag;
            }
        }
        TAGS[best]
    }
}

impl LinearPosModel {
    pub fn new(path: &str) -> Result<Self, String> {
        let artifact =
            fs::read(Path::new(path)).map_err(|error| format!("cannot read POS model: {error}"))?;
        Self::from_artifact(artifact)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn tokenizer_id(&self) -> &str {
        &self.tokenizer_id
    }

    pub fn artifact_bytes(&self) -> usize {
        self.artifact_bytes
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// Tags an explicitly bounded document/sentence/form tensor without
    /// changing any of its boundaries.
    pub fn tag(&self, documents: Vec<Vec<Vec<String>>>) -> Result<Vec<Vec<Vec<String>>>, String> {
        self.tag_documents(&documents)
    }
}

fn utf8_field(bytes: &[u8], name: &str) -> Result<String, String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| format!("POS model {name} is not valid UTF-8"))
}

fn require_strictly_sorted(previous: Option<u64>, value: u64, table: &str) -> Result<(), String> {
    if previous.is_some_and(|previous| value <= previous) {
        return Err(format!(
            "POS model {table} entries must be strictly sorted and unique"
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn ascii_lower(value: &str) -> String {
    value.to_ascii_lowercase()
}

fn scalar_prefix(value: &str, count: usize) -> String {
    value.chars().take(count).collect()
}

fn scalar_suffix(value: &str, count: usize) -> String {
    let chars: Vec<_> = value.chars().collect();
    chars[chars.len().saturating_sub(count)..].iter().collect()
}

fn shape(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_uppercase() {
                'X'
            } else if character.is_lowercase() || character.is_alphabetic() {
                'x'
            } else if character.is_numeric() {
                'd'
            } else {
                character
            }
        })
        .collect()
}

fn is_title(value: &str) -> bool {
    let mut letters = value.chars().filter(|character| character.is_alphabetic());
    let Some(first) = letters.next() else {
        return false;
    };
    first.is_uppercase()
        && letters.clone().next().is_some()
        && letters.all(|character| character.is_lowercase())
}

fn feature_strings(sentence: &[String], index: usize) -> Vec<String> {
    let form = &sentence[index];
    let lower = ascii_lower(form);
    let previous = index.checked_sub(1).map(|index| &sentence[index]);
    let next = sentence.get(index + 1);
    let previous_lower = previous.map_or_else(|| "<BOS>".to_string(), |value| ascii_lower(value));
    let next_lower = next.map_or_else(|| "<EOS>".to_string(), |value| ascii_lower(value));
    let previous_shape = previous.map_or_else(|| "<BOS>".to_string(), |value| shape(value));
    let next_shape = next.map_or_else(|| "<EOS>".to_string(), |value| shape(value));

    let mut features = Vec::with_capacity(FEATURE_LIMIT as usize);
    features.push(format!("W={form}"));
    features.push(format!("L={lower}"));
    features.push(format!("S={}", shape(form)));
    for count in 1..=4 {
        features.push(format!("P{count}={}", scalar_prefix(&lower, count)));
    }
    for count in 1..=4 {
        features.push(format!("U{count}={}", scalar_suffix(&lower, count)));
    }
    features.push(format!("PL={previous_lower}"));
    features.push(format!("PS={previous_shape}"));
    features.push(format!("NL={next_lower}"));
    features.push(format!("NS={next_shape}"));
    features.push(format!("PC={previous_lower}{PAIR_SEPARATOR}{lower}"));
    features.push(format!("CN={lower}{PAIR_SEPARATOR}{next_lower}"));
    if form.chars().any(|character| character.is_uppercase()) {
        features.push("F=upper".to_string());
    }
    if is_title(form) {
        features.push("F=title".to_string());
    }
    if form.chars().any(|character| character.is_numeric()) {
        features.push("F=digit".to_string());
    }
    if form.chars().any(|character| {
        matches!(
            character,
            '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}'
        )
    }) {
        features.push("F=hyphen".to_string());
    }
    if form
        .chars()
        .any(|character| matches!(character, '\'' | '\u{2019}'))
    {
        features.push("F=apostrophe".to_string());
    }
    assert!(features.len() <= FEATURE_LIMIT as usize);
    features
}

fn rule_tag(form: &str) -> Option<usize> {
    if form.chars().all(is_punctuation) {
        Some(12) // PUNCT
    } else if is_number_like(form) {
        Some(8) // NUM
    } else if form.chars().all(is_symbol) {
        Some(14) // SYM
    } else {
        None
    }
}

fn is_punctuation(character: char) -> bool {
    matches!(
        character,
        '!' | '"'
            | '\''
            | '('
            | ')'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '?'
            | '['
            | ']'
            | '_'
            | '{'
            | '}'
            | '\\'
            | '\u{00ab}'
            | '\u{00bb}'
            | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2015}'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{201a}'
            | '\u{201b}'
            | '\u{201c}'
            | '\u{201d}'
            | '\u{201e}'
            | '\u{201f}'
            | '\u{2026}'
    )
}

fn is_number_like(form: &str) -> bool {
    let mut seen_digit = false;
    for character in form.chars() {
        if character.is_numeric() {
            seen_digit = true;
        } else if !matches!(
            character,
            '+' | '-' | '\u{2212}' | '.' | ',' | '_' | '/' | '%' | '\u{2030}' | 'e' | 'E'
        ) {
            return false;
        }
    }
    seen_digit
}

fn is_symbol(character: char) -> bool {
    !character.is_whitespace()
        && !character.is_alphanumeric()
        && !is_punctuation(character)
        && !character.is_control()
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| "POS model length overflow".to_string())?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "POS model is truncated".to_string())?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("fixed length"),
        ))
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(
            self.take(2)?.try_into().expect("fixed length"),
        ))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed length"),
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed length"),
        ))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(
        biases: [i16; TAG_COUNT],
        direct: &[(u64, u8)],
        candidates: &[(u64, u32)],
        weights: &[i8],
    ) -> Vec<u8> {
        let bucket_count = weights.len() / TAG_COUNT;
        assert!(bucket_count.is_power_of_two());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(TAG_COUNT as u16).to_le_bytes());
        bytes.extend_from_slice(&(bucket_count as u32).to_le_bytes());
        bytes.extend_from_slice(&FEATURE_LIMIT.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&5_u16.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&(direct.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(candidates.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(weights.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"model");
        bytes.extend_from_slice(b"tok1");
        for bias in biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        for (hash, tag) in direct {
            bytes.extend_from_slice(&hash.to_le_bytes());
            bytes.push(*tag);
        }
        for (hash, mask) in candidates {
            bytes.extend_from_slice(&hash.to_le_bytes());
            bytes.extend_from_slice(&mask.to_le_bytes());
        }
        bytes.extend(weights.iter().map(|weight| *weight as u8));
        bytes
    }

    fn empty_weights() -> Vec<i8> {
        vec![0; TAG_COUNT]
    }

    #[test]
    fn parses_golden_artifact_and_reports_sha256() {
        let bytes = artifact([0; TAG_COUNT], &[], &[], &empty_weights());
        let model = LinearPosModel::from_artifact(bytes.clone()).unwrap();
        assert_eq!(model.model_id, "model");
        assert_eq!(model.tokenizer_id, "tok1");
        assert_eq!(model.artifact_bytes, bytes.len());
        assert_eq!(model.artifact_sha256, sha256_hex(&bytes));
        assert_eq!(
            model
                .tag_documents(&[vec![vec!["word".to_string()]]])
                .unwrap(),
            [[["ADJ".to_string()]]]
        );
    }

    #[test]
    fn parser_rejects_bad_header_order_and_collisions() {
        let mut bad_magic = artifact([0; TAG_COUNT], &[], &[], &empty_weights());
        bad_magic[0] = b'X';
        assert!(LinearPosModel::from_artifact(bad_magic).is_err());

        let duplicate = fnv1a(b"same");
        let unsorted = artifact(
            [0; TAG_COUNT],
            &[(duplicate, 0), (duplicate, 1)],
            &[],
            &empty_weights(),
        );
        assert!(LinearPosModel::from_artifact(unsorted).is_err());

        let shared = fnv1a(b"shared");
        let colliding = artifact(
            [0; TAG_COUNT],
            &[(shared, 0)],
            &[(shared, 1)],
            &empty_weights(),
        );
        assert!(LinearPosModel::from_artifact(colliding).is_err());
    }

    #[test]
    fn direct_candidate_and_lower_index_ties_are_deterministic() {
        let direct_hash = fnv1a(b"direct");
        let candidate_hash = fnv1a(b"candidate");
        let model = LinearPosModel::from_artifact(artifact(
            [0; TAG_COUNT],
            &[(direct_hash, 15)],
            &[(candidate_hash, (1 << 7) | (1 << 15))],
            &empty_weights(),
        ))
        .unwrap();
        let sentence = vec![
            "direct".to_string(),
            "candidate".to_string(),
            "unknown".to_string(),
        ];
        assert_eq!(model.tag_token(&sentence, 0), "VERB");
        assert_eq!(model.tag_token(&sentence, 1), "NOUN");
        assert_eq!(model.tag_token(&sentence, 2), "ADJ");
    }

    #[test]
    fn rules_precede_model_tables() {
        let mut direct = vec![
            (fnv1a(b"!"), 15),
            (fnv1a(b"$"), 15),
            (fnv1a(b"1,200.5%"), 15),
        ];
        direct.sort_unstable_by_key(|(hash, _)| *hash);
        let model =
            LinearPosModel::from_artifact(artifact([0; TAG_COUNT], &direct, &[], &empty_weights()))
                .unwrap();
        let sentence = vec!["!".to_string(), "1,200.5%".to_string(), "$".to_string()];
        assert_eq!(model.tag_token(&sentence, 0), "PUNCT");
        assert_eq!(model.tag_token(&sentence, 1), "NUM");
        assert_eq!(model.tag_token(&sentence, 2), "SYM");
    }

    #[test]
    fn golden_feature_strings_are_bounded_and_unicode_scalar_based() {
        let sentence = vec!["Hi".to_string(), "FOO!".to_string(), "\u{00e9}".to_string()];
        assert_eq!(
            feature_strings(&sentence, 1),
            vec![
                "W=FOO!",
                "L=foo!",
                "S=XXX!",
                "P1=f",
                "P2=fo",
                "P3=foo",
                "P4=foo!",
                "U1=!",
                "U2=o!",
                "U3=oo!",
                "U4=foo!",
                "PL=hi",
                "PS=Xx",
                "NL=\u{00e9}",
                "NS=x",
                "PC=hi|foo!",
                "CN=foo!|\u{00e9}",
                "F=upper",
            ]
        );
        assert_eq!(scalar_prefix("\u{00e9}x", 1), "\u{00e9}");
        assert_eq!(scalar_suffix("x\u{00e9}", 1), "\u{00e9}");
    }
}
