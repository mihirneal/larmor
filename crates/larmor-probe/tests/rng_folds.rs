//! The RNG ports and folds against draws saved from numpy, torch and sklearn by
//! `tools/reference/probe_rng.py`.

use larmor_probe::folds::{kfold, stratified_kfold};
use larmor_probe::rng::{Generator, RandomState, SeedSequence, torch_randperm};
use serde_json::Value;

fn reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/rng_folds.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn as_u128(value: &Value) -> u128 {
    value.as_str().unwrap().parse().unwrap()
}

fn seed_sequence(case: &Value) -> SeedSequence {
    let words: Vec<u32> =
        case["seed_words"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    SeedSequence::from_words(&words)
}

fn as_usizes(value: &Value) -> Vec<usize> {
    value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
}

#[test]
fn mt19937_raw_stream_matches_numpy() {
    let data = reference();
    for (seed, expected) in data["mt19937_raw"].as_object().unwrap() {
        let mut rng = RandomState::new(seed.parse().unwrap());
        for (i, value) in expected.as_array().unwrap().iter().enumerate() {
            assert_eq!(u64::from(rng.next_u32()), value.as_u64().unwrap(), "seed {seed}, draw {i}");
        }
    }
}

#[test]
fn legacy_shuffle_matches_numpy() {
    let data = reference();
    let cases = data["shuffle"].as_array().unwrap();
    for case in cases {
        let n = case["n"].as_u64().unwrap() as usize;
        let mut values: Vec<usize> = (0..n).collect();
        RandomState::new(case["seed"].as_u64().unwrap() as u32).shuffle(&mut values);
        assert_eq!(values, as_usizes(&case["permutation"]), "seed {}, n {n}", case["seed"]);
    }
    assert!(cases.len() > 80);
}

#[test]
fn seed_sequence_matches_numpy() {
    let data = reference();
    for case in data["seed_sequence"].as_array().unwrap() {
        let state = seed_sequence(case).generate_state_u64(4);
        let expected: Vec<u64> =
            case["state"].as_array().unwrap().iter().map(|v| as_u128(v) as u64).collect();
        assert_eq!(state, expected, "seed {}", case["seed"]);
    }
}

#[test]
fn pcg64_raw_stream_matches_numpy() {
    let data = reference();
    for (seed, expected) in data["pcg64_raw"].as_object().unwrap() {
        let mut rng = Generator::default_rng(seed.parse().unwrap());
        for value in expected.as_array().unwrap() {
            assert_eq!(u128::from(rng.next_raw()), as_u128(value), "seed {seed}");
        }
    }
}

#[test]
fn integers_match_numpy_across_calls() {
    let data = reference();
    let cases = data["integers"].as_array().unwrap();
    for case in cases {
        let high = as_u128(&case["high"]) as i64;
        let mut rng = Generator::from_seed_sequence(&seed_sequence(case));
        for draws in case["draws"].as_array().unwrap() {
            let expected: Vec<i64> = draws.as_array().unwrap().iter().map(|v| as_u128(v) as i64).collect();
            assert_eq!(rng.integers(0, high, expected.len()), expected, "seed {}, high {high}", case["seed"]);
        }
    }
    assert!(cases.len() > 90);
}

#[test]
fn randperm_matches_torch() {
    let data = reference();
    for case in data["randperm"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let expected: Vec<i64> =
            case["permutation"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(
            torch_randperm(n, case["seed"].as_u64().unwrap()),
            expected,
            "seed {}, n {n}",
            case["seed"]
        );
    }
}

#[test]
fn kfold_matches_sklearn() {
    let data = reference();
    for case in data["kfold"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let n_splits = case["n_splits"].as_u64().unwrap() as usize;
        let seed = case["seed"].as_u64().map(|s| s as u32);
        let seed = if case["shuffle"].as_bool().unwrap() { seed } else { None };
        let folds = kfold(n, n_splits, seed).unwrap();
        let tests: Vec<Vec<usize>> = case["tests"].as_array().unwrap().iter().map(as_usizes).collect();
        let ours: Vec<Vec<usize>> = folds.iter().map(|f| f.test.clone()).collect();
        assert_eq!(ours, tests, "n {n}, n_splits {n_splits}, seed {seed:?}");
        for fold in &folds {
            assert_eq!(fold.train.len() + fold.test.len(), n);
        }
    }
}

#[test]
fn stratified_kfold_matches_sklearn() {
    let data = reference();
    for case in data["stratified_kfold"].as_array().unwrap() {
        let labels: Vec<i64> =
            case["labels"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let n_splits = case["n_splits"].as_u64().unwrap() as usize;
        let seed = case["seed"].as_u64().map(|s| s as u32);
        let seed = if case["shuffle"].as_bool().unwrap() { seed } else { None };
        let folds = stratified_kfold(&labels, n_splits, seed).unwrap();
        let tests: Vec<Vec<usize>> = case["tests"].as_array().unwrap().iter().map(as_usizes).collect();
        let ours: Vec<Vec<usize>> = folds.iter().map(|f| f.test.clone()).collect();
        assert_eq!(ours, tests, "labels {labels:?}, n_splits {n_splits}, seed {seed:?}");
    }
}
