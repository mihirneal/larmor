//! Cross-validation folds identical to sklearn's `KFold` and `StratifiedKFold`.

use crate::rng::RandomState;
use thiserror::Error;

/// One train/test split; both index lists are sorted ascending, like sklearn's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// Why folds could not be built, or a cross-validated fit cannot run on them.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FoldError {
    #[error("cannot split {n_samples} samples into {n_splits} folds")]
    TooFewSamples { n_samples: usize, n_splits: usize },
    #[error(
        "{n_splits}-fold stratified cross-validation needs at least {n_splits} samples of each class; got \
         {negatives} negative and {positives} positive (scikit-learn raises here)"
    )]
    TooFewInClass { negatives: usize, positives: usize, n_splits: usize },
    #[error("the features contain NaN or infinite values (scikit-learn raises \"Input X contains NaN\")")]
    NonFinite,
    #[error("n_splits={n_splits} is greater than the number of members in every class")]
    TooFewMembers { n_splits: usize },
    #[error("fold column must hold every fold id from 0 to {max}, but fold {missing} is empty")]
    EmptyFold { max: usize, missing: usize },
}

/// `sklearn.model_selection.KFold(n_splits, shuffle, random_state).split(range(n_samples))`.
///
/// Mirrors `_split.py::KFold._iter_test_indices`: optionally shuffle `arange(n)` with
/// `RandomState(seed)`, then cut it into contiguous blocks, the first `n % n_splits` one longer.
pub fn kfold(n_samples: usize, n_splits: usize, shuffle_seed: Option<u32>) -> Result<Vec<Fold>, FoldError> {
    if n_splits < 2 || n_splits > n_samples {
        return Err(FoldError::TooFewSamples { n_samples, n_splits });
    }
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if let Some(seed) = shuffle_seed {
        RandomState::new(seed).shuffle(&mut indices);
    }
    let mut fold_of = vec![0usize; n_samples];
    let mut start = 0;
    for fold in 0..n_splits {
        let size = n_samples / n_splits + usize::from(fold < n_samples % n_splits);
        for &index in &indices[start..start + size] {
            fold_of[index] = fold;
        }
        start += size;
    }
    // sklearn turns each test block into a mask, so the yielded indices come out sorted
    Ok(folds_from_assignment(&fold_of, n_splits))
}

/// `sklearn.model_selection.StratifiedKFold(n_splits, shuffle, random_state).split(X, labels)`.
///
/// Mirrors `_split.py::StratifiedKFold._make_test_folds`: classes are numbered by first appearance,
/// each class's fold ids are allotted round robin over the sorted labels, and with shuffling each class's
/// block of fold ids is shuffled in turn by the same `RandomState`.
pub fn stratified_kfold<L: PartialEq + Copy>(
    labels: &[L],
    n_splits: usize,
    shuffle_seed: Option<u32>,
) -> Result<Vec<Fold>, FoldError> {
    let n_samples = labels.len();
    if n_splits < 2 || n_splits > n_samples {
        return Err(FoldError::TooFewSamples { n_samples, n_splits });
    }
    let encoded = encode_by_first_appearance(labels);
    let n_classes = encoded.iter().max().map_or(0, |&m| m + 1);
    let mut counts = vec![0usize; n_classes];
    for &class in &encoded {
        counts[class] += 1;
    }
    if counts.iter().all(|&count| n_splits > count) {
        return Err(FoldError::TooFewMembers { n_splits });
    }

    // allocation[class][fold]: how many samples of each class go to each test fold
    let mut sorted = encoded.clone();
    sorted.sort_unstable();
    let mut allocation = vec![vec![0usize; n_splits]; n_classes];
    for (position, &class) in sorted.iter().enumerate() {
        allocation[class][position % n_splits] += 1;
    }

    let mut rng = shuffle_seed.map(RandomState::new);
    let mut test_fold = vec![0usize; n_samples];
    for (class, per_fold) in allocation.iter().enumerate() {
        let mut folds_for_class: Vec<usize> =
            per_fold.iter().enumerate().flat_map(|(fold, &count)| std::iter::repeat_n(fold, count)).collect();
        if let Some(rng) = rng.as_mut() {
            rng.shuffle(&mut folds_for_class);
        }
        let members = (0..n_samples).filter(|&i| encoded[i] == class);
        for (sample, fold) in members.zip(folds_for_class) {
            test_fold[sample] = fold;
        }
    }
    Ok(folds_from_assignment(&test_fold, n_splits))
}

/// Folds from an explicit fold id per sample (the `fold` column of `labels.tsv`, 0-based).
pub fn folds_from_column(fold_ids: &[usize]) -> Result<Vec<Fold>, FoldError> {
    let n_splits = fold_ids.iter().max().map_or(0, |&m| m + 1);
    if n_splits < 2 {
        return Err(FoldError::TooFewSamples { n_samples: fold_ids.len(), n_splits });
    }
    let folds = folds_from_assignment(fold_ids, n_splits);
    if let Some(missing) = folds.iter().position(|fold| fold.test.is_empty()) {
        return Err(FoldError::EmptyFold { max: n_splits - 1, missing });
    }
    Ok(folds)
}

/// The fold id of every sample, the inverse of a list of folds.
pub fn fold_assignment(folds: &[Fold], n_samples: usize) -> Vec<usize> {
    let mut fold_of = vec![0usize; n_samples];
    for (fold, split) in folds.iter().enumerate() {
        for &index in &split.test {
            fold_of[index] = fold;
        }
    }
    fold_of
}

fn folds_from_assignment(test_fold: &[usize], n_splits: usize) -> Vec<Fold> {
    (0..n_splits)
        .map(|fold| {
            let (test, train): (Vec<usize>, Vec<usize>) =
                (0..test_fold.len()).partition(|&i| test_fold[i] == fold);
            Fold { train, test }
        })
        .collect()
}

/// Class ids numbered by order of first appearance, like `np.unique(y_idx, return_inverse=True)`.
fn encode_by_first_appearance<L: PartialEq + Copy>(labels: &[L]) -> Vec<usize> {
    let mut seen: Vec<L> = Vec::new();
    labels
        .iter()
        .map(|label| match seen.iter().position(|s| s == label) {
            Some(class) => class,
            None => {
                seen.push(*label);
                seen.len() - 1
            }
        })
        .collect()
}
