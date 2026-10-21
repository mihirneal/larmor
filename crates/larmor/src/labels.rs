//! `labels.tsv`: one row per scan, like a BIDS `participants.tsv`.
//!
//! ```text
//! # comment lines start with '#'
//! scan                              age   group     lesion                         fold
//! sub-01/anat/sub-01_T1w.nii.gz     34    control   sub-01/anat/sub-01_lesion.nii.gz   0
//! ```
//!
//! `scan` is a path relative to the TSV (absolute paths work too). Every other column is a label;
//! `fold` (0-based) optionally fixes the outer cross-validation folds. Empty cells and `n/a` are
//! missing values: rows missing the label being fitted are left out.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::hashing::sha256_hex;

/// A parsed label table.
#[derive(Clone, Debug)]
pub struct LabelTable {
    pub path: PathBuf,
    /// SHA-256 of the file bytes, recorded in `probe.toml`.
    pub sha256: String,
    pub columns: Vec<String>,
    pub rows: Vec<LabelRow>,
}

/// One row.
#[derive(Clone, Debug)]
pub struct LabelRow {
    /// Line number in the file, from 1.
    pub line: usize,
    /// The scan path, resolved against the TSV's folder.
    pub scan: PathBuf,
    /// The cells, one per column (the `scan` cell included), trimmed.
    pub cells: Vec<String>,
}

/// The rows that have a label, with their values.
pub struct LabelColumn<'a> {
    pub name: String,
    pub rows: Vec<&'a LabelRow>,
    pub values: Vec<String>,
    /// The `fold` ids of the same rows, when the table has a `fold` column. Ids left without rows
    /// (all their labels missing) are closed up, so `0, 2, 3` become `0, 1, 2`.
    pub folds: Option<Vec<usize>>,
    /// Rows left out because their label was missing.
    pub skipped: usize,
}

pub fn is_missing(cell: &str) -> bool {
    cell.is_empty() || cell.eq_ignore_ascii_case("n/a")
}

impl LabelTable {
    pub fn read(path: &Path) -> Result<LabelTable> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let text =
            String::from_utf8(bytes.clone()).with_context(|| format!("{} is not UTF-8", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
        let (_, header) = lines.next().with_context(|| format!("{} has no header row", path.display()))?;
        let columns: Vec<String> = header.split('\t').map(|c| c.trim().to_string()).collect();
        let scan_column = columns.iter().position(|c| c == "scan").with_context(|| {
            format!("{} has no `scan` column (columns: {})", path.display(), columns.join(", "))
        })?;
        let mut rows = Vec::new();
        for (index, line) in lines {
            let cells: Vec<String> = line.split('\t').map(|c| c.trim().to_string()).collect();
            ensure!(
                cells.len() == columns.len(),
                "{} line {}: {} cells for {} columns (cells are separated by tabs)",
                path.display(),
                index + 1,
                cells.len(),
                columns.len()
            );
            ensure!(
                !is_missing(&cells[scan_column]),
                "{} line {}: empty scan path",
                path.display(),
                index + 1
            );
            let scan = base.join(&cells[scan_column]);
            rows.push(LabelRow { line: index + 1, scan, cells });
        }
        ensure!(!rows.is_empty(), "{} has no rows", path.display());
        Ok(LabelTable { path: path.to_path_buf(), sha256: sha256_hex(&bytes), columns, rows })
    }

    fn column_index(&self, name: &str) -> Result<usize> {
        let labels: Vec<&str> =
            self.columns.iter().map(String::as_str).filter(|c| *c != "scan" && *c != "fold").collect();
        if name == "scan" || name == "fold" {
            bail!("`{name}` is not a label column; labels in {}: {}", self.path.display(), labels.join(", "));
        }
        self.columns.iter().position(|c| c == name).with_context(|| {
            format!("{} has no column {name:?}; label columns: {}", self.path.display(), labels.join(", "))
        })
    }

    /// The rows with a value in column `name`, with their `fold` ids when there is a fold column.
    pub fn column(&self, name: &str) -> Result<LabelColumn<'_>> {
        let index = self.column_index(name)?;
        let fold_index = self.columns.iter().position(|c| c == "fold");
        let (mut rows, mut values, mut folds) = (Vec::new(), Vec::new(), Vec::new());
        for row in &self.rows {
            let value = &row.cells[index];
            if is_missing(value) {
                continue;
            }
            if let Some(f) = fold_index {
                let cell = &row.cells[f];
                let fold = cell.parse::<usize>().ok().with_context(|| {
                    format!(
                        "{} line {}: fold {cell:?} is not a 0-based integer",
                        self.path.display(),
                        row.line
                    )
                })?;
                folds.push(fold);
            }
            rows.push(row);
            values.push(value.clone());
        }
        let skipped = self.rows.len() - rows.len();
        let folds = fold_index.map(|_| dense_ranks(&folds));
        Ok(LabelColumn { name: name.to_string(), rows, values, folds, skipped })
    }
}

/// Each id's rank among the distinct ids, from 0.
fn dense_ranks(ids: &[usize]) -> Vec<usize> {
    let mut distinct = ids.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    ids.iter().map(|id| distinct.binary_search(id).expect("id is in the list")).collect()
}

impl LabelColumn<'_> {
    /// The values as numbers, for regression.
    pub fn numbers(&self, table: &Path) -> Result<Vec<f64>> {
        self.rows
            .iter()
            .zip(&self.values)
            .map(|(row, value)| {
                value.parse::<f64>().ok().filter(|v| v.is_finite()).with_context(|| {
                    format!(
                        "{} line {}: {} = {value:?} is not a number",
                        table.display(),
                        row.line,
                        self.name
                    )
                })
            })
            .collect()
    }

    /// The two classes for classification, `[negative, positive]`: sorted numerically when every
    /// value is a number, else alphabetically, so `0`/`1` and `false`/`true` give positive = the
    /// second, as sklearn's `classes_[1]`.
    pub fn classes(&self) -> Result<[String; 2]> {
        let mut classes: Vec<&String> = self.values.iter().collect();
        classes.sort_by(|a, b| match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(y)) => x.total_cmp(&y),
            _ => a.cmp(b),
        });
        classes.dedup();
        match classes.as_slice() {
            [negative, positive] => Ok([(*negative).clone(), (*positive).clone()]),
            _ => bail!(
                "classification needs exactly two classes in column {}; found {}: {}",
                self.name,
                classes.len(),
                classes.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    /// The label paths for segmentation, resolved against the TSV's folder.
    pub fn paths(&self, table: &Path) -> Vec<PathBuf> {
        let base = table.parent().unwrap_or(Path::new("."));
        self.values.iter().map(|v| base.join(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> (tempfile::TempDir, LabelTable) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.tsv");
        std::fs::write(&path, text).unwrap();
        let table = LabelTable::read(&path).unwrap();
        (dir, table)
    }

    #[test]
    fn parses_columns_comments_and_missing_values() {
        let (dir, t) = table(
            "# a comment\nscan\tage\tgroup\tfold\na.nii.gz\t30\tpatient\t0\nb.nii.gz\tn/a\tcontrol\t1\n\n",
        );
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.rows[0].scan, dir.path().join("a.nii.gz"));
        let age = t.column("age").unwrap();
        assert_eq!((age.values.len(), age.skipped), (1, 1));
        assert_eq!(age.numbers(&t.path).unwrap(), vec![30.0]);
        assert_eq!(age.folds, Some(vec![0]));
        let group = t.column("group").unwrap();
        assert_eq!(group.classes().unwrap(), ["control".to_string(), "patient".to_string()]);
        assert!(t.column("height").is_err());
        assert!(t.column("fold").is_err());
    }

    #[test]
    fn folds_without_rows_are_closed_up() {
        let (_dir, t) = table("scan\tseg\tfold\na\tx.nii\t0\nb\tn/a\t1\nc\ty.nii\t2\nd\tz.nii\t0\n");
        assert_eq!(t.column("seg").unwrap().folds, Some(vec![0, 1, 0]));
    }

    #[test]
    fn the_annotated_example_parses() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/labels.tsv");
        let t = LabelTable::read(&path).unwrap();
        assert_eq!(t.rows.len(), 5);
        assert_eq!(t.column("age").unwrap().numbers(&t.path).unwrap(), vec![34.5, 61.0, 47.2, 72.0]);
        assert_eq!(t.column("group").unwrap().classes().unwrap()[1], "patient");
        assert_eq!(t.column("lesion").unwrap().folds, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn numeric_classes_sort_numerically() {
        let (_dir, t) = table("scan\tlabel\na\t10\nb\t9\nc\t10\n");
        assert_eq!(t.column("label").unwrap().classes().unwrap(), ["9".to_string(), "10".to_string()]);
    }

    #[test]
    fn rejects_ragged_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.tsv");
        std::fs::write(&path, "scan\tage\na.nii.gz\n").unwrap();
        assert!(LabelTable::read(&path).is_err());
    }
}
