//! A minimal dense row-major matrix: one row per sample, one column per feature.

use std::fmt;

/// A dense row-major `rows x cols` matrix, like a C-contiguous 2-D numpy array.
#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({} x {})", self.rows, self.cols)
    }
}

impl<T: Copy> Matrix<T> {
    /// Wraps row-major `data`; panics if its length is not `rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match {rows} x {cols}");
        Self { rows, cols, data }
    }

    /// Stacks equally long rows.
    pub fn from_rows(rows: &[Vec<T>]) -> Self {
        let cols = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|row| row.len() == cols), "rows differ in length");
        Self::new(rows.len(), cols, rows.concat())
    }

    /// Number of rows (samples).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (features).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Row `i`, mutable.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The rows in order.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |i| self.row(i))
    }

    /// Element `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.cols + j]
    }

    /// All elements, row-major.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// All elements, row-major, by value.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// The rows at `indices`, in that order (`x[indices]` in numpy).
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let mut data = Vec::with_capacity(indices.len() * self.cols);
        for &i in indices {
            data.extend_from_slice(self.row(i));
        }
        Self::new(indices.len(), self.cols, data)
    }

    /// Applies `f` to every element.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Matrix<U> {
        Matrix::new(self.rows, self.cols, self.data.iter().map(|&v| f(v)).collect())
    }
}

impl Matrix<f32> {
    /// Widens to f64, as `validate_data(..., dtype=np.float64)` does.
    pub fn to_f64(&self) -> Matrix<f64> {
        self.map(f64::from)
    }
}
