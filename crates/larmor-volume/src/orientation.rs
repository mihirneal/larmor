//! Voxel-axis orientation: which world axis each array axis runs along, and reorienting to RAS+.

use ndarray::{ArrayView3, Axis};
use thiserror::Error;

use crate::affine::Affine;
use crate::volume::Volume;

/// Why an affine has no well-defined voxel orientation.
#[derive(Debug, Error, PartialEq)]
pub enum OrientationError {
    /// The 3×3 block is rank-deficient, so some voxel axis has no world direction.
    #[error("affine is degenerate (rank-deficient rotation/zoom block), so its orientation is undefined")]
    Degenerate,
}

/// Where one voxel axis points: the world axis (0 = x/R, 1 = y/A, 2 = z/S) it is closest to,
/// and whether it runs against that axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisOrientation {
    /// Index of the closest world axis.
    pub axis: usize,
    /// True when increasing the voxel index moves towards L, P or I.
    pub flip: bool,
}

/// One [`AxisOrientation`] per voxel axis: nibabel's `ornt` array of `(axis, ±1)` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation(pub [AxisOrientation; 3]);

impl Orientation {
    /// The orientation of an array that is already RAS+ (`[[0, 1], [1, 1], [2, 1]]`).
    pub const RAS: Orientation = Orientation([
        AxisOrientation { axis: 0, flip: false },
        AxisOrientation { axis: 1, flip: false },
        AxisOrientation { axis: 2, flip: false },
    ]);

    /// Axis codes such as `"RAS"` or `"LPI"`.
    ///
    /// Mirrors `nibabel.orientations.ornt2axcodes` with the default labels.
    pub fn axcodes(&self) -> String {
        const LABELS: [[char; 2]; 3] = [['R', 'L'], ['A', 'P'], ['S', 'I']];
        self.0.iter().map(|a| LABELS[a.axis][a.flip as usize]).collect()
    }

    /// The axis order of the reoriented array: result axis `i` is input axis `order[i]`.
    ///
    /// This is `np.argsort(ornt[:, 0])` from `apply_orientation`.
    pub fn transpose_order(&self) -> [usize; 3] {
        let mut order = [0usize; 3];
        for (input_axis, a) in self.0.iter().enumerate() {
            order[a.axis] = input_axis;
        }
        order
    }

    /// The shape an array of `shape` has after [`Orientation::apply`].
    pub fn reoriented_shape(&self, shape: [usize; 3]) -> [usize; 3] {
        self.transpose_order().map(|axis| shape[axis])
    }

    /// Flips and transposes an array from this orientation to RAS+, without copying.
    ///
    /// Mirrors `nibabel.orientations.apply_orientation(arr, ornt)`.
    pub fn apply<'a, T>(&self, view: ArrayView3<'a, T>) -> ArrayView3<'a, T> {
        let mut view = view;
        for (axis, a) in self.0.iter().enumerate() {
            if a.flip {
                view.invert_axis(Axis(axis));
            }
        }
        view.permuted_axes(self.transpose_order())
    }

    /// The affine that maps reoriented voxel indices back to the original ones.
    ///
    /// Mirrors `nibabel.orientations.inv_ornt_aff(ornt, shape)`.
    pub fn inv_ornt_aff(&self, shape: [usize; 3]) -> Affine {
        // undo_flip @ undo_reorder, written out: every product is exact
        let mut out = Affine([[0.0; 4]; 4]);
        out.0[3][3] = 1.0;
        for (row, a) in self.0.iter().enumerate() {
            let sign = if a.flip { -1.0 } else { 1.0 };
            let center = -((shape[row] as f64) - 1.0) / 2.0;
            out.0[row][a.axis] = sign;
            out.0[row][3] = sign * center - center;
        }
        out
    }

    /// The affine of the array after [`Orientation::apply`].
    ///
    /// Mirrors `affine.dot(inv_ornt_aff(ornt, shape))` in `SpatialImage.as_reoriented`.
    pub fn reoriented_affine(&self, affine: &Affine, shape: [usize; 3]) -> Affine {
        affine.dot(&self.inv_ornt_aff(shape))
    }
}

/// The orientation of each voxel axis relative to RAS+ world axes.
///
/// Mirrors `nibabel.orientations.io_orientation(affine)`: normalise the columns, take the closest
/// orthogonal matrix (the polar factor `P @ Qs` of the SVD), then assign world axes greedily,
/// strongest voxel axis first (stable on ties), each to its largest-magnitude row (first on ties).
/// The polar factor comes from an eigen-decomposition rather than LAPACK's SVD, so it can differ
/// in the last bits; that could only matter for an affine exactly between two axis assignments
/// (exact 45° cases agree with nibabel in the tests).
pub fn io_orientation(affine: &Affine) -> Result<Orientation, OrientationError> {
    let rzs = affine.rzs();
    let zooms = affine.voxel_sizes().map(|z| if z == 0.0 { 1.0 } else { z });
    let rs: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| rzs[i][j] / zooms[j]));
    let mut r = polar_factor(rs).ok_or(OrientationError::Degenerate)?;

    let strength: [f64; 3] = std::array::from_fn(|col| {
        (0..3).map(|row| -(r[row][col] * r[row][col])).fold(f64::INFINITY, f64::min)
    });
    let mut in_axes = [0usize, 1, 2];
    in_axes.sort_by(|&a, &b| strength[a].total_cmp(&strength[b]));

    let mut ornt = [None; 3];
    for in_axis in in_axes {
        let col: [f64; 3] = std::array::from_fn(|row| r[row][in_axis]);
        // np.allclose(col, 0)
        if col.iter().all(|v| v.abs() <= 1e-8) {
            continue;
        }
        let mut out_axis = 0;
        for row in 1..3 {
            if col[row].abs() > col[out_axis].abs() {
                out_axis = row;
            }
        }
        ornt[in_axis] = Some(AxisOrientation { axis: out_axis, flip: col[out_axis] < 0.0 });
        r[out_axis] = [0.0; 3];
    }
    match ornt {
        [Some(a), Some(b), Some(c)] => Ok(Orientation([a, b, c])),
        _ => Err(OrientationError::Degenerate),
    }
}

/// Axis codes of an affine, such as `"RAS"`.
///
/// Mirrors `nibabel.aff2axcodes(affine)`.
pub fn aff2axcodes(affine: &Affine) -> Result<String, OrientationError> {
    Ok(io_orientation(affine)?.axcodes())
}

impl<T: Clone> Volume<T> {
    /// This volume reoriented so its voxel axes run closest to R, A and S.
    ///
    /// Mirrors `nibabel.as_closest_canonical(img)`: an image that is already RAS+ comes back
    /// unchanged; otherwise the data are flipped and transposed (into a new C-order array) and the
    /// affine is adjusted to match.
    pub fn as_closest_canonical(&self) -> Result<Volume<T>, OrientationError> {
        let ornt = io_orientation(&self.affine)?;
        if ornt == Orientation::RAS {
            return Ok(self.clone());
        }
        let data = ornt.apply(self.data.view()).as_standard_layout().into_owned();
        let affine = ornt.reoriented_affine(&self.affine, self.shape());
        Ok(Volume { zooms: self.zooms_for(&affine), data, affine })
    }
}

/// The orthogonal matrix closest to `m`, or `None` if `m` is (numerically) singular.
///
/// `m (mᵀm)^(-1/2)` equals the SVD's `P @ Qs`; the rank test is numpy's
/// `S > S.max() * 3 * eps` from `io_orientation`.
pub(crate) fn polar_factor(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut gram = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            gram[i][j] = (0..3).map(|k| m[k][i] * m[k][j]).sum();
        }
    }
    let (eigenvalues, v) = symmetric_eigen(gram);
    let singular = eigenvalues.map(|l| l.max(0.0).sqrt());
    let tol = singular.iter().cloned().fold(0.0, f64::max) * 3.0 * f64::EPSILON;
    if singular.iter().any(|&s| s <= tol) {
        return None;
    }
    // inverse square root of the Gram matrix: V diag(1/s) Vᵀ
    let mut inv_sqrt = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv_sqrt[i][j] = (0..3).map(|k| v[i][k] * v[j][k] / singular[k]).sum();
        }
    }
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| m[i][k] * inv_sqrt[k][j]).sum();
        }
    }
    Some(r)
}

/// Eigenvalues and eigenvectors (as columns) of a symmetric 3×3 matrix, by cyclic Jacobi sweeps.
fn symmetric_eigen(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let scale = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-30 * scale {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine(rzs: [[f64; 3]; 3]) -> Affine {
        Affine::from_matvec(rzs, [0.0; 3])
    }

    #[test]
    fn axis_permutations_and_flips() {
        let lps = affine([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(aff2axcodes(&lps).unwrap(), "LPS");
        let psr = affine([[0.0, 0.0, 1.2], [-0.9, 0.0, 0.0], [0.0, 1.1, 0.0]]);
        assert_eq!(aff2axcodes(&psr).unwrap(), "PSR");
        assert_eq!(io_orientation(&Affine::IDENTITY).unwrap(), Orientation::RAS);
    }

    #[test]
    fn degenerate_affine_is_an_error() {
        let flat = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
        assert_eq!(io_orientation(&flat), Err(OrientationError::Degenerate));
    }

    #[test]
    fn canonical_view_matches_reoriented_affine() {
        let lps =
            Affine::from_matvec([[-2.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 3.0, 0.0]], [10.0, 20.0, 30.0]);
        let shape = [4, 5, 6];
        let ornt = io_orientation(&lps).unwrap();
        let new = ornt.reoriented_affine(&lps, shape);
        let new_shape = ornt.reoriented_shape(shape);
        // the reoriented voxel [i, j, k] must land where its source voxel did
        let data = ndarray::Array3::from_shape_fn(shape, |(i, j, k)| [i, j, k]);
        let view = ornt.apply(data.view());
        assert_eq!(view.shape(), new_shape);
        for ((i, j, k), src) in view.indexed_iter() {
            let a = new.apply([i as f64, j as f64, k as f64]);
            let b = lps.apply(src.map(|c| c as f64));
            assert_eq!(a, b);
        }
    }
}
