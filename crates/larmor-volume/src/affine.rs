//! 4×4 voxel-to-world affines, computed the way nibabel and numpy compute them.
//!
//! numpy on this Mac runs matrix products through Accelerate, which uses fused multiply-adds in a
//! fixed order. The products here reproduce that order, so affines built from "awkward" numbers
//! (oblique qforms, non-dyadic voxel sizes) come out bit-identical to nibabel's, not just close.

use std::fmt;
use std::ops::Range;

/// A 4×4 homogeneous affine, row-major, like a numpy `(4, 4)` float64 array.
///
/// `affine.0[row][col]`; the last row is `[0, 0, 0, 1]` for the affines nibabel produces.
#[derive(Clone, Copy, PartialEq)]
pub struct Affine(pub [[f64; 4]; 4]);

impl Affine {
    /// The identity transform (`np.eye(4)`).
    pub const IDENTITY: Affine =
        Affine([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]);

    /// Builds an affine from its rotation-zoom-shear block and translation.
    ///
    /// Mirrors `nibabel.affines.from_matvec`.
    pub fn from_matvec(rzs: [[f64; 3]; 3], translation: [f64; 3]) -> Affine {
        let mut out = Affine::IDENTITY;
        for row in 0..3 {
            out.0[row][..3].copy_from_slice(&rzs[row]);
            out.0[row][3] = translation[row];
        }
        out
    }

    /// The upper-left 3×3 block (rotations, zooms, shears).
    ///
    /// Mirrors the first half of `nibabel.affines.to_matvec`.
    pub fn rzs(&self) -> [[f64; 3]; 3] {
        let m = &self.0;
        [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]
    }

    /// The translation column.
    ///
    /// Mirrors the second half of `nibabel.affines.to_matvec`.
    pub fn translation(&self) -> [f64; 3] {
        [self.0[0][3], self.0[1][3], self.0[2][3]]
    }

    /// Matrix product `self @ other`.
    ///
    /// Mirrors numpy's `A.dot(B)` for float64 4×4 arrays.
    pub fn dot(&self, other: &Affine) -> Affine {
        let (a, b) = (&self.0, &other.0);
        let mut out = [[0.0; 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                // Accelerate's dgemm keeps two accumulators, even and odd k, each fused.
                let even = a[i][2].mul_add(b[2][j], a[i][0] * b[0][j]);
                let odd = a[i][3].mul_add(b[3][j], a[i][1] * b[1][j]);
                *value = even + odd;
            }
        }
        Affine(out)
    }

    /// The inverse matrix, or `None` when the matrix is singular.
    ///
    /// Mirrors `numpy.linalg.inv`, which is LAPACK `dgesv` (LU with partial pivoting, then two
    /// triangular solves) from Accelerate. The factorisation follows `dgetrf2`'s recursive
    /// halving and the fused multiply-adds Accelerate uses; this was matched bit for bit against
    /// numpy on random and real scan affines.
    pub fn inverse(&self) -> Option<Affine> {
        let mut lu = self.0;
        let mut pivots = [0usize; 4];
        if !getrf2(&mut lu, 0..4, 0..4, &mut pivots) {
            return None;
        }

        let mut inv = Affine::IDENTITY.0;
        for (row, &pivot) in pivots.iter().enumerate() {
            inv.swap(row, pivot);
        }
        for rhs in 0..4 {
            // L (unit lower) then U, both column by column with fused updates
            for k in 0..4 {
                for row in k + 1..4 {
                    inv[row][rhs] = (-lu[row][k]).mul_add(inv[k][rhs], inv[row][rhs]);
                }
            }
            for k in (0..4).rev() {
                inv[k][rhs] *= 1.0 / lu[k][k];
                for row in 0..k {
                    inv[row][rhs] = (-lu[row][k]).mul_add(inv[k][rhs], inv[row][rhs]);
                }
            }
        }
        Some(Affine(inv))
    }

    /// Voxel sizes: the Euclidean norm of each of the first three columns.
    ///
    /// Mirrors `nibabel.affines.voxel_sizes` (`sqrt(sum(rzs**2, axis=0))`, summed top to bottom).
    pub fn voxel_sizes(&self) -> [f64; 3] {
        let m = &self.0;
        std::array::from_fn(|col| {
            (m[0][col] * m[0][col] + m[1][col] * m[1][col] + m[2][col] * m[2][col]).sqrt()
        })
    }

    /// Voxel sizes rounded to float32, which is how nibabel stores them in `pixdim`.
    ///
    /// Mirrors `img.header.get_zooms()` for an image whose header nibabel built from this affine.
    pub fn header_zooms(&self) -> [f32; 3] {
        self.voxel_sizes().map(|size| size as f32)
    }

    /// Maps one voxel coordinate to world space.
    ///
    /// Mirrors `nibabel.affines.apply_affine(affine, point)` for a single point: numpy multiplies
    /// a `(1, 3)` by a `(3, 3)` matrix without BLAS, summing left to right, then adds the
    /// translation.
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let m = &self.0;
        std::array::from_fn(|row| {
            point[0] * m[row][0] + point[1] * m[row][1] + point[2] * m[row][2] + m[row][3]
        })
    }

    /// Maps several voxel coordinates at once.
    ///
    /// Mirrors `nibabel.affines.apply_affine(affine, points)` for an `(N, 3)` array with `N > 1`,
    /// where numpy hands `points @ rzs.T` to BLAS, which computes each output as one fused
    /// multiply-add chain.
    pub fn apply_many(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if points.len() == 1 {
            return vec![self.apply(points[0])];
        }
        let m = &self.0;
        points
            .iter()
            .map(|p| {
                std::array::from_fn(|row| {
                    let mut sum = p[0] * m[row][0];
                    sum = p[1].mul_add(m[row][1], sum);
                    sum = p[2].mul_add(m[row][2], sum);
                    sum + m[row][3]
                })
            })
            .collect()
    }

    /// Whether every entry is within `atol + rtol * |other|` of `other`'s.
    ///
    /// Mirrors `np.allclose(self, other)` with its defaults (`rtol=1e-5`, `atol=1e-8`).
    pub fn allclose(&self, other: &Affine) -> bool {
        self.0
            .iter()
            .flatten()
            .zip(other.0.iter().flatten())
            .all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs())
    }

    /// Determinant of the 3×3 block: the signed voxel volume in mm³.
    pub fn determinant3(&self) -> f64 {
        let m = self.rzs();
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/// LU factorisation of the block `rows × cols` in place, in LAPACK `dgetrf2` order: factor the
/// left half of the columns, solve and update the right half, then factor what remains.
/// `pivots[col]` is the row swapped into place at that column. Returns false on a zero pivot.
fn getrf2(a: &mut [[f64; 4]; 4], rows: Range<usize>, cols: Range<usize>, pivots: &mut [usize; 4]) -> bool {
    let (r0, c0) = (rows.start, cols.start);
    if cols.len() == 1 {
        // first row with the largest magnitude, like LAPACK's idamax
        let mut pivot = r0;
        for row in r0 + 1..rows.end {
            if a[row][c0].abs() > a[pivot][c0].abs() {
                pivot = row;
            }
        }
        pivots[c0] = pivot;
        if a[pivot][c0] == 0.0 {
            return false;
        }
        a.swap(r0, pivot);
        let reciprocal = 1.0 / a[r0][c0];
        for row in r0 + 1..rows.end {
            a[row][c0] *= reciprocal;
        }
        return true;
    }
    let half = rows.len().min(cols.len()) / 2;
    let (left, right) = (c0..c0 + half, c0 + half..cols.end);
    if !getrf2(a, rows.clone(), left, pivots) {
        return false;
    }
    // A12 = L11⁻¹ A12 (dtrsm, unit lower), plain multiply-subtract
    for k in right.clone() {
        for i in 1..half {
            for j in 0..i {
                a[r0 + i][k] -= a[r0 + i][c0 + j] * a[r0 + j][k];
            }
        }
    }
    // A22 -= A21 A12 (dgemm): the dot product is one fused chain, then a separate subtract
    for i in r0 + half..rows.end {
        for k in right.clone() {
            let mut dot = a[i][c0] * a[r0][k];
            for j in 1..half {
                dot = a[i][c0 + j].mul_add(a[r0 + j][k], dot);
            }
            a[i][k] -= dot;
        }
    }
    getrf2(a, r0 + half..rows.end, right, pivots)
}

impl fmt::Debug for Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Affine[")?;
        for row in &self.0 {
            writeln!(f, "  [{:>12.6}, {:>12.6}, {:>12.6}, {:>12.6}],", row[0], row[1], row[2], row[3])?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_of_axis_aligned_affine_is_exact() {
        let affine =
            Affine::from_matvec([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 4.0]], [-91.0, 12.5, 3.0]);
        let inv = affine.inverse().unwrap();
        assert_eq!(inv.dot(&affine), Affine::IDENTITY);
        assert_eq!(inv.translation(), [45.5, -25.0, -0.75]);
    }

    #[test]
    fn singular_affine_has_no_inverse() {
        let affine = Affine::from_matvec([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
        assert!(affine.inverse().is_none());
    }
}
