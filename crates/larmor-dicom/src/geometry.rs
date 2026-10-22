//! dcm2niix's spatial transform for a stack of slices, in its own float32 arithmetic.
//!
//! dcm2niix computes the NIfTI sform in `float` (`mat44` holds `float m[4][4]`), and the binary this
//! Mac and the PyPI wheel ship is built by clang, which fuses `a * b + c` into one fused
//! multiply-add (one rounding instead of two) inside a single C expression (`-ffp-contract=on`, its
//! default). Each function here is a literal port of the dcm2niix function it names, with
//! `mul_add` exactly where clang fuses: for `a*b + c*d` clang fuses the left product,
//! `fma(a, b, c*d)`; for `a*b - c*d`, `fma(a, b, -(c*d))`; a `+=` of a product fuses into the
//! running sum. Products by 0 or ±1 and single products are exact either way.
//!
//! References are to `console/nii_dicom.cpp`, `nii_dicom_batch.cpp`, `nii_ortho.cpp` and
//! `nifti1_io_core.cpp` of dcm2niix v1.0.20260724.

/// A 3x3 float matrix, `m[row][column]` (dcm2niix's `mat33`).
pub(crate) type Mat33 = [[f32; 3]; 3];
/// A 4x4 float matrix, `m[row][column]` (dcm2niix's `mat44`).
pub(crate) type Mat44 = [[f32; 4]; 4];

/// `isSameFloat`: equal within `FLT_EPSILON`.
pub(crate) fn is_same_float(a: f32, b: f32) -> bool {
    (a - b).abs() <= f32::EPSILON
}

/// `isSameFloatGE`: equal within 0.0001.
pub(crate) fn is_same_float_ge(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.0001
}

/// `nifti_mat33_mul`: `C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]`.
fn mat33_mul(a: &Mat33, b: &Mat33) -> Mat33 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][2].mul_add(b[2][j], a[i][0].mul_add(b[0][j], a[i][1] * b[1][j])))
    })
}

/// `nifti_mat33_transpose`.
fn transpose(a: &Mat33) -> Mat33 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

/// `nifti_mat33_determ`, in double precision like dcm2niix (only its sign is used).
fn determinant(r: &Mat33) -> f64 {
    let m = r.map(|row| row.map(f64::from));
    m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[2][1] * m[1][2] - m[1][0] * m[0][1] * m[2][2]
        + m[1][0] * m[2][1] * m[0][2]
        + m[2][0] * m[0][1] * m[1][2]
        - m[2][0] * m[1][1] * m[0][2]
}

/// `nifti_vect44mat44_mul`: `m · v`, each row summed with `vO += m[i][j] * v[j]` from 0.
pub(crate) fn mat44_apply(m: &Mat44, v: [f32; 4]) -> [f32; 4] {
    std::array::from_fn(|i| (0..4).fold(0f32, |sum, j| m[i][j].mul_add(v[j], sum)))
}

/// `crossProduct` (`nifti1_io_core.cpp`).
pub(crate) fn cross(u: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    [
        u[1].mul_add(v[2], -(v[1] * u[2])),
        (-u[0]).mul_add(v[2], v[0] * u[2]),
        u[0].mul_add(v[1], -(v[0] * u[1])),
    ]
}

/// `dotProduct`: `u0*v0 + v1*u1 + v2*u2`.
pub(crate) fn dot(u: [f32; 3], v: [f32; 3]) -> f32 {
    v[2].mul_add(u[2], u[0].mul_add(v[0], v[1] * u[1]))
}

/// `vec3Length` (`nii_dicom_batch.cpp`).
pub(crate) fn length(v: [f32; 3]) -> f32 {
    v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1])).sqrt()
}

/// `intersliceDistance`: the distance between two slice positions. `sqr` is its own function, so
/// clang does not fuse its products into the sum; the gantry tilt factor is 1 for MR.
pub(crate) fn interslice_distance(p1: [f32; 3], p2: [f32; 3]) -> f32 {
    let sqr = |v: f32| v * v;
    (sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2])).sqrt()
}

/// `intersliceDistanceSigned`: the distance from slice 1 to slice 2, negative when slice 2 lies
/// behind slice 1 along the normal of slice 1's orientation.
pub(crate) fn interslice_distance_signed(p1: [f32; 3], p2: [f32; 3], orientation: [f32; 6]) -> f32 {
    let along = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let len = length(along);
    if is_same_float(len, 0.0) {
        return len;
    }
    let read = [orientation[0], orientation[1], orientation[2]];
    let phase = [orientation[3], orientation[4], orientation[5]];
    if dot(cross(read, phase), along) < 0.0 { -len } else { len }
}

/// `nifti_dicom2mat`: the LPS voxel-to-patient matrix of a slice from its ImageOrientationPatient,
/// ImagePositionPatient and voxel sizes (column spacing, row spacing, slice spacing).
pub(crate) fn dicom2mat(orientation: [f32; 6], position: [f32; 3], voxel_mm: [f32; 3]) -> Mat44 {
    let o = orientation;
    let mut q: Mat33 = [[o[0], o[1], o[2]], [o[3], o[4], o[5]], [0.0; 3]];
    for (row, fallback) in [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])] {
        // the squared norm is a float expression, converted to double afterwards
        let r = q[row];
        let val = f64::from(r[2].mul_add(r[2], r[0].mul_add(r[0], r[1] * r[1])));
        if val > 0.0 {
            let scale = (1.0 / val.sqrt()) as f32;
            q[row] = r.map(|v| v * scale);
        } else {
            q[row] = fallback;
        }
    }
    q[2] = [
        q[0][1].mul_add(q[1][2], -(q[0][2] * q[1][1])),
        q[0][2].mul_add(q[1][0], -(q[0][0] * q[1][2])),
        q[0][0].mul_add(q[1][1], -(q[0][1] * q[1][0])),
    ];
    let mut q = transpose(&q);
    if determinant(&q) < 0.0 {
        for row in &mut q {
            row[2] = -row[2];
        }
    }
    let diag: Mat33 = [[voxel_mm[0], 0.0, 0.0], [0.0, voxel_mm[1], 0.0], [0.0, 0.0, voxel_mm[2]]];
    let q = mat33_mul(&q, &diag);
    [
        [q[0][0], q[0][1], q[0][2], position[0]],
        [q[1][0], q[1][1], q[1][2], position[1]],
        [q[2][0], q[2][1], q[2][2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// `verify_slice_dir`: points the third column of `r` (LPS, from [`dicom2mat`]) from the first
/// slice towards the others, negating it when the second slice lies on the other side. Returns the
/// slice axis (1 sagittal, 2 coronal, 3 axial), negative when the column was negated, or `None`
/// when the second slice does not move along that axis (dcm2niix then guesses; Larmor refuses).
pub(crate) fn verify_slice_dir(
    first: [f32; 3],
    second: [f32; 3],
    n_slices: usize,
    r: &mut Mat44,
) -> Option<i32> {
    if n_slices < 2 {
        return Some(0);
    }
    let column = [r[0][2].abs(), r[1][2].abs(), r[2][2].abs()];
    let mut axis = 1;
    if column[1] >= column[0] && column[1] >= column[2] {
        axis = 2;
    }
    if column[2] >= column[0] && column[2] >= column[1] {
        axis = 3;
    }
    let row = axis as usize - 1;
    let pos = second[row];
    if is_same_float(pos, first[row]) {
        return None;
    }
    let last = mat44_apply(r, [0.0, 0.0, (n_slices as f64 - 1.0) as f32, 1.0])[row];
    let origin = r[row][3];
    let flip = (pos > origin) != (last > origin);
    if flip {
        for row in r.iter_mut() {
            row[2] = -row[2];
        }
    }
    Some(if flip { -axis } else { axis })
}

/// LPS (DICOM) to RAS (NIfTI): negate the first two rows.
pub(crate) fn lps_to_ras(r: &mut Mat44) {
    for row in &mut r[..2] {
        for v in row.iter_mut() {
            *v = -*v;
        }
    }
}

/// The sform after `nii_flipZ`: the last slice becomes the origin and the slice axis is negated.
pub(crate) fn flip_z(s: &Mat44, n_slices: usize) -> Mat44 {
    flip_axis(s, 2, n_slices)
}

/// The sform after `nii_flipY`: the last row becomes the origin and the row axis is negated.
pub(crate) fn flip_y(s: &Mat44, n_rows: usize) -> Mat44 {
    flip_axis(s, 1, n_rows)
}

fn flip_axis(s: &Mat44, axis: usize, n: usize) -> Mat44 {
    let mut v = [0.0, 0.0, 0.0, 1.0];
    v[axis] = n as f32 - 1.0;
    let origin = mat44_apply(s, v);
    let mut flip: Mat33 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    flip[axis][axis] = -1.0;
    let r = mat33_mul(&rzs(s), &flip);
    with_rzs(&r, [origin[0], origin[1], origin[2]])
}

fn rzs(s: &Mat44) -> Mat33 {
    std::array::from_fn(|i| [s[i][0], s[i][1], s[i][2]])
}

fn with_rzs(r: &Mat33, translation: [f32; 3]) -> Mat44 {
    [
        [r[0][0], r[0][1], r[0][2], translation[0]],
        [r[1][0], r[1][1], r[1][2], translation[1]],
        [r[2][0], r[2][1], r[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// How `nii_setOrtho` reorients a 3-D acquisition: output axis `i` is input axis
/// `|axes[i]| - 1`, reversed when `axes[i]` is negative (dcm2niix's `orientVec`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ortho {
    pub(crate) axes: [i32; 3],
    /// The new sform.
    pub(crate) sform: Mat44,
}

/// `xyz2mm`: `R[i][0]*v0 + R[i][1]*v1 + R[i][2]*v2 + R[i][3]`.
fn xyz2mm(r: &Mat44, v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| r[i][2].mul_add(v[2], r[i][0].mul_add(v[0], r[i][1] * v[1])) + r[i][3])
}

/// `getDistance`: computed in double (`pow` promotes), returned as float.
fn distance(v: [f32; 3], min: [f32; 3]) -> f32 {
    let d = |i: usize| f64::from(v[i] - min[i]).powi(2);
    (d(0) + d(1) + d(2)).sqrt() as f32
}

/// `nii_setOrtho`: the orthogonal reorientation dcm2niix applies to 3-D acquisitions, or `None`
/// when the image is already in its best orthogonal orientation.
pub(crate) fn set_ortho(s: &Mat44, dims: [usize; 3]) -> Option<Ortho> {
    let canonical = (0..3).all(|i| (0..3).all(|j| if i == j { s[i][j] > 0.0 } else { s[i][j] == 0.0 }));
    if canonical {
        return None;
    }
    // minCornerFlip: the corner nearest the most left, posterior, inferior point becomes voxel 0
    let flips: [[f32; 3]; 8] =
        std::array::from_fn(|i| [0, 1, 2].map(|bit| if i & (1 << bit) != 0 { -1.0 } else { 1.0 }));
    let corners: [[f32; 3]; 8] = std::array::from_fn(|i| {
        let voxel =
            std::array::from_fn(|axis| if flips[i][axis] < 1.0 { (dims[axis] - 1) as f32 } else { 0.0 });
        xyz2mm(s, voxel)
    });
    let mut min = corners[0];
    for corner in &corners[1..] {
        for j in 0..3 {
            if corner[j] < min[j] {
                min[j] = corner[j];
            }
        }
    }
    let mut nearest = 0;
    let mut nearest_distance = distance(corners[0], min);
    for (i, &corner) in corners.iter().enumerate().skip(1) {
        let d = distance(corner, min);
        if d < nearest_distance {
            nearest_distance = d;
            nearest = i;
        }
    }
    let min = corners[nearest];
    let f = flips[nearest];
    // getBestOrient: of the six axis permutations (with those flips), the one whose product with the
    // sform has the largest trace, the first one on ties
    let permutations: [Mat33; 6] = [
        [[f[0], 0.0, 0.0], [0.0, f[1], 0.0], [0.0, 0.0, f[2]]],
        [[f[0], 0.0, 0.0], [0.0, 0.0, f[1]], [0.0, f[2], 0.0]],
        [[0.0, f[0], 0.0], [f[1], 0.0, 0.0], [0.0, 0.0, f[2]]],
        [[0.0, f[0], 0.0], [0.0, 0.0, f[1]], [f[2], 0.0, 0.0]],
        [[0.0, 0.0, f[0]], [f[1], 0.0, 0.0], [0.0, f[2], 0.0]],
        [[0.0, 0.0, f[0]], [0.0, f[1], 0.0], [f[2], 0.0, 0.0]],
    ];
    let original = rzs(s);
    let mut best = 0f32;
    let mut orient = None;
    for permutation in &permutations {
        let mut residual = 0f32;
        for i in 0..3 {
            for j in 0..3 {
                residual += original[i][j] * permutation[j][i];
            }
        }
        if residual > best {
            best = residual;
            orient = Some(*permutation);
        }
    }
    // dcm2niix would read an uninitialised matrix here; no real affine gets this far
    let orient = orient?;
    // setOrientVec
    let mut axes = [0i32; 3];
    for (i, row) in orient.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v > 0.0 {
                axes[j] = i as i32 + 1;
            }
            if v < 0.0 {
                axes[j] = -(i as i32 + 1);
            }
        }
    }
    if axes == [1, 2, 3] {
        return None;
    }
    // matMul33: mult[j][i] += a[j][k] * b[k][i]
    let r: Mat33 = std::array::from_fn(|j| {
        std::array::from_fn(|i| (0..3).fold(0f32, |sum, k| original[j][k].mul_add(orient[k][i], sum)))
    });
    Some(Ortho { axes, sform: with_rzs(&r, min) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_axial_stack_gets_the_ras_affine() {
        // axial slices, rows along +y (LPS posterior), ascending in z: LPS identity directions
        let mut r = dicom2mat([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [-100.0, -120.0, -50.0], [0.5, 0.75, 2.0]);
        assert_eq!(verify_slice_dir([-100.0, -120.0, -50.0], [-100.0, -120.0, -48.0], 30, &mut r), Some(3));
        lps_to_ras(&mut r);
        assert_eq!(
            r,
            [
                [-0.5, -0.0, -0.0, 100.0],
                [-0.0, -0.75, -0.0, 120.0],
                [0.0, 0.0, 2.0, -50.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );
        // descending slices flip the slice column
        let mut r = dicom2mat([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 10.0], [1.0, 1.0, 2.0]);
        assert_eq!(verify_slice_dir([0.0, 0.0, 10.0], [0.0, 0.0, 8.0], 5, &mut r), Some(-3));
        assert_eq!(r[2][2], -2.0);
        // flipping z then puts the last slice at the origin
        let flipped = flip_z(&r, 5);
        assert_eq!([flipped[2][2], flipped[2][3]], [2.0, 2.0]);
        // flipping y moves the origin to the last row
        let flipped = flip_y(&r, 4);
        assert_eq!([flipped[1][1], flipped[1][3]], [-1.0, 3.0]);
    }

    #[test]
    fn the_cross_product_is_fused_like_clang() {
        // a*b - c*d with one rounding for a*b: the fused result differs from the twice-rounded one
        let (a, b, c, d) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON, 1.0f32, 1.0f32);
        let fused = cross([0.0, a, c], [0.0, d, b])[0];
        assert_eq!(fused, a.mul_add(b, -(d * c)));
        assert_ne!(fused, a * b - d * c);
    }

    #[test]
    fn set_ortho_turns_a_sagittal_stack_towards_ras() {
        // sagittal: columns along -y (A), rows along -z (I), slices along +x (R) in RAS
        let s: Mat44 =
            [[0.0, 0.0, 1.0, -90.0], [-1.0, 0.0, 0.0, 120.0], [0.0, -1.0, 0.0, 110.0], [0.0, 0.0, 0.0, 1.0]];
        let ortho = set_ortho(&s, [240, 220, 180]).unwrap();
        assert_eq!(ortho.axes, [3, -1, -2]);
        assert_eq!(
            ortho.sform,
            [[1.0, 0.0, 0.0, -90.0], [0.0, 1.0, 0.0, -119.0], [0.0, 0.0, 1.0, -109.0], [0.0, 0.0, 0.0, 1.0]]
        );
        let identity: Mat44 =
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        assert_eq!(set_ortho(&identity, [10, 10, 10]), None);
    }

    #[test]
    fn signed_distance_follows_the_slice_normal() {
        let axial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(interslice_distance_signed([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], axial), 3.0);
        assert_eq!(interslice_distance_signed([0.0, 0.0, 0.0], [0.0, 0.0, -3.0], axial), -3.0);
        assert_eq!(interslice_distance([0.0, 0.0, 0.0], [0.0, 3.0, 4.0]), 5.0);
    }
}
