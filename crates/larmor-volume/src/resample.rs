//! Resampling one voxel grid onto another: scipy's `affine_transform` and nibabel's wrappers.

use ndarray::{Array3, ArrayView3, Axis};
use rayon::prelude::*;

use crate::affine::Affine;
use crate::volume::Volume;

/// A voxel type that scipy.ndimage can resample.
///
/// scipy interpolates in float64 and writes the result back in the input dtype, so integer images
/// come back rounded. `from_interpolated` is that write-back (`CASE_INTERP_OUT_*` in
/// `ni_interpolation.c`).
pub trait Voxel: Copy + Send + Sync + 'static {
    /// The value as the float64 scipy computes with.
    fn to_f64(self) -> f64;
    /// An interpolated float64 written back to this type the way scipy does it.
    fn from_interpolated(t: f64) -> Self;
}

macro_rules! voxel_float {
    ($($t:ty),*) => {$(
        impl Voxel for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_interpolated(t: f64) -> Self {
                t as $t
            }
        }
    )*};
}

macro_rules! voxel_signed {
    ($($t:ty),*) => {$(
        impl Voxel for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            /// Rounds half away from zero, then clamps (`CASE_INTERP_OUT_INT`).
            fn from_interpolated(t: f64) -> Self {
                let t = if t > 0.0 { t + 0.5 } else { t - 0.5 };
                t.clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t
            }
        }
    )*};
}

macro_rules! voxel_unsigned {
    ($($t:ty),*) => {$(
        impl Voxel for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            /// Rounds half up, negatives to 0, then clamps (`CASE_INTERP_OUT_UINT`).
            fn from_interpolated(t: f64) -> Self {
                let t = if t > 0.0 { t + 0.5 } else { 0.0 };
                t.min(<$t>::MAX as f64) as $t
            }
        }
    )*};
}

voxel_float!(f32, f64);
voxel_signed!(i8, i16, i32, i64);
voxel_unsigned!(u8, u16, u32, u64);

/// Spline order for resampling. Only the orders nanobrain uses are supported, because higher
/// orders need scipy's spline prefilter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// `order=0`: nearest neighbour, `floor(x + 0.5)`.
    Nearest,
    /// `order=1`: trilinear.
    Linear,
}

impl Interpolation {
    /// The interpolation for a scipy spline `order`, or `None` if it is not 0 or 1.
    pub fn from_order(order: i64) -> Option<Interpolation> {
        match order {
            0 => Some(Interpolation::Nearest),
            1 => Some(Interpolation::Linear),
            _ => None,
        }
    }
}

/// Resamples `input` at `matrix @ out_index + offset` for every output voxel.
///
/// Mirrors `scipy.ndimage.affine_transform(input, matrix, offset, output_shape, order,
/// mode="constant", cval=cval)` with a full 3×3 matrix (`NI_GeometricTransform`). A sample is
/// taken only when its coordinate lies in `[0, n - 1]` on every axis; anything else is `cval`.
/// At the upper edge the linear stencil's second tap is mirrored, but it has zero weight there.
pub fn affine_transform<T: Voxel>(
    input: ArrayView3<'_, T>,
    matrix: [[f64; 3]; 3],
    offset: [f64; 3],
    output_shape: [usize; 3],
    order: Interpolation,
    cval: f64,
) -> Array3<T> {
    let grid = Resampling { matrix, offset, output_shape, order, cval };
    grid.run(input, T::to_f64, T::from_interpolated)
}

/// One `affine_transform` call: the voxel mapping, output grid, order and fill value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Resampling {
    pub matrix: [[f64; 3]; 3],
    pub offset: [f64; 3],
    pub output_shape: [usize; 3],
    pub order: Interpolation,
    pub cval: f64,
}

impl Resampling {
    /// The mapping `inv(from_affine) @ to_affine` of `resample_from_to`, or `None` if
    /// `from_affine` is singular.
    pub fn between(
        from_affine: &Affine,
        to_shape: [usize; 3],
        to_affine: &Affine,
        order: Interpolation,
    ) -> Option<Self> {
        let to_vox2from_vox = from_affine.inverse()?.dot(to_affine);
        Some(Resampling {
            matrix: to_vox2from_vox.rzs(),
            offset: to_vox2from_vox.translation(),
            output_shape: to_shape,
            order,
            cval: 0.0,
        })
    }

    /// Runs the resampling. `read` gives the float64 value scipy sees for a stored voxel and
    /// `write` converts an interpolated float64 to the output type; reading through a scale
    /// factor this way equals resampling a pre-scaled float64 copy, without making one.
    pub fn run<T: Copy + Sync, O: Copy + Send + Sync>(
        &self,
        input: ArrayView3<'_, T>,
        read: impl Fn(T) -> f64 + Sync,
        write: impl Fn(f64) -> O + Sync,
    ) -> Array3<O> {
        let mut output = Array3::from_elem(self.output_shape, write(self.cval));
        if input.is_empty() {
            return output;
        }
        let dims = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let (matrix, offset) = (&self.matrix, &self.offset);
        output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut plane)| {
            for ((j, k), out) in plane.indexed_iter_mut() {
                let index = [i as f64, j as f64, k as f64];
                // scipy's loop `tmp = shift; tmp += coord * m` is compiled with fused multiply-adds
                let coords: [f64; 3] = std::array::from_fn(|axis| {
                    let row = &matrix[axis];
                    let mut t = offset[axis];
                    t = index[0].mul_add(row[0], t);
                    t = index[1].mul_add(row[1], t);
                    index[2].mul_add(row[2], t)
                });
                if let Some(value) = sample(&input, dims, coords, self.order, &read) {
                    *out = write(value);
                }
            }
        });
        output
    }
}

/// The interpolated value at `coords`, or `None` outside the "constant" mode's valid range.
fn sample<T: Copy>(
    input: &ArrayView3<'_, T>,
    dims: [usize; 3],
    coords: [f64; 3],
    order: Interpolation,
    read: &impl Fn(T) -> f64,
) -> Option<f64> {
    if (0..3).any(|axis| !(coords[axis] >= 0.0 && coords[axis] <= (dims[axis] - 1) as f64)) {
        return None;
    }
    match order {
        Interpolation::Nearest => {
            let [x, y, z] = coords.map(|c| (c + 0.5).floor() as usize);
            Some(read(input[[x, y, z]]))
        }
        Interpolation::Linear => {
            let mut lower = [0usize; 3];
            let mut upper = [0usize; 3];
            let mut weights = [[0.0f64; 2]; 3];
            for axis in 0..3 {
                let start = coords[axis].floor();
                lower[axis] = start as usize;
                // past the last voxel scipy mirrors the stencil (n -> n - 2); its weight is 0
                upper[axis] =
                    if lower[axis] + 1 < dims[axis] { lower[axis] + 1 } else { dims[axis].saturating_sub(2) };
                let w0 = 1.0 - (coords[axis] - start);
                weights[axis] = [w0, 1.0 - w0];
            }
            // scipy visits the 8 corners with z fastest and multiplies each value by the x, y and
            // z weights in turn before adding it; no step is fused
            let mut t = 0.0;
            for (a, x) in [lower[0], upper[0]].into_iter().enumerate() {
                for (b, y) in [lower[1], upper[1]].into_iter().enumerate() {
                    for (c, z) in [lower[2], upper[2]].into_iter().enumerate() {
                        t += read(input[[x, y, z]]) * weights[0][a] * weights[1][b] * weights[2][c];
                    }
                }
            }
            Some(t)
        }
    }
}

/// Resamples `from` (with affine `from_affine`) onto the grid `to_shape` / `to_affine`.
///
/// Mirrors the data part of `nibabel.processing.resample_from_to(from_img, (to_shape, to_affine),
/// order, mode="constant", cval=0)`: the voxel mapping is `inv(from_affine) @ to_affine`, the
/// output keeps the input dtype. Returns `None` if `from_affine` is singular.
pub fn resample_from_to<T: Voxel>(
    from: ArrayView3<'_, T>,
    from_affine: &Affine,
    to_shape: [usize; 3],
    to_affine: &Affine,
    order: Interpolation,
) -> Option<Array3<T>> {
    let grid = Resampling::between(from_affine, to_shape, to_affine, order)?;
    Some(grid.run(from, T::to_f64, T::from_interpolated))
}

/// The smallest axis-aligned grid with `voxel_sizes` that holds every voxel centre of an image.
///
/// Mirrors `nibabel.spaces.vox2out_vox((shape, affine), voxel_sizes)`: the output affine is
/// `diag(voxel_sizes)` translated to the minimum corner, and the shape is
/// `ceil((max - min) / voxel_sizes) + 1`.
pub fn vox2out_vox(shape: [usize; 3], affine: &Affine, voxel_sizes: [f64; 3]) -> ([usize; 3], Affine) {
    let corners: Vec<[f64; 3]> = (0..8)
        .map(|corner: usize| {
            std::array::from_fn(
                |axis| if (corner >> (2 - axis)) & 1 == 1 { (shape[axis] - 1) as f64 } else { 0.0 },
            )
        })
        .collect();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for world in affine.apply_many(&corners) {
        for axis in 0..3 {
            min[axis] = min[axis].min(world[axis]);
            max[axis] = max[axis].max(world[axis]);
        }
    }
    let out_shape =
        std::array::from_fn(|axis| (((max[axis] - min[axis]) / voxel_sizes[axis]).ceil() + 1.0) as usize);
    let diagonal = [[voxel_sizes[0], 0.0, 0.0], [0.0, voxel_sizes[1], 0.0], [0.0, 0.0, voxel_sizes[2]]];
    (out_shape, Affine::from_matvec(diagonal, min))
}

impl<T: Voxel> Volume<T> {
    /// This volume resampled onto the grid `to_shape` / `to_affine`.
    ///
    /// Mirrors `nibabel.processing.resample_from_to(img, (to_shape, to_affine), order)`, including
    /// the zooms the result's header ends up with. Returns `None` if this affine is singular.
    pub fn resample_to(
        &self,
        to_shape: [usize; 3],
        to_affine: &Affine,
        order: Interpolation,
    ) -> Option<Volume<T>> {
        let data = resample_from_to(self.data.view(), &self.affine, to_shape, to_affine, order)?;
        Some(Volume { data, affine: *to_affine, zooms: self.zooms_for(to_affine) })
    }

    /// This volume resampled to world-aligned voxels of `voxel_sizes` mm.
    ///
    /// Mirrors `nibabel.processing.resample_to_output(img, voxel_sizes, order)`.
    pub fn resample_to_output(&self, voxel_sizes: [f64; 3], order: Interpolation) -> Option<Volume<T>> {
        let (shape, affine) = vox2out_vox(self.shape(), &self.affine, voxel_sizes);
        self.resample_to(shape, &affine, order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_write_back_rounds_like_scipy() {
        assert_eq!(i16::from_interpolated(2.5), 3);
        assert_eq!(i16::from_interpolated(-2.5), -3);
        assert_eq!(i16::from_interpolated(-0.2), 0);
        assert_eq!(i16::from_interpolated(1e9), i16::MAX);
        assert_eq!(u8::from_interpolated(-3.0), 0);
        assert_eq!(u8::from_interpolated(254.5), 255);
        assert_eq!(u8::from_interpolated(300.0), 255);
    }

    #[test]
    fn output_grid_covers_all_voxel_centres() {
        let (shape, affine) = vox2out_vox([3, 4, 5], &Affine::IDENTITY, [2.0, 2.0, 2.0]);
        assert_eq!(shape, [2, 3, 3]);
        assert_eq!(affine.translation(), [0.0, 0.0, 0.0]);
    }
}
