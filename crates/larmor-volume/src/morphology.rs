//! Connected components and hole filling on boolean masks, with scipy's 6-connectivity.

use ndarray::{Array3, ArrayView3};

/// Union-find over provisional labels, where the root of a set is always its smallest label.
struct Forest {
    parent: Vec<u32>,
}

impl Forest {
    fn new() -> Forest {
        // label 0 is the background and never joins a set
        Forest { parent: vec![0] }
    }

    fn make_set(&mut self) -> u32 {
        let label = self.parent.len() as u32;
        self.parent.push(label);
        label
    }

    fn find(&mut self, mut label: u32) -> u32 {
        while self.parent[label as usize] != label {
            let grandparent = self.parent[self.parent[label as usize] as usize];
            self.parent[label as usize] = grandparent;
            label = grandparent;
        }
        label
    }

    fn union(&mut self, a: u32, b: u32) -> u32 {
        let (ra, rb) = (self.find(a), self.find(b));
        let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[child as usize] = root;
        root
    }
}

/// Labels the 6-connected components of a C-order mask with shape `dims`: 0 is background,
/// components are numbered from 1 in the C order of their first voxel.
fn label_flat(mask: &[bool], dims: [usize; 3]) -> (Vec<u32>, usize) {
    let [nx, ny, nz] = dims;
    let (stride_x, stride_y) = (ny * nz, nz);
    let mut labels = vec![0u32; mask.len()];
    let mut forest = Forest::new();
    // first pass: provisional labels from the three neighbours already visited in raster order.
    // A component's first voxel always opens a new label, so with min-rooted unions every
    // component's root is the label of its first voxel.
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let i = x * stride_x + y * stride_y + z;
                if !mask[i] {
                    continue;
                }
                let mut label = 0;
                for (has_neighbour, j) in [
                    (z > 0, i.wrapping_sub(1)),
                    (y > 0, i.wrapping_sub(stride_y)),
                    (x > 0, i.wrapping_sub(stride_x)),
                ] {
                    if has_neighbour && labels[j] != 0 {
                        label =
                            if label == 0 { forest.find(labels[j]) } else { forest.union(label, labels[j]) };
                    }
                }
                labels[i] = if label == 0 { forest.make_set() } else { label };
            }
        }
    }
    // second pass: number the roots consecutively, which keeps first-voxel order
    let mut number = vec![0u32; forest.parent.len()];
    let mut count = 0u32;
    for label in 1..forest.parent.len() as u32 {
        if forest.find(label) == label {
            count += 1;
            number[label as usize] = count;
        }
    }
    for l in labels.iter_mut().filter(|l| **l != 0) {
        *l = number[forest.find(*l) as usize];
    }
    (labels, count as usize)
}

/// Labels the 6-connected components of `mask`: 0 is background, components are numbered from 1
/// in the C order of their first voxel. Returns the labels and the number of components.
///
/// Mirrors `scipy.ndimage.label(mask)` with the default structure, whose numbering follows the
/// same raster order.
pub fn label(mask: ArrayView3<'_, bool>) -> (Array3<u32>, usize) {
    let dims = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mask = mask.as_standard_layout();
    let (labels, count) = label_flat(mask.as_slice().unwrap(), dims);
    (Array3::from_shape_vec(dims, labels).unwrap(), count)
}

/// Keeps only the largest 6-connected component; on a size tie, the one labelled first.
///
/// Mirrors nanobrain's `utils/preprocessing.py::largest_component` (`ndi.label`, `ndi.sum_labels`,
/// `np.argmax`, which returns the first maximum). Masks with fewer than two components are
/// returned unchanged.
pub fn largest_component(mask: ArrayView3<'_, bool>) -> Array3<bool> {
    let (labels, count) = label(mask);
    if count < 2 {
        return mask.to_owned();
    }
    let mut sizes = vec![0usize; count + 1];
    for &l in labels.iter() {
        sizes[l as usize] += 1;
    }
    let mut largest = 1;
    for l in 2..=count {
        if sizes[l] > sizes[largest] {
            largest = l;
        }
    }
    labels.mapv(|l| l as usize == largest)
}

/// Fills background regions that are not 6-connected to the array border.
///
/// Mirrors `scipy.ndimage.binary_fill_holes(mask)`: scipy dilates from outside the array
/// (`border_value=1`) through the background until nothing changes, and inverts what it reached.
/// Here the background is labelled instead, and every background component that touches a face
/// of the array is what the dilation reaches.
pub fn binary_fill_holes(mask: ArrayView3<'_, bool>) -> Array3<bool> {
    let dims = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    if mask.is_empty() {
        return mask.to_owned();
    }
    let background: Vec<bool> = mask.as_standard_layout().iter().map(|&v| !v).collect();
    let (labels, count) = label_flat(&background, dims);
    let labels = Array3::from_shape_vec(dims, labels).unwrap();

    let mut reaches_border = vec![false; count + 1];
    for axis in 0..3 {
        for index in [0, dims[axis] - 1] {
            for &l in labels.index_axis(ndarray::Axis(axis), index).iter() {
                reaches_border[l as usize] = true;
            }
        }
    }
    // label 0 is the mask itself, which stays set
    reaches_border[0] = false;
    labels.mapv(|l| !reaches_border[l as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    #[test]
    fn hollow_cube_is_filled_but_open_cup_is_not() {
        let mut cube = Array3::from_elem([7, 7, 7], false);
        cube.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
        cube.slice_mut(s![2..5, 2..5, 2..5]).fill(false);
        let filled = binary_fill_holes(cube.view());
        assert_eq!(filled.iter().filter(|&&v| v).count(), 125);

        cube.slice_mut(s![2..5, 2..5, 5..6]).fill(false); // open the lid
        let cup = binary_fill_holes(cube.view());
        assert_eq!(cup, cube);
    }

    #[test]
    fn size_tie_keeps_the_first_component() {
        let mut mask = Array3::from_elem([3, 5, 3], false);
        mask[[0, 0, 0]] = true;
        mask[[0, 0, 1]] = true;
        mask[[2, 4, 2]] = true;
        mask[[2, 4, 1]] = true;
        mask[[1, 2, 1]] = true;
        let kept = largest_component(mask.view());
        assert!(kept[[0, 0, 0]] && kept[[0, 0, 1]]);
        assert_eq!(kept.iter().filter(|&&v| v).count(), 2);
    }

    #[test]
    fn u_shape_merges_into_one_component_numbered_by_first_voxel() {
        // two arms that only join at the bottom: the raster scan meets them as separate labels
        let mut mask = Array3::from_elem([1, 4, 5], false);
        mask.slice_mut(s![0, 0..4, 0]).fill(true);
        mask.slice_mut(s![0, 0..4, 4]).fill(true);
        mask.slice_mut(s![0, 3, 0..5]).fill(true);
        mask[[0, 1, 2]] = true; // an island, first seen after both arms
        let (labels, count) = label(mask.view());
        assert_eq!(count, 2);
        assert_eq!(labels[[0, 0, 4]], 1);
        assert_eq!(labels[[0, 1, 2]], 2);
    }
}
