//! Loading the reference tensors written by `tests/data/make_test_data.py`.

#![allow(dead_code)]

use std::path::PathBuf;

use larmor_volume::Affine;
use larmor_volume::ndarray::{Array3, ArrayD, IxDyn};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

/// One module's reference data: tensors plus the case list.
pub struct Reference {
    tensors: SafeTensors<'static>,
    /// `{case name: {parameters}}` from the JSON sidecar.
    pub cases: serde_json::Map<String, Value>,
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

impl Reference {
    pub fn load(module: &str) -> Reference {
        let bytes =
            std::fs::read(data_dir().join(format!("{module}.safetensors"))).expect("reference tensors");
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let tensors = SafeTensors::deserialize(bytes).expect("valid safetensors");
        let json =
            std::fs::read_to_string(data_dir().join(format!("{module}.json"))).expect("reference cases");
        let Value::Object(cases) = serde_json::from_str(&json).unwrap() else {
            panic!("cases must be an object")
        };
        Reference { tensors, cases }
    }

    /// Case names of one kind, sorted.
    pub fn cases_of(&self, kind: &str) -> Vec<(String, Value)> {
        let mut out: Vec<_> = self
            .cases
            .iter()
            .filter(|(_, v)| v["kind"] == kind)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(!out.is_empty(), "no reference cases of kind {kind}");
        out
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    pub fn dtype(&self, name: &str) -> Dtype {
        self.tensors.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}")).dtype()
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.tensors.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}")).shape().to_vec()
    }

    /// Raw little-endian bytes of a tensor, checked against the expected dtype.
    fn raw(&self, name: &str, dtype: Dtype) -> (&'static [u8], Vec<usize>) {
        let view = self.tensors.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}"));
        assert_eq!(view.dtype(), dtype, "dtype of {name}");
        (view.data(), view.shape().to_vec())
    }
}

macro_rules! getter {
    ($name:ident, $array:ident, $t:ty, $dtype:ident) => {
        impl Reference {
            pub fn $name(&self, name: &str) -> Vec<$t> {
                let (bytes, _) = self.raw(name, Dtype::$dtype);
                bytes
                    .chunks_exact(std::mem::size_of::<$t>())
                    .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            pub fn $array(&self, name: &str) -> ArrayD<$t> {
                let (_, shape) = self.raw(name, Dtype::$dtype);
                ArrayD::from_shape_vec(IxDyn(&shape), self.$name(name)).unwrap()
            }
        }
    };
}

getter!(f32s, f32_array, f32, F32);
getter!(f64s, f64_array, f64, F64);
getter!(i64s, i64_array, i64, I64);
getter!(u8s, u8_array, u8, U8);
getter!(u32s, u32_array, u32, U32);
getter!(i8s, i8_array, i8, I8);
getter!(i16s, i16_array, i16, I16);
getter!(u16s, u16_array, u16, U16);
getter!(i32s, i32_array, i32, I32);
getter!(u64s, u64_array, u64, U64);

impl Reference {
    pub fn bools(&self, name: &str) -> Array3<bool> {
        let (bytes, shape) = self.raw(name, Dtype::BOOL);
        Array3::from_shape_vec([shape[0], shape[1], shape[2]], bytes.iter().map(|&b| b != 0).collect())
            .unwrap()
    }

    pub fn affine(&self, name: &str) -> Affine {
        let v = self.f64s(name);
        assert_eq!(v.len(), 16, "{name} is not a 4x4 matrix");
        Affine(std::array::from_fn(|i| std::array::from_fn(|j| v[4 * i + j])))
    }

    pub fn matrix3(&self, name: &str) -> [[f64; 3]; 3] {
        let v = self.f64s(name);
        std::array::from_fn(|i| std::array::from_fn(|j| v[3 * i + j]))
    }

    pub fn vec3(&self, name: &str) -> [f64; 3] {
        self.f64s(name).try_into().unwrap()
    }

    pub fn array3<T: Clone>(array: ArrayD<T>) -> Array3<T> {
        array.into_dimensionality().unwrap()
    }
}

/// Bit-exact float comparison that treats NaNs as equal and reports the first mismatch.
pub fn assert_bits_eq_f32(
    actual: impl IntoIterator<Item = f32>,
    expected: impl IntoIterator<Item = f32>,
    what: &str,
) {
    let mut mismatches = 0;
    let mut first = None;
    let mut count = 0;
    for (i, (a, e)) in actual.into_iter().zip(expected).enumerate() {
        count += 1;
        if a.to_bits() != e.to_bits() && !(a.is_nan() && e.is_nan()) {
            mismatches += 1;
            first.get_or_insert((i, a, e));
        }
    }
    assert!(mismatches == 0, "{what}: {mismatches} of {count} values differ, first at {:?}", first.unwrap());
}

pub fn assert_bits_eq_f64(
    actual: impl IntoIterator<Item = f64>,
    expected: impl IntoIterator<Item = f64>,
    what: &str,
) {
    let mut mismatches = 0;
    let mut first = None;
    let mut count = 0;
    for (i, (a, e)) in actual.into_iter().zip(expected).enumerate() {
        count += 1;
        if a.to_bits() != e.to_bits() && !(a.is_nan() && e.is_nan()) {
            mismatches += 1;
            first.get_or_insert((i, a, e));
        }
    }
    assert!(mismatches == 0, "{what}: {mismatches} of {count} values differ, first at {:?}", first.unwrap());
}
