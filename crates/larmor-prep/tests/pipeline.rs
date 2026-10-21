//! End-to-end parity on small synthetic scans, from `tests/data/make_test_data.py` (nanobrain's
//! own functions with a 24 × 32 × 24 mm field of view). Runs without Python or fixtures.

use std::path::PathBuf;

use larmor_prep::Recipe;
use larmor_volume::Nifti;
use larmor_volume::ndarray::{Array3, ArrayD, IxDyn};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

struct Case<'a> {
    name: String,
    params: Value,
    tensors: &'a SafeTensors<'a>,
}

impl Case<'_> {
    fn raw(&self, field: &str, dtype: Dtype) -> (&[u8], Vec<usize>) {
        let t = self.tensors.tensor(&format!("{}/{field}", self.name)).unwrap();
        assert_eq!(t.dtype(), dtype, "{}/{field}", self.name);
        (t.data(), t.shape().to_vec())
    }

    fn array<T: Clone>(&self, field: &str, dtype: Dtype, parse: impl Fn(&[u8]) -> T) -> Array3<T> {
        let (bytes, shape) = self.raw(field, dtype);
        let size = bytes.len() / shape.iter().product::<usize>().max(1);
        let values: Vec<T> = bytes.chunks_exact(size).map(parse).collect();
        ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap().into_dimensionality().unwrap()
    }

    fn f64s(&self, field: &str) -> Vec<f64> {
        self.raw(field, Dtype::F64).0.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()
    }

    fn i64s(&self, field: &str) -> Vec<i64> {
        self.raw(field, Dtype::I64).0.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()
    }
}

fn f32_le(c: &[u8]) -> f32 {
    f32::from_le_bytes(c.try_into().unwrap())
}

#[test]
fn synthetic_scans_match_nanobrain_end_to_end() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let bytes = std::fs::read(dir.join("pipeline.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let cases: serde_json::Map<String, Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("pipeline.json")).unwrap()).unwrap();
    assert_eq!(cases.len(), 3);

    for (name, params) in cases {
        let case = Case { name: name.clone(), params, tensors: &tensors };
        let scan = Nifti::from_bytes(case.raw("scan", Dtype::U8).0).unwrap();
        assert_eq!(scan.data.dtype().name(), case.params["dtype"].as_str().unwrap(), "{name}");

        let mut recipe = Recipe::nanobrain_eval();
        let fov: Vec<f64> =
            case.params["fov"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        recipe.conform.fov = fov.try_into().unwrap();
        let p = recipe.run(&scan).unwrap_or_else(|e| panic!("{name}: {e}"));

        let head = case.array("head_mask", Dtype::U8, |c| c[0] > 0);
        assert_eq!(p.native_mask, head, "{name}: head mask");
        let conform = case.array("conform", Dtype::F32, f32_le);
        let same_bits = p.conformed.iter().zip(conform.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same_bits, "{name}: conform differs");
        assert_eq!(p.mask, case.array("fit_mask", Dtype::BOOL, |c| c[0] != 0), "{name}: fit mask");
        let quantized = case.array("quantized", Dtype::U16, |c| u16::from_le_bytes(c.try_into().unwrap()));
        assert_eq!(p.quantized, quantized, "{name}: quantized");
        let window = case.f64s("window");
        assert_eq!([p.vmin, p.vmax], [window[0], window[1]], "{name}: quantile window");
        assert_eq!(p.centroid.to_vec(), case.i64s("centroid"), "{name}: centroid");
        let fit_affine: Vec<f64> = p.fit_affine.0.iter().flatten().copied().collect();
        assert_eq!(fit_affine, case.f64s("fit_affine"), "{name}: fit affine");

        // torch's float32 mean can sit one ulp from the correctly rounded one (see max_zscore)
        let image = case.array("image", Dtype::F32, f32_le);
        let max_diff = p.image.iter().zip(image.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff <= 1e-6, "{name}: normalized image differs by {max_diff}");

        let patch = case.params["patch"].as_u64().unwrap() as usize;
        let tokens = p.tokens(patch).unwrap();
        let coords: Vec<[u16; 3]> =
            case.i64s("coords").chunks(3).map(|c| [c[0] as u16, c[1] as u16, c[2] as u16]).collect();
        assert_eq!(tokens.coords, coords, "{name}: token coordinates");
        assert_eq!(tokens.len() as u64, case.params["tokens"].as_u64().unwrap());

        let labels = case.array("labels", Dtype::U8, |c| c[0]);
        let native = p.to_native(labels.view()).unwrap();
        assert_eq!(
            native,
            case.array("native_labels", Dtype::U8, |c| c[0]),
            "{name}: labels back in native space"
        );
    }
}

#[test]
fn to_native_rejects_volumes_off_the_grid() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let bytes = std::fs::read(dir.join("pipeline.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let scan = Nifti::from_bytes(tensors.tensor("pil_float32/scan").unwrap().data()).unwrap();
    let mut recipe = Recipe::nanobrain_eval();
    recipe.conform.fov = [24.0, 32.0, 24.0];
    let prepared = recipe.run(&scan).unwrap();
    let wrong = Array3::<u8>::zeros([3, 4, 5]);
    assert!(prepared.to_native(wrong.view()).is_err());
    assert!(prepared.tokens(5).is_err());
    assert!(prepared.mask_volume_ml > 0.0);
}
