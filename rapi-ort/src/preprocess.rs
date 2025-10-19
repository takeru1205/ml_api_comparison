use ndarray::{arr1, Array1};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;


#[derive(Debug, Clone, Deserialize)]
pub struct InputData {
pub Pclass: i32,
pub Sex: String,
pub Age: f32,
pub SibSp: i32,
pub Parch: i32,
pub Fare: f32,
}


#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorParams {
/// 列順: [Pclass, Sex, Age, SibSp, Parch, Fare]
pub imputer_fill: Option<[f32; 6]>,
pub scaler_mean: Option<[f32; 6]>,
pub scaler_scale: Option<[f32; 6]>,
}


#[derive(Debug, Clone)]
pub struct Preprocessor {
params: PreprocessorParams,
}


impl Preprocessor {
pub fn identity() -> Self { Self { params: PreprocessorParams { imputer_fill: None, scaler_mean: None, scaler_scale: None } } }
pub fn load_json(path: &str) -> anyhow::Result<Self> {
let bytes = std::fs::read(path)?;
let params: PreprocessorParams = serde_json::from_slice(&bytes)?;
Ok(Self { params })
}


pub fn transform(&self, x: &InputData) -> Array1<f32> {
// Sex を label encode: {female:0, male:1}
let sex = match x.Sex.as_str() { "female" => 0.0, "male" => 1.0, _ => 1.0 };
let mut v = [x.Pclass as f32, sex, x.Age, x.SibSp as f32, x.Parch as f32, x.Fare];


// Simple impute
if let Some(fill) = &self.params.imputer_fill { for (t, f) in v.iter_mut().zip(fill.iter()) { if !t.is_finite() { *t = *f; } } }
// Standard scale: (x - mean) / scale
if let (Some(mean), Some(scale)) = (&self.params.scaler_mean, &self.params.scaler_scale) {
for i in 0..v.len() { v[i] = (v[i] - mean[i]) / scale[i].max(1e-6); }
}
arr1(&v)
}
}
