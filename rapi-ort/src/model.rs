use std::sync::{Arc, Mutex};
use ndarray::{Array1, Array2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

#[derive(Clone, Copy)]
pub enum ModelType { TorchOnnx, LgbOnnx }


pub struct InferModel {
_ty: ModelType,
session: Arc<Mutex<Session>>,
input_name: String,
output_name: String,
threshold: f32,
}


pub struct ModelPool {
pub torch_onnx: Arc<InferModel>,
pub lgb_onnx: Arc<InferModel>,
pub preproc: Arc<crate::preprocess::Preprocessor>,
}



impl InferModel {
pub fn new(ty: ModelType, model_path: &str) -> anyhow::Result<Self> {
// ort v2: Session::builder() -> commit_from_memory
let model_bytes = std::fs::read(model_path)?;
let sess = Session::builder()
    .map_err(|e| anyhow::anyhow!("Failed to create session builder: {:?}", e))?
    .with_optimization_level(GraphOptimizationLevel::Level3)
    .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {:?}", e))?
    .with_intra_threads(num_cpus::get())
    .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {:?}", e))?
    .commit_from_memory(&model_bytes)
    .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {:?}", model_path, e))?;

    // 入出力名はモデルから取得（固定名に依存しない）
    let input_name = sess.inputs[0].name.clone();
    let output_name = sess.outputs[0].name.clone();

    Ok(Self { _ty: ty, session: Arc::new(Mutex::new(sess)), input_name, output_name, threshold: 0.5 })
}


pub fn predict(&self, x: &Array1<f32>) -> i32 {
let batch: Array2<f32> = x.clone().insert_axis(ndarray::Axis(0)); // [1,6]

// ort v2: ndarray そのままでは渡せないので Tensor を作る
let input: Tensor<f32> = Tensor::from_array(batch).expect("to tensor");
let mut session = self.session.lock().expect("lock session");
let outputs = session
    .run(ort::inputs![ self.input_name.as_str() => input ])
    .expect("run");

// モデルタイプに応じて出力を処理
match self._ty {
    ModelType::TorchOnnx => {
        // PyTorchモデルはf32を返す（確率）
        let y: ndarray::ArrayD<f32> = outputs[0].try_extract_array().expect("extract f32").to_owned();
        let val = *y.first().unwrap_or(&0.0);
        (val > self.threshold) as i32
    }
    ModelType::LgbOnnx => {
        // LightGBMモデルはi64を返す（クラスラベル）
        let y: ndarray::ArrayD<i64> = outputs[0].try_extract_array().expect("extract i64").to_owned();
        *y.first().unwrap_or(&0) as i32
    }
}
}
}
