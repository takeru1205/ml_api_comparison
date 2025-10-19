use std::sync::Arc;
use ndarray::{Array1, Array2};
use tract_onnx::prelude::*;

#[derive(Clone, Copy)]
pub enum ModelType { TorchOnnx, LgbOnnx }


pub struct InferModel {
_ty: ModelType,
model: Arc<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>,
threshold: f32,
}


pub struct ModelPool {
pub torch_onnx: Arc<InferModel>,
pub lgb_onnx: Arc<InferModel>,
pub preproc: Arc<crate::preprocess::Preprocessor>,
}



impl InferModel {
pub fn new(ty: ModelType, model_path: &str) -> anyhow::Result<Self> {
// tract: ONNXモデルを読み込み、入力形状を指定して最適化
let mut model = tract_onnx::onnx()
    .model_for_path(model_path)
    .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {:?}", model_path, e))?;

    // 入力形状を明示的に指定 [batch_size, features] = [1, 6]
    model
        .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1, 6]))
        .map_err(|e| anyhow::anyhow!("Failed to set input fact: {:?}", e))?;

    // 型推論を完了させる
    let model = model
        .into_typed()
        .map_err(|e| anyhow::anyhow!("Failed to type model: {:?}", e))?
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("Failed to optimize model: {:?}", e))?
        .into_runnable()
        .map_err(|e| anyhow::anyhow!("Failed to make model runnable: {:?}", e))?;

    Ok(Self { _ty: ty, model: Arc::new(model), threshold: 0.5 })
}


pub fn predict(&self, x: &Array1<f32>) -> i32 {
let batch: Array2<f32> = x.clone().insert_axis(ndarray::Axis(0)); // [1,6]

// tract: ndarrayをTensorに変換して推論実行
let input = tract_ndarray::Array2::from_shape_vec(batch.dim(), batch.iter().copied().collect())
    .expect("to tract array");
let outputs = self.model
    .run(tvec![input.into_tensor().into()])
    .expect("run");

// モデルタイプに応じて出力を処理
match self._ty {
    ModelType::TorchOnnx => {
        // PyTorchモデルはf32を返す（確率）
        let y = outputs[0].to_array_view::<f32>().expect("extract f32");
        let val = *y.iter().next().unwrap_or(&0.0);
        (val > self.threshold) as i32
    }
    ModelType::LgbOnnx => {
        // LightGBMモデル: 必要に応じてf32にキャストして処理
        let output = outputs[0].cast_to_dt(f32::datum_type()).expect("cast to f32");
        let y = output.to_array_view::<f32>().expect("extract f32");
        let val = *y.iter().next().unwrap_or(&0.0);
        (val > self.threshold) as i32
    }
}
}
}
