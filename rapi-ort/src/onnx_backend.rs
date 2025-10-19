use crate::model::{InputData, Model};
use crate::preprocess::{preprocess, PreprocessParams};
use anyhow::{anyhow, Context};
use async_trait::async_trait;
use ndarray::{Array, Ix2};
use ort::{self, session::Session};
use ort::value::TensorRef; // ndarrayのviewから入力テンソルを作る
use std::path::PathBuf;
use tokio::sync::Mutex; // 非同期下でのミュータブル借用を安全に扱う

/// ONNX backend implemented with the `ort` crate (v2 API).
pub struct OnnxBackend {
    // v2のSession::runはミュータブル参照を要求するため、Mutexで包む
    session: Mutex<Session>,
    pp: PreprocessParams,
    threshold: f32,
}

impl OnnxBackend {
    /// Create a new backend from a model file path and preprocessor params.
    pub fn new(model_path: PathBuf, pp: PreprocessParams) -> anyhow::Result<Self> {
        // v2: Environment不要。builder() -> commit_from_file() でロード
        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("failed to load ONNX model")?;

        Ok(Self {
            session: Mutex::new(session),
            pp,
            threshold: 0.5, // Python 実装に合わせる
        })
    }
}

#[async_trait]
impl Model for OnnxBackend {
    async fn predict(&self, x: &InputData) -> anyhow::Result<i32> {
        // 1) 前処理（6列）
        let row = preprocess(x, &self.pp);
        let arr: Array<f32, Ix2> =
            Array::from_shape_vec((1, 6), row.to_vec()).context("ndarray shape error")?;

        // 2) ndarray -> TensorRef にしてから inputs! へ渡す（v2）
        //    TensorRefは所有権を持たずビューから値を作れるのでコストが低い
        //    （公式ドキュメントのサンプルと同様の手順）
        let inputs = ort::inputs![TensorRef::from_array_view(arr.view())?];

        // 3) 推論実行（Sessionはミュータブル借用が必要なため、Mutexでロック）
        let mut session = self.session.lock().await;
        let outputs = session
            .run(inputs)
            .context("onnx session run failed")?;

        // 4) 先頭出力を f32 の ArrayView に変換（v2: try_extract_array）
        let view = outputs[0]
            .try_extract_array::<f32>()
            .context("failed to extract f32 array")?;

        // [1] or [1,1] を想定して先頭を取る
        let score = *view
            .as_slice()
            .and_then(|s| s.first())
            .ok_or_else(|| anyhow!("empty output tensor"))?;

        Ok(if score > self.threshold { 1 } else { 0 })
    }
}

