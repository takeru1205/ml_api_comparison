use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*; // TypedModel, TypedRunnableModel, Tensor, tvec!, ArrayView など

use crate::model::{InputData, Model};

/// 前処理パラメータ（SimpleImputer + StandardScaler 相当）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    /// SimpleImputer の統計量（列順: Pclass, Sex, Age, SibSp, Parch, Fare）
    pub imputer_statistics: [f32; 6],
    /// StandardScaler: 平均
    pub scaler_mean: [f32; 6],
    /// StandardScaler: 標準偏差（scale_）
    pub scaler_scale: [f32; 6],
}

/// 入力JSONの両フォーマットを受け付ける（untagged で自動判別）
#[derive(Deserialize)]
#[serde(untagged)]
enum PpEither {
    /// フラット形式:
    /// {"imputer_means":[...], "scaler_means":[...], "scaler_stds":[...]}
    Flat {
        imputer_means: Vec<f64>,
        scaler_means: Vec<f64>,
        scaler_stds: Vec<f64>,
    },
    /// プレビュー形式:
    /// {"type":"dict","size":...,"preview":{"imputer":{"statistics_":[...]}, "scaler":{"mean_":[...], "scale_":[...]}}}
    Preview { preview: Preview },
}

#[derive(Deserialize)]
struct Preview {
    imputer: Imputer,
    scaler: Scaler,
}
#[derive(Deserialize)]
struct Imputer {
    #[serde(rename = "statistics_")]
    statistics_: Vec<f64>,
}
#[derive(Deserialize)]
struct Scaler {
    #[serde(rename = "mean_")]
    mean_: Vec<f64>,
    #[serde(rename = "scale_")]
    scale_: Vec<f64>,
}

impl Preprocessor {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let either: PpEither = serde_json::from_str(&json)?;

        let to6 = |v: Vec<f64>, name: &str| -> Result<[f32; 6]> {
            anyhow::ensure!(v.len() == 6, "expected 6 elements in {}, got {}", name, v.len());
            Ok([
                v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32, v[4] as f32, v[5] as f32,
            ])
        };

        match either {
            PpEither::Flat {
                imputer_means,
                scaler_means,
                scaler_stds,
            } => Ok(Self {
                imputer_statistics: to6(imputer_means, "imputer_means")?,
                scaler_mean: to6(scaler_means, "scaler_means")?,
                scaler_scale: to6(scaler_stds, "scaler_stds")?,
            }),
            PpEither::Preview { preview } => Ok(Self {
                imputer_statistics: to6(preview.imputer.statistics_, "statistics_")?,
                scaler_mean: to6(preview.scaler.mean_, "mean_")?,
                scaler_scale: to6(preview.scaler.scale_, "scale_")?,
            }),
        }
    }

    /// 1レコードを欠損補完→標準化し、shape=[1,6] のテンソルにする
    pub fn transform(&self, x: &InputData) -> Tensor {
        // Sex を 0/1 にエンコード（未知は補完値）
        let sex = match x.Sex.as_str() {
            "male" | "Male" | "M" | "m" => 1.0f32,
            "female" | "Female" | "F" | "f" => 0.0f32,
            _ => self.imputer_statistics[1],
        };

        // 欠損は imputer の統計量で補完
        let raw = [
            x.Pclass as f32,
            sex,
            if x.Age.is_nan() { self.imputer_statistics[2] } else { x.Age as f32 },
            x.SibSp as f32,
            x.Parch as f32,
            if x.Fare.is_nan() { self.imputer_statistics[5] } else { x.Fare as f32 },
        ];

        // StandardScaler: (x - mean) / scale
        let mut stdz = [0f32; 6];
        for i in 0..6 {
            let denom = if self.scaler_scale[i] == 0.0 { 1.0 } else { self.scaler_scale[i] };
            stdz[i] = (raw[i] - self.scaler_mean[i]) / denom;
        }

        // ndarray -> Tensor（tract は IntoTensor 実装あり）
        let arr = tract_onnx::prelude::tract_ndarray::Array2::<f32>::from_shape_vec((1, 6), stdz.to_vec())
            .expect("shape 1x6");
        arr.into_tensor()
    }
}

/// tract バックエンド（RunnablePlan を保持）
#[derive(Clone)]
pub struct OnnxBackend {
    /// 実行可能モデル（TypedRunnableModel は SimplePlan の別名）: 型根拠は prelude の型一覧。 
    model: TypedRunnableModel<TypedModel>,
    pre: Arc<Preprocessor>,
    threshold: f32,
}

impl OnnxBackend {
    /// `model_path`: ONNX モデルファイル, `preprocessor_json`: 前処理パラメータJSON（フラット or プレビュー）
    pub fn new<P: AsRef<Path>>(model_path: P, preprocessor_json: P) -> Result<Self> {
        // 1) モデルをロード
        let mut model = tract_onnx::onnx()
            .model_for_path(&model_path)
            .with_context(|| format!("failed to load onnx: {}", model_path.as_ref().display()))?;

        // 2) 入力型・形状を設定（f32, [1,6]）
        model.set_input_fact(0, f32::fact([1, 6]).into())?;

        // 3) 最適化 → 実行計画化
        let runnable: TypedRunnableModel<TypedModel> = model
            .into_optimized()
            .context("optimize failed")?
            .into_runnable()
            .context("into_runnable failed")?;

        // 4) 前処理パラメータ（両フォーマット対応）
        let pre = Arc::new(Preprocessor::from_json_file(preprocessor_json)?);

        Ok(Self {
            model: runnable,
            pre,
            threshold: 0.5, // Python 実装に合わせる
        })
    }
}

#[async_trait]
impl Model for OnnxBackend {
    async fn predict(&self, x: &InputData) -> Result<i32> {
        // 1) 前処理して Tensor を用意
        let input: Tensor = self.pre.transform(x);

        // 2) 推論実行（tvec![Tensor] を渡す）
        let mut outputs = self
            .model
            .run(tvec![input.into()])
            .context("tract run failed")?;

        // 3) 出力を取り出し
        anyhow::ensure!(!outputs.is_empty(), "no outputs from model");
        let out_tensor = outputs.remove(0);
        let y = out_tensor
            .to_array_view::<f32>()
            .context("output to f32 view failed")?;

        // 4) [1] or [1,1] を想定してスカラー化
        let score = if y.ndim() == 2 { y[[0, 0]] } else { y[0] };
        Ok((score >= self.threshold) as i32)
    }
}

