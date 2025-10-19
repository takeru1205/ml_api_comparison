use crate::model::InputData;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct PreprocessParams {
    // それぞれ [6] 長のベクトルを想定（列順は InputData と同じ）
    pub imputer_means: [f64; 6],
    pub scaler_means: [f64; 6],
    pub scaler_stds: [f64; 6],
}

// Python版の BaseMLModel と同じラベルエンコード: {"female":0,"male":1} を適用 :contentReference[oaicite:7]{index=7}
fn encode_sex(s: &str) -> f64 {
    match s {
        "female" => 0.0,
        "male" => 1.0,
        _ => 1.0, // 未知は便宜的に male 扱い（必要ならエラーに）
    }
}

// Python版は: LabelEncode → Imputer → Scaler の順で適用 
pub fn preprocess(x: &InputData, pp: &PreprocessParams) -> [f32; 6] {
    // 1) label encode
    let mut v = [
        x.Pclass as f64,
        encode_sex(&x.Sex),
        x.Age,
        x.SibSp as f64,
        x.Parch as f64,
        x.Fare,
    ];

    // 2) impute (単純平均埋め)
    for i in 0..6 {
        if v[i].is_nan() {
            v[i] = pp.imputer_means[i];
        }
    }

    // 3) scale (StandardScaler 相当: (x-mean)/std)
    for i in 0..6 {
        let std = pp.scaler_stds[i];
        v[i] = if std > 0.0 {
            (v[i] - pp.scaler_means[i]) / std
        } else {
            0.0
        };
    }

    // ONNXRuntime は f32 を好む
    [
        v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32, v[4] as f32, v[5] as f32,
    ]
}
