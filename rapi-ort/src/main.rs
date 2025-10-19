use std::{net::SocketAddr, sync::Arc};


use axum::{routing::{get, post}, Json, Router};
use tracing::{error, info};


mod model;
mod preprocess;


use model::{InferModel, ModelPool, ModelType};
use preprocess::{InputData, Preprocessor};


#[tokio::main]
async fn main() -> anyhow::Result<()> {
tracing_subscriber::fmt()
.with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
.with_target(false)
.compact()
.init();


// モデル/前処理の初期化
let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "../models".to_string());
let torch_onnx = format!("{model_dir}/mlp_torch.onnx");
let lgb_onnx = format!("{model_dir}/titanic_lgb.onnx");
let preproc_json = std::env::var("PREPROCESSOR_JSON").unwrap_or_else(|_| format!("{model_dir}/mlp_torch_preprocess.json"));


let preproc = Arc::new(Preprocessor::load_json(&preproc_json).unwrap_or_else(|e| {
error!(error = %e, path = %preproc_json, "failed to load preprocessor; fallback to identity");
Preprocessor::identity()
}));


// セッションは使い回す（起動時にロード）
let torch_model = Arc::new(InferModel::new(ModelType::TorchOnnx, &torch_onnx)?);
let lgb_model = Arc::new(InferModel::new(ModelType::LgbOnnx, &lgb_onnx)?);


// 軽量プール（複数同時推論に備え clone で Arc 参照を配布）
let pool = Arc::new(ModelPool { torch_onnx: torch_model, lgb_onnx: lgb_model, preproc });


let app = Router::new()
.route("/", get(|| async { Json(serde_json::json!({"message": "Hello axum"})) }))
.route("/predict/torch_onnx/", post(predict_torch_onnx))
.route("/predict/lgb_onnx/", post(predict_lgb_onnx))
.with_state(pool.clone());


// ポートは PORT があれば尊重
let port: u16 = std::env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8000);
let addr: SocketAddr = ([0, 0, 0, 0], port).into();
info!(%addr, "listening");


let server = axum::serve(
tokio::net::TcpListener::bind(addr).await?,
app.into_make_service(),
);


// Ctrl+C で優雅に終了
tokio::select! {
r = server => { r?; }
_ = tokio::signal::ctrl_c() => { info!("shutdown"); }
}
Ok(())
}


async fn predict_torch_onnx(
axum::extract::State(pool): axum::extract::State<Arc<ModelPool>>,
Json(input): Json<InputData>,
) -> Json<serde_json::Value> {
let x = pool.preproc.transform(&input);
let y = pool.torch_onnx.predict(&x);
Json(serde_json::json!({"prediction": y}))
}


async fn predict_lgb_onnx(
axum::extract::State(pool): axum::extract::State<Arc<ModelPool>>,
Json(input): Json<InputData>,
) -> Json<serde_json::Value> {
let x = pool.preproc.transform(&input);
let y = pool.lgb_onnx.predict(&x);
Json(serde_json::json!({"prediction": y}))
}
