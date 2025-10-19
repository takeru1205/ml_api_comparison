mod model;
mod onnx_backend;
mod preprocess;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use model::{DynModel, InputData, PredictResp};
use onnx_backend::OnnxBackend;
use preprocess::PreprocessParams;
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    torch: DynModel,
    lgb: DynModel,
    torch_onnx: DynModel,
    lgb_onnx: DynModel,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "axum=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // === env config ===
    let models_root = env::var("MODELS_ROOT").unwrap_or_else(|_| "../models".into());
    let pp_path = env::var("PREPROCESSOR_JSON")
        .unwrap_or_else(|_| format!("{}/mlp_torch_preprocess.json", models_root));
    let torch_onnx_path =
        env::var("TORCH_ONNX").unwrap_or_else(|_| format!("{}/mlp_torch.onnx", models_root));
    let lgb_onnx_path =
        env::var("LGB_ONNX").unwrap_or_else(|_| format!("{}/titanic_lgb.onnx", models_root));

    // === load preprocess params ===
    let pp: PreprocessParams = {
        let data = std::fs::read_to_string(&pp_path)?;
        serde_json::from_str(&data)?
    };

    // === backends (torch/lgb も内部は ONNX を使用) ===
    let torch_backend = Arc::new(OnnxBackend::new(PathBuf::from(&torch_onnx_path), pp.clone())?);
    let lgb_backend = Arc::new(OnnxBackend::new(PathBuf::from(&lgb_onnx_path), pp.clone())?);

    let state = AppState {
        torch: torch_backend.clone(),
        lgb: lgb_backend.clone(),
        torch_onnx: torch_backend,
        lgb_onnx: lgb_backend,
    };

    // === router (Flask互換ルート) ===
    let app = Router::new()
        .route("/", get(root))
        .route("/predict/torch/", post(predict_torch))
        .route("/predict/lgb/", post(predict_lgb))
        .route("/predict/torch_onnx/", post(predict_torch_onnx))
        .route("/predict/lgb_onnx/", post(predict_lgb_onnx))
        .with_state(state);

    // === serve (Axum 0.7) ===
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app).await?; // axum 0.7 の基本形

    Ok(())
}

async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({"message": "Hello axum"}))
}

async fn predict_impl(model: &DynModel, body: InputData) -> anyhow::Result<PredictResp> {
    let y = model.predict(&body).await?;
    Ok(PredictResp { prediction: y })
}

async fn predict_torch(
    State(st): State<AppState>,
    Json(body): Json<InputData>,
) -> Result<Json<PredictResp>, axum::http::StatusCode> {
    predict_impl(&st.torch, body)
        .await
        .map(Json)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)
}

async fn predict_lgb(
    State(st): State<AppState>,
    Json(body): Json<InputData>,
) -> Result<Json<PredictResp>, axum::http::StatusCode> {
    predict_impl(&st.lgb, body)
        .await
        .map(Json)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)
}

async fn predict_torch_onnx(
    State(st): State<AppState>,
    Json(body): Json<InputData>,
) -> Result<Json<PredictResp>, axum::http::StatusCode> {
    predict_impl(&st.torch_onnx, body)
        .await
        .map(Json)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)
}

async fn predict_lgb_onnx(
    State(st): State<AppState>,
    Json(body): Json<InputData>,
) -> Result<Json<PredictResp>, axum::http::StatusCode> {
    predict_impl(&st.lgb_onnx, body)
        .await
        .map(Json)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)
}

