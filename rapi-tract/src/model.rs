use serde::{Deserialize, Serialize};
use std::sync::Arc;
use async_trait::async_trait;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
pub struct InputData {
    pub Pclass: i32,
    pub Sex: String,
    pub Age: f64,
    pub SibSp: i32,
    pub Parch: i32,
    pub Fare: f64,
}

#[derive(Debug, Serialize)]
pub struct PredictResp {
    pub prediction: i32,
}

#[async_trait]
pub trait Model: Send + Sync {
    async fn predict(&self, x: &InputData) -> anyhow::Result<i32>;
}

pub type DynModel = Arc<dyn Model>;

