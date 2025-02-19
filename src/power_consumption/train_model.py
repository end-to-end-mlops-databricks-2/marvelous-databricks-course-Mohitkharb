from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient
from power_consumption.utils import get_experiment_name, get_model_name
from power_consumption.config import ProjectConfig

mlflow.set_tracking_uri("databricks_uc")

def create_model_pipeline() -> Pipeline:
    """Create a pipeline with XGBoost regressor."""
    return Pipeline([
        ('xgboost', XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
    ])

def _train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Pipeline:
    """Train a single model and log parameters."""
    # Log model parameters
    xgb_params = pipeline.named_steps['xgboost'].get_params()
    mlflow.log_params(xgb_params)
    
    # Log feature names
    mlflow.log_param("features", list(X_train.columns))
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name=f"{model_name}"
    )
    
    return pipeline

def _evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a single model and log metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mape': percentage_error,
        'rmse': rmse,
        'r2': r2
    }
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    return metrics

def run_experiment(
    train_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    test_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    experiment_name: str,
    model_name: str
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Run MLflow experiment for training and evaluation."""
    models = {}
    results = {}
    
    # Set or create experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    for zone, (X_train, y_train) in train_data.items():
        X_test, y_test = test_data[zone]
        
        with mlflow.start_run(run_name=f"model_{zone}") as run:
            # Create and train model
            pipeline = create_model_pipeline()
            models[zone] = _train_model(pipeline, X_train, y_train, f"{model_name}_{zone}")
            
            # Evaluate model
            results[zone] = _evaluate_model(models[zone], X_test, y_test)
    
    return models, results

def train_and_evaluate(
    train_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    test_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    project_config: ProjectConfig
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """
    Train and evaluate models for all zones, with MLflow tracking and Unity Catalog registration.
    """

    experiment_name = get_experiment_name(workspace_location= project_config.data_settings.workspace_location, user_name=project_config.data_settings.user_name, type=project_config.ml_model_settings.type, use_case_name=project_config.use_case_name, branch_name=project_config.branch_name)
    model_name = get_model_name(use_case_name=project_config.use_case_name, type=project_config.ml_model_settings.type, branch_name=project_config.branch_name)
    # Run experiment with MLflow tracking
    models, results = run_experiment(train_data, test_data, experiment_name, model_name)
    
    # Register models in Unity Catalog
    # register_models_in_unity_catalog(
    #     models,
    #     catalog_name,
    #     schema_name,
    #     version_description="Power consumption prediction model"
    # )
    
    return models, results

def register_models_in_unity_catalog(
    models: Dict[str, Pipeline],
    catalog_name: str,
    schema_name: str,
    version_description: str = "Initial version"
) -> None:
    """
    Register models in Unity Catalog.
    
    Args:
        models: Dictionary of trained model pipelines
        catalog_name: Name of the Unity Catalog
        schema_name: Name of the schema in Unity Catalog
        version_description: Description for the model version
    """
    client = MlflowClient()
    
    for zone, model in models.items():
        model_name = f"{catalog_name}.{schema_name}.power_consumption_{zone}"
        
        # Register model in Unity Catalog
        try:
            registered_model = client.create_registered_model(model_name)
        except:
            registered_model = client.get_registered_model(model_name)
        
        # Create new version
        model_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name
        )
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=version_description
        )
