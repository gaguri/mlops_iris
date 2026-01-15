import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pycaret.classification import setup, compare_models
import mlflow
from mlflow.tracking import MlflowClient
import os

# Глобальные настройки системы
MLFLOW_URI = "http://mlflow:5000"
MODEL_NAME = "iris_model"
ALIAS_CHALLENGER = "challenger"

mlflow.set_tracking_uri(MLFLOW_URI)

# Предобработка имен признаков
def clean_column_names(df):
    df.columns = [c.replace(' (cm)', '').replace(' ', '_').lower() for c in df.columns]
    return df

# Использование теста Колмогорова-Смирнова для определения дрифта
def check_for_drift(reference_path='/opt/airflow/data/reference_data.csv', 
                    current_path='/opt/airflow/data/current_data.csv', 
                    threshold=0.05):
    print(f"Checking drift between {reference_path} and {current_path}")
    
    ref = clean_column_names(pd.read_csv(reference_path))
    cur = clean_column_names(pd.read_csv(current_path))
    
    drifts = []
    features = [c for c in ref.columns if c != 'target']
    
    for col in features:
        stat, p_value = ks_2samp(ref[col], cur[col])
        drifts.append(p_value < 0.05)
    
    drift_share = np.mean(drifts)
    print(f"Drifted features share: {drift_share:.2f}")
    
    return bool(drift_share > threshold)

# Переобучение и регистрация в Mlflow
def train_and_register_model(data_path='/opt/airflow/data/current_data.csv'):
    print("Starting training with PyCaret...")
    df = clean_column_names(pd.read_csv(data_path))
    # Контекстный менеджер для автоматического трекинга всех метрик и параметров
    with mlflow.start_run(run_name="retraining_run"):
        # Инициализация эксперимента PyCaret
        s = setup(df, target='target', 
                  log_experiment=True, 
                  experiment_name='iris_ab_test', 
                  html=False, verbose=False)
        
        # Сравнение моделей, выбор лучшей по Accuracy
        best_model = compare_models()
        print(f"Best model found: {best_model}")
        
        # Логирование артефакта и автоматическое создание версии в реестре
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model", 
            registered_model_name=MODEL_NAME
        )
        
        # Алиас 'challenger' новой версии
        client = MlflowClient()
        new_version = model_info.registered_model_version
        
        client.set_registered_model_alias(MODEL_NAME, ALIAS_CHALLENGER, str(new_version))
        print(f"Successfully registered Version {new_version} as '{ALIAS_CHALLENGER}'")
        
        return {"version": new_version, "alias": ALIAS_CHALLENGER}

if __name__ == "__main__":
    if check_for_drift():
        train_and_register_model()
    else:
        print("Everything is fine, skipping training.")