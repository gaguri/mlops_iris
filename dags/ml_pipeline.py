from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import sys

sys.path.append('/opt/airflow/scripts')
import drift_retrain

# Проверка дрифта и передача данных дальше
def _check_drift_task(**context):
    drift_detected = drift_retrain.check_for_drift()
    
    ti = context['ti']
    ti.xcom_push(key='drift_status', value=drift_detected)
    return drift_detected

# Узел ветвления
def _branching_logic(**context):
    """Задача: выбрать путь на основе результата проверки дрифта"""
    ti = context['ti']
    drift_detected = ti.xcom_pull(key='drift_status', task_ids='check_drift')
    
    if drift_detected:
        print("!!! Drift detected! Moving to retraining.")
        return 'retrain_model'
    else:
        print("--- No drift. Everything is stable.")
        return 'skip_retraining'

# Параметры автоматизированного конвейера
with DAG(
    dag_id="iris_pipeline",
    start_date=days_ago(1),
    schedule_interval="@daily",
    catchup=False,
    tags=['mlops', 'ab_test']
) as dag:

    start = EmptyOperator(task_id="start")

    # Этап 1. Мониторинг
    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=_check_drift_task
    )

    # Этап 2. Ветвление
    branching = BranchPythonOperator(
        task_id="decide_next_step",
        python_callable=_branching_logic
    )

    # Этап 3.1. Переобучение
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=drift_retrain.train_and_register_model
    )

    # Этап 3.2. Пропуск
    skip = EmptyOperator(task_id="skip_retraining")

    # Этап 4. Завершение
    end = EmptyOperator(
        task_id="end", 
        trigger_rule="none_failed_min_one_success"
    )

    # Архитектура графа
    start >> check_drift >> branching
    branching >> retrain >> end
    branching >> skip >> end