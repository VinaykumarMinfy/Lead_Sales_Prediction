from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2025, 1, 1),
    'catchup': False
}

with DAG('lead_conversion_pipeline',
         default_args=default_args,
         description='Run Lead Conversion ML pipeline',
         schedule_interval='@daily',
         tags=['lead', 'ml', 'sagemaker']) as dag:

    run_main_script = BashOperator(
        task_id='lead_prediction.ipynb',
        bash_command='python /opt/airflow/src/lead_prediction.ipynb'
    )
