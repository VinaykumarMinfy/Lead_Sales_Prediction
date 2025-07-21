from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'lead_scoring_sagemaker_pipeline',
    default_args=default_args,
    description='Lead scoring pipeline on SageMaker using Redshift data',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
) as dag:

    s3_bucket = 'airflow132'  
    mlflow_tracking_uri = 'arn:aws:sagemaker:ap-south-1:820028474211:mlflow-tracking-server/mlflow-sg'
    model_name = 'LeadConversionModel12'
    local_script_dir = '/home/ec2-user/SageMaker/custom-scripts'

    def fetch_data_from_redshift(**kwargs):
        import boto3
        import pandas as pd
        import time

        region = 'ap-south-1'
        workgroup_name = 'default-workgroup'
        database_name = 'dev'
        secret_arn = 'arn:aws:secretsmanager:ap-south-1:820028474211:secret:redshift-secret-nxgY56'
        sql = 'SELECT * FROM lead_pre LIMIT 100'

        client = boto3.client('redshift-data', region_name=region)
        response = client.execute_statement(
            WorkgroupName=workgroup_name,
            Database=database_name,
            SecretArn=secret_arn,
            Sql=sql
        )
        statement_id = response['Id']

        desc = client.describe_statement(Id=statement_id)
        while desc['Status'] not in ['FINISHED', 'FAILED', 'ABORTED']:
            time.sleep(1)
            desc = client.describe_statement(Id=statement_id)
        if desc['Status'] != 'FINISHED':
            raise Exception(f"Query failed: {desc}")

        result = client.get_statement_result(Id=statement_id)
        columns = [col['name'] for col in result['ColumnMetadata']]
        rows = result['Records']
        data = []
        for row in rows:
            data.append([list(col.values())[0] if col else None for col in row])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f's3://{s3_bucket}/raw/lead_scoring_{kwargs["ds"]}.csv', index=False)

    def apply_feature_engineering(**kwargs):
        from feature_engineering import apply_feature_engineering
        apply_feature_engineering(
            s3_bucket=s3_bucket,
            s3_input_key=f"raw/lead_scoring_{kwargs['ds']}.csv",
            s3_output_key=f"engineered/lead_scoring_{kwargs['ds']}.csv"
        )

    def preprocess_data(**kwargs):
        from data_preprocessing import preprocess_data
        preprocess_data(
            s3_bucket=s3_bucket,
            s3_input_key=f"engineered/lead_scoring_{kwargs['ds']}.csv",
            s3_output_key=f"preprocessed/lead_scoring_{kwargs['ds']}.parquet"
        )

    def train_and_register_model(**kwargs):
        from modeling import train_log_and_shap_classification
        train_log_and_shap_classification(
            s3_bucket=s3_bucket,
            s3_data_key=f"preprocessed/lead_scoring_{kwargs['ds']}.parquet",
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
            local_script_dir=local_script_dir
        )

    def generate_evidently_report(**kwargs):
        from monitoring import generate_evidently_report
        generate_evidently_report(
            s3_bucket=s3_bucket,
            s3_reference_key='reference/lead_scoring_reference.parquet',
            s3_current_key=f"preprocessed/lead_scoring_{kwargs['ds']}.parquet",
            report_path=f"s3://{s3_bucket}/reports/evidently_{kwargs['ds']}.html"
        )

    fetch_data_task = PythonOperator(task_id='fetch_data_from_redshift', python_callable=fetch_data_from_redshift)
    feature_engineering_task = PythonOperator(task_id='apply_feature_engineering', python_callable=apply_feature_engineering)
    preprocess_task = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data)
    train_model_task = PythonOperator(task_id='train_and_register_model', python_callable=train_and_register_model)
    monitoring_task = PythonOperator(task_id='generate_evidently_report', python_callable=generate_evidently_report)

    fetch_data_task >> feature_engineering_task >> preprocess_task >> train_model_task >> monitoring_task
