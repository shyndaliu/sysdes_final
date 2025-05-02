from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os

default_args = {
    'owner': 'uldana',
    'start_date': days_ago(1),
    'retries': 1,
}

PROJECT_ROOT = '/opt/project'


CLICKHOUSE_PATH = os.path.join(PROJECT_ROOT, 'clickhouse')
MLFLOW_PATH = os.path.join(PROJECT_ROOT, 'mlflow')
INFERENCE_API_PATH = os.path.join(PROJECT_ROOT, 'inference-api')
TRAINING_SERVER_PATH = os.path.join(PROJECT_ROOT, 'training-server')
EVALUATION_SERVER_PATH = os.path.join(PROJECT_ROOT, 'evaluation-server')

dag = DAG(
    'sysdes_component_pipeline',
    default_args=default_args,
    description='Orchestrate sysdes components with Docker and Airflow',
    schedule_interval=None,
    catchup=False,
)

# 1. Clickhouse
start_clickhouse = BashOperator(
    task_id='start_clickhouse',
    bash_command=f'cd {CLICKHOUSE_PATH} && docker-compose up -d --build',
    dag=dag,
)

# 2. MLflow (depends on Clickhouse)
start_mlflow = BashOperator(
    task_id='start_mlflow',
    bash_command=f'cd {MLFLOW_PATH} && docker-compose up -d --build',
    dag=dag,
)

# 3. Build inference-api actually its needs to be running constantly и мы просто накатываем новые модельки но тут больше для простоты запуска
build_inference = BashOperator(
    task_id='build_inference_api',
    bash_command=f'cd {INFERENCE_API_PATH} && docker build -t inference-api .',
    dag=dag,
)

# 4. Build training-server
build_training = BashOperator(
    task_id='build_training_server',
    bash_command=f'cd {TRAINING_SERVER_PATH} && docker build -t training-server .',
    dag=dag,
)

# 5. Run training-server
run_training = BashOperator(
    task_id='run_training',
    bash_command='docker run --rm --network="host" training-server',
    dag=dag,
)

# 6. Build evaluation-server
build_evaluation = BashOperator(
    task_id='build_evaluation_server',
    bash_command=f'cd {EVALUATION_SERVER_PATH} && docker build -t evaluation-server .',
    dag=dag,
)

# 7. Run evaluation-server
run_evaluation = BashOperator(
    task_id='run_evaluation',
    bash_command='docker run --rm --network="host" evaluation-server',
    dag=dag,
)

# 8. Run inference-api
run_inference = BashOperator(
    task_id='run_inference_api',
    bash_command=f'cd {INFERENCE_API_PATH} && docker run -d -p 5002:5002 inference-api',
    dag=dag,
)

# Define task dependencies
start_clickhouse >> start_mlflow >> [build_inference, build_training]
build_training >> run_training >> build_evaluation >> run_evaluation >> run_inference
