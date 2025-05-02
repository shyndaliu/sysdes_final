import subprocess
import os

def stop_services():

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    CLICKHOUSE_PATH = os.path.join(PROJECT_ROOT, 'clickhouse')
    MLFLOW_PATH = os.path.join(PROJECT_ROOT, 'mlflow')
    INFERENCE_API_PATH = os.path.join(PROJECT_ROOT, 'inference-api')
    directories = [
        CLICKHOUSE_PATH,
        MLFLOW_PATH, 
        INFERENCE_API_PATH
    ]
    
    for directory in directories:
        try:
            print(f"Stopping services in {directory}...")
            subprocess.run(['docker-compose', 'down', '-v'], cwd=directory, check=True)
            print(f"Successfully stopped services in {directory}.")
        except subprocess.CalledProcessError as e:
            print(f"Error stopping services in {directory}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Run the function to stop services
stop_services()
