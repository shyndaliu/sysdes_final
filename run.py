import os
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLICKHOUSE_PATH = os.path.join(PROJECT_ROOT, 'clickhouse')
MLFLOW_PATH = os.path.join(PROJECT_ROOT, 'mlflow')
INFERENCE_API_PATH = os.path.join(PROJECT_ROOT, 'inference-api')
TRAINING_SERVER_PATH = os.path.join(PROJECT_ROOT, 'training-server')
EVALUATION_SERVER_PATH = os.path.join(PROJECT_ROOT, 'evaluation-server')

def run_command(command, cwd=None):
    """Run shell command and wait for it to complete."""
    process = subprocess.Popen(command, shell=True, cwd=cwd)
    process.communicate()  # Wait for command to finish
    if process.returncode != 0:
        raise Exception(f"Command failed: {command}")
    print(f"Command succeeded: {command}")

def wait_for_service(service_name, timeout=300):
    """Wait for a service to become available (e.g., Clickhouse)."""
    print(f"Waiting for {service_name} to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Replace with actual command to check if the service is ready (e.g., curl or docker ps)
            subprocess.check_output(['docker', 'ps', '-f', f'name={service_name}'])
            print(f"{service_name} is ready!")
            return
        except subprocess.CalledProcessError:
            time.sleep(10)  # Wait before retrying
    raise TimeoutError(f"{service_name} did not become available in time.")

def main():
    try:
        # Step 1: Start Clickhouse
        print("Starting Clickhouse...")
        run_command(f'cd {CLICKHOUSE_PATH} && docker-compose up -d --build')

        # Step 2: Wait for Clickhouse to be ready
        wait_for_service("clickhouse")

        # Step 3: Start MLflow
        print("Starting MLflow...")
        run_command(f'cd {MLFLOW_PATH} && docker-compose up -d --build')

        # Step 4: Wait for MLflow to be ready
        wait_for_service("mlflow")

        # Step 5: Build Inference API
        print("Building Inference API...")
        run_command(f'cd {INFERENCE_API_PATH} && docker build -t inference-api .')

        # Step 6: Build Training Server
        print("Building Training Server...")
        run_command(f'cd {TRAINING_SERVER_PATH} && docker build -t training-server .')

        # Step 7: Run Training Server
        print("Running Training Server...")
        run_command('docker run --rm --network="host" training-server')

        # Step 8: Build Evaluation Server
        print("Building Evaluation Server...")
        run_command(f'cd {EVALUATION_SERVER_PATH} && docker build -t evaluation-server .')

        # Step 9: Run Evaluation Server
        print("Running Evaluation Server...")
        run_command('docker run --rm --network="host" evaluation-server')

        # Step 10: Run Inference API
        print("Running Inference API...")
        run_command(f'cd {INFERENCE_API_PATH} && docker run -d -p 5002:5002 inference-api')

        print("All tasks completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
