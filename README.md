# Project README

## Overview

This project orchestrates the setup of various services like **ClickHouse**, **MLflow**, **Inference API**, **Training Server**, and **Evaluation Server** using Docker containers. It provides an end-to-end pipeline that includes model training, evaluation, and inference, while allowing you to interact with the services via simple scripts.

---

## Setup Instructions

### Prerequisites

Before running the project, ensure you have the following installed:

- Docker
- Docker Compose
- Python (for the script that manages the project)
- Git (for cloning the repository)

### Clone the Repository

```
git clone https://github.com/shyndaliu/sysdes_final.git
cd sysdes_final
```

### Starting the Project

To start all services, simply run the following Python script:

```
python start.py
```


This will automatically start the necessary Docker containers for **ClickHouse**, **MLflow**, **Inference API**, **Training Server**, and **Evaluation Server**.

- **ClickHouse**: Starts the ClickHouse service.
- **MLflow**: Starts the MLflow service with UI accessible at [localhost:5001](http://localhost:5001).
- **Inference API**: Starts the Inference API on port 5002.
- **Training Server**: Starts the training server.
- **Evaluation Server**: Starts the evaluation server.

### Stopping the Project

To stop all running services, you can use the following Python script:

```
python stop.py
```


This will stop all Docker containers for the project.


### Starting component

The project uses Docker containers to run various services. You can also run them separately

#### Step 1: Navigate to the relevant directories

Ensure you're in the root project directory (where the `docker-compose.yml` or `Dockerfile` files are located) before running the following commands.

---

## Dataset

The data used in the project is provided in the following GitHub repository:

- **Dataset URL**: [Piki Music Dataset](https://github.com/sstoikov/piki-music-dataset/tree/main/data)

Dataset is prepared to use, script is included no need to run anything

---

## Making a Request to the Inference API

Once the services are running, you can interact with the **Inference API**. Here’s an example of how to make a request to the `/recommend` endpoint:

```
curl --location 'http://localhost:5002/recommend' \
--header 'Content-Type: application/json' \
--data '{
    "user_id": 3721203,
    "top_k" : 10
}'
```


### Expected Response

If the request is successful, you will receive a response like this:
```
{
    "recommendations": [
        6767,
        8108902,
        78335,
        8128433,
        78200,
        11285,
        5858183,
        5556060,
        257,
        5424340
    ],
    "user_id": 3721203
}
```

## Notes

- **MLflow** UI is accessible at [localhost:5001](http://localhost:5001).
- The project does not currently have a working implementation for **Airflow** (it was initially planned but is not yet functional).
- The **Inference API** is available at [localhost:5002](http://localhost:5002).
- Dataset is prepared beforehand, but 