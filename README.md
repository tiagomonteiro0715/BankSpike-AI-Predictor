# BankSpike AI Predictor


## Table of Contents

- [Overview](#overview)
- [Video Demo](#video-demo)
- [Background](#background)
- [Features](#features)
- [Why Spiking Neural Networks?](#why-spiking-neural-networks)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Create Virtual Environment](#create-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Configure Jupyter Kernel](#configure-jupyter-kernel)
- [Usage](#usage)
  - [Streamlit Web Application](#streamlit-web-application)
  - [FastAPI Service](#fastapi-service)
  - [Docker](#docker)
  - [API Endpoints](#api-endpoints)
- [Example API Requests](#example-api-requests)
  - [Example 1: Married Admin Professional](#example-1-married-admin-professional)
  - [Example 2: Single Technician](#example-2-single-technician)
  - [Example 3: Divorced Blue-Collar Worker](#example-3-divorced-blue-collar-worker)
  - [Example 4: Retired Client](#example-4-retired-client)
  - [Example 5: Student](#example-5-student)
  - [Example 6: Entrepreneur](#example-6-entrepreneur)
- [Input Features](#input-features)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Citation](#citation)

## Overview

BankSpike AI Predictor is a deep learning system that predicts with 88% confidence whether a client will choose a term deposit. The system uses spiking neural networks (SNNs), a bio-inspired approach to neural computation that offers unique advantages for temporal pattern recognition.

## Video Demo

https://github.com/user-attachments/assets/dcf692c5-cf4d-472d-9b22-150868f812d7


## Background

This project is based on research from the article "A data-driven approach to predict the success of bank telemarketing" published in Decision Support Systems.

Reference: https://doi.org/10.1016/j.dss.2014.03.001

## Features

* Spiking neural network implementation for binary classification
* Hyperparameter optimization using Optuna
* Interactive web interface built with Streamlit
* RESTful API service using FastAPI
* Comprehensive preprocessing pipeline with saved parameters
* Model persistence and versioning support


## Why Spiking Neural Networks?


Unlike traditional ANNs, SNNs communicate through discrete spike trains that encode temporal patterns in the data. For banking predictions involving time-dependent features (contact duration, days since last contact, campaign frequency), SNNs leverage spike timing. This way, SSNs capture these relationships while consuming 20-50x less energy per inference than equivalent deep ANNs. 


## Project Structure

```
BankSpike/
├── bank_spike_predictor.py          # Streamlit web application
├── bank_spike_api.py                 # FastAPI REST API service
├── test_bank_spike_api.py            # API test suite
├── hyperparameter_optimization.ipynb # Jupyter notebook for model tuning
├── best_snn_model_optuna.pth         # Trained model weights
├── preprocessing_params_*.json       # Saved preprocessing parameters
├── optuna_results_*.json             # Hyperparameter optimization results
├── requirements.txt                  # Python dependencies
├── archive/                          # Historical files
├── logs/                             # Application logs
└── versions/                         # Model versions
```

## Installation

### Create Virtual Environment

**Windows:**

```bash
python -m venv env
cd env/Scripts
Activate
```

**Mac/Linux:**

```bash
python -m venv env
source env/bin/activate
```

### Install Dependencies

```bash
pip install -U "torch>=2.1" "snntorch>=0.9.1" "scikit-learn>=1.3" "ucimlrepo>=0.0.7" "notebook>=7.0" "ipykernel>=6.25" "optuna>=3.5" "streamlit>=1.31" "plotly>=5.18" "fastapi[standard]>=0.110" "uvicorn[standard]>=0.30" "pydantic>=2.6" "numpy>=1.24" "pandas>=2.1" --no-cache-dir --verbose
```

### Configure Jupyter Kernel

```bash
python -m ipykernel install --user --name=env --display-name "Spiking NN env"
```

## Usage

### Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run bank_spike_predictor.py
```

### FastAPI Service

Start the REST API server:

```bash
uvicorn bank_spike_api:app --reload
```

### Docker

Build and run the suite in Docker:

```bash
docker pull tiagomonteiro0715/bank-spike-suite

docker run -p 8000:8000 -p 8501:8501 tiagomonteiro0715/bank-spike-suite
```

### API Endpoints

**Health Check:**

```bash
curl http://127.0.0.1:8000/health
```

**Prediction Endpoint:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{...}"
```

## Example API Requests

### Example 1: Married Admin Professional

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":35,\"job\":\"admin.\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":1200,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day_of_week\":15,\"month\":\"may\",\"duration\":180,\"campaign\":2,\"pdays\":999,\"previous\":0,\"poutcome\":\"unknown\"}"
```

### Example 2: Single Technician

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":28,\"job\":\"technician\",\"marital\":\"single\",\"education\":\"tertiary\",\"default\":\"no\",\"balance\":800,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"telephone\",\"day_of_week\":10,\"month\":\"jan\",\"duration\":200,\"campaign\":1,\"pdays\":999,\"previous\":0,\"poutcome\":\"success\"}"
```

### Example 3: Divorced Blue-Collar Worker

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":45,\"job\":\"blue-collar\",\"marital\":\"divorced\",\"education\":\"primary\",\"default\":\"yes\",\"balance\":1500,\"housing\":\"no\",\"loan\":\"yes\",\"contact\":\"cellular\",\"day_of_week\":20,\"month\":\"sep\",\"duration\":90,\"campaign\":3,\"pdays\":20,\"previous\":1,\"poutcome\":\"failure\"}"
```

### Example 4: Retired Client

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":67,\"job\":\"retired\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":3000,\"housing\":\"no\",\"loan\":\"no\",\"contact\":\"telephone\",\"day_of_week\":5,\"month\":\"dec\",\"duration\":300,\"campaign\":1,\"pdays\":999,\"previous\":0,\"poutcome\":\"other\"}"
```

### Example 5: Student

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":22,\"job\":\"student\",\"marital\":\"single\",\"education\":\"tertiary\",\"default\":\"no\",\"balance\":200,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day_of_week\":2,\"month\":\"mar\",\"duration\":60,\"campaign\":1,\"pdays\":999,\"previous\":0,\"poutcome\":\"success\"}"
```

### Example 6: Entrepreneur

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"age\":38,\"job\":\"entrepreneur\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":5000,\"housing\":\"yes\",\"loan\":\"yes\",\"contact\":\"telephone\",\"day_of_week\":7,\"month\":\"nov\",\"duration\":400,\"campaign\":4,\"pdays\":5,\"previous\":3,\"poutcome\":\"failure\"}"
```

## Input Features

The model accepts the following client attributes:

* age: Client age in years
* job: Type of job (admin., technician, blue-collar, etc.)
* marital: Marital status (married, single, divorced)
* education: Education level (primary, secondary, tertiary)
* default: Has credit in default? (yes, no)
* balance: Account balance in euros
* housing: Has housing loan? (yes, no)
* loan: Has personal loan? (yes, no)
* contact: Contact communication type (cellular, telephone)
* day_of_week: Last contact day of the week
* month: Last contact month of year
* duration: Last contact duration in seconds
* campaign: Number of contacts performed during this campaign
* pdays: Number of days since client was last contacted (999 means not previously contacted)
* previous: Number of contacts performed before this campaign
* poutcome: Outcome of previous marketing campaign (success, failure, unknown, other)

## Model Performance

The trained spiking neural network achieves 88% confidence in predicting term deposit subscription outcomes.

## Technologies Used

* PyTorch: Deep learning framework
* snnTorch: Spiking neural network library
* Streamlit: Web application framework
* FastAPI: Modern API framework
* Optuna: Hyperparameter optimization
* scikit-learn: Machine learning utilities
* Pandas & NumPy: Data manipulation
* Plotly: Interactive visualizations

## License

See LICENSE file for details.
