import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
RECENT_LOG_FILE = LOG_DIR / "predictions_recent.json"
MAX_RECENT_LOGS = 1000

# Logging utilities
class PredictionLogger:
    @staticmethod
    def ensure_dir():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def monthly_path(ts: datetime) -> Path:
        return LOG_DIR / f"predictions_{ts.year:04d}_{ts.month:02d}.jsonl"
    
    @classmethod
    def load_recent(cls) -> List[Dict[str, Any]]:
        if not RECENT_LOG_FILE.exists():
            return []
        try:
            with open(RECENT_LOG_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    
    @classmethod
    def append(cls, entry: Dict[str, Any]):
        cls.ensure_dir()
        
        # Archive to monthly JSONL
        try:
            ts = datetime.fromisoformat(entry["timestamp"]) if isinstance(entry.get("timestamp"), str) else datetime.now(datetime.UTC)
            with open(cls.monthly_path(ts), "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
        
        # Update recent log
        try:
            recent = cls.load_recent()
            recent.append(entry)
            recent = recent[-MAX_RECENT_LOGS:]
            with open(RECENT_LOG_FILE, "w") as f:
                json.dump(recent, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# Spiking Neural Network
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, num_inputs: int, hidden_sizes: List[int], num_outputs: int, beta: float, threshold: float):
        super().__init__()
        layers = [num_inputs] + (hidden_sizes or [])
        
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(hidden_sizes))])
        self.lif_layers = nn.ModuleList([snn.Leaky(beta=beta, threshold=threshold) for _ in hidden_sizes])
        self.fc_out = nn.Linear(layers[-1], num_outputs)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold)
    
    def init_state(self, batch_size: int):
        mems = [torch.zeros(batch_size, layer.out_features) for layer in self.layers]
        spks = [torch.zeros(batch_size, layer.out_features) for layer in self.layers]
        mem_out = torch.zeros(batch_size, self.fc_out.out_features)
        return mems, spks, mem_out
    
    def forward(self, x, mems, spks, mem_out):
        cur = x
        for i, (lin, lif) in enumerate(zip(self.layers, self.lif_layers)):
            cur = lin(cur)
            spks[i], mems[i] = lif(cur, mems[i])
            cur = spks[i]
        
        cur = self.fc_out(cur)
        spk_out, mem_out = self.lif_out(cur, mem_out)
        return mems, spks, mem_out, spk_out

# Pydantic Models
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day_of_week: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if not 18 <= v <= 95:
            raise ValueError('Age must be between 18 and 95')
        return v
    
    @field_validator('balance')
    @classmethod
    def validate_balance(cls, v):
        if not -10000 <= v <= 200000:
            raise ValueError('Balance must be between -10000 and 200000')
        return v
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v):
        if not 0 <= v <= 5000:
            raise ValueError('Duration must be between 0 and 5000')
        return v
    
    @field_validator('campaign')
    @classmethod
    def validate_campaign(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('Campaign must be between 1 and 100')
        return v
    
    @field_validator('day_of_week')
    @classmethod
    def validate_day(cls, v):
        if not 1 <= v <= 31:
            raise ValueError('Day of week must be between 1 and 31')
        return v

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probability_no: float
    probability_yes: float
    spike_count: int
    processing_time: float

# Model Manager
class ModelManager:
    def __init__(self):
        self.model: Optional[SpikingNeuralNetwork] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model_info: Optional[Dict[str, Any]] = None
    
    def load(self):
        # Load model configuration
        with open('optuna_results_20251015_113406.json', 'r') as f:
            optuna_info = json.load(f)
        
        hidden_sizes = optuna_info.get("model_architecture", {}).get(
            "hidden_sizes", 
            [optuna_info.get("model_architecture", {}).get("num_hidden", 100)]
        )
        
        self.model_info = {
            "model_architecture": {
                "num_inputs": optuna_info["model_architecture"]["num_inputs"],
                "num_hidden": hidden_sizes[0],
                "num_outputs": optuna_info["model_architecture"]["num_outputs"],
                "total_parameters": optuna_info["model_architecture"].get("total_parameters", 0)
            },
            "hyperparameters": {
                "beta": optuna_info["best_hyperparameters"]["beta"],
                "threshold": optuna_info["best_hyperparameters"]["threshold"],
                "num_steps": optuna_info["best_hyperparameters"].get("num_steps", 25),
                "batch_size": optuna_info["best_hyperparameters"].get("batch_size", 32),
                "learning_rate": optuna_info["best_hyperparameters"]["learning_rate"],
                "num_epochs": optuna_info.get("final_results", {}).get("best_epoch", 0),
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss"
            },
            "best_results": {
                "best_dev_accuracy": optuna_info["final_results"]["best_dev_accuracy"],
                "best_epoch": optuna_info["final_results"]["best_epoch"],
                "final_test_accuracy": optuna_info["final_results"]["test_accuracy"],
                "final_test_loss": optuna_info["final_results"]["test_loss"]
            }
        }
        
        # Initialize model
        self.model = SpikingNeuralNetwork(
            num_inputs=self.model_info["model_architecture"]["num_inputs"],
            hidden_sizes=[self.model_info["model_architecture"]["num_hidden"]],
            num_outputs=self.model_info["model_architecture"]["num_outputs"],
            beta=self.model_info["hyperparameters"]["beta"],
            threshold=self.model_info["hyperparameters"]["threshold"]
        )
        
        # Load weights
        weight_file = 'best_snn_model_optuna.pth' if Path('best_snn_model_optuna.pth').exists() else 'best_snn_model.pth'
        self.model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        self.model.eval()
        
        # Setup preprocessors
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64']).columns.tolist()
        
        self.column_transformer = ColumnTransformer([
            ("one-hot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ("scaler", StandardScaler(), numerical_cols)
        ], remainder='passthrough')
        
        self.column_transformer.fit(X)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y.values.ravel())
        
        logger.info("Model and preprocessors loaded successfully")
    
    def predict(self, customer_data: CustomerData, num_steps: Optional[int] = None) -> Dict[str, Any]:
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        num_steps = num_steps or self.model_info["hyperparameters"]["num_steps"]
        
        # Prepare input
        input_df = pd.DataFrame([customer_data.model_dump()])
        input_transformed = self.column_transformer.transform(input_df)
        input_tensor = torch.FloatTensor(input_transformed)
        
        # Run SNN forward pass
        mems, spks, mem_out = self.model.init_state(batch_size=input_tensor.shape[0])
        spk_rec, mem_rec = [], []
        
        with torch.no_grad():
            for _ in range(num_steps):
                mems, spks, mem_out, spk_out = self.model(input_tensor, mems, spks, mem_out)
                spk_rec.append(spk_out)
                mem_rec.append(mem_out)
        
        # Process results
        spike_count = torch.stack(spk_rec).sum(dim=0)
        probabilities = torch.softmax(torch.stack(mem_rec)[-1], dim=1)
        
        predicted_class = spike_count.argmax().item()
        
        return {
            "prediction": "Will Subscribe" if predicted_class == 1 else "Will Not Subscribe",
            "confidence": probabilities[0][predicted_class].item(),
            "probability_no": probabilities[0][0].item(),
            "probability_yes": probabilities[0][1].item(),
            "spike_count": int(spike_count.sum().item())
        }

# Initialize FastAPI and ModelManager
app = FastAPI(
    title="BankSpike AI Predictor API",
    description="Spiking Neural Network for Bank Marketing Prediction",
    version="1.0.0"
)

model_manager = ModelManager()

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})

@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "error": str(exc)})

# Endpoints
@app.on_event("startup")
async def startup_event():
    logger.info("Starting BankSpike AI Predictor API...")
    model_manager.load()
    logger.info("API startup complete")

@app.get("/")
def root():
    return {
        "message": "BankSpike AI Predictor API",
        "description": "Spiking Neural Network for Bank Marketing Prediction",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_manager.model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.model is not None,
        "model_accuracy": model_manager.model_info["best_results"]["final_test_accuracy"] if model_manager.model_info else None
    }

@app.post("/predict", response_model=PredictionResult)
def predict(customer_data: CustomerData):
    start_time = datetime.now()
    
    try:
        prediction_result = model_manager.predict(customer_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = PredictionResult(
            **prediction_result,
            processing_time=processing_time
        )
        
        # Log prediction
        try:
            pred_class = 1 if prediction_result["prediction"] == "Will Subscribe" else 0
            log_entry = {
                "timestamp": datetime.now(datetime.UTC).isoformat(),
                "inputs": customer_data.model_dump(),
                "prediction": {"class": pred_class, "label": prediction_result["prediction"]},
                "confidence": float(prediction_result["confidence"]),
                "probability_no": float(prediction_result["probability_no"]),
                "probability_yes": float(prediction_result["probability_yes"]),
                "spike_count": int(prediction_result["spike_count"]),
                "processing_time": float(processing_time)
            }
            PredictionLogger.append(log_entry)
        except Exception:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)