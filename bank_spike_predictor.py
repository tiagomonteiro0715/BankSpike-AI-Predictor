import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import snntorch as snn
import streamlit as st
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')

# Constants
LOG_DIR = Path("logs")
RECENT_LOG_FILE = LOG_DIR / "predictions_recent.json"
MAX_RECENT_LOGS = 1000

# Page configuration
st.set_page_config(
    page_title="Spiking Neural Network - Bank Marketing Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

# ===== Logging =====
def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _monthly_archive_path(ts: datetime) -> Path:
    return LOG_DIR / f"predictions_{ts.year:04d}_{ts.month:02d}.jsonl"

def load_recent_logs() -> List[Dict[str, Any]]:
    if RECENT_LOG_FILE.exists():
        try:
            with open(RECENT_LOG_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def append_prediction_log(entry: Dict[str, Any]):
    _ensure_log_dir()
    try:
        archive_path = _monthly_archive_path(datetime.fromisoformat(entry.get("timestamp")))
        with open(archive_path, "a") as fa:
            fa.write(json.dumps(entry) + "\n")
        recent = load_recent_logs()
        recent.append(entry)
        if len(recent) > MAX_RECENT_LOGS:
            recent = recent[-MAX_RECENT_LOGS:]
        with open(RECENT_LOG_FILE, "w") as fr:
            json.dump(recent, fr, indent=2)
    except:
        pass

# ===== Data Loading =====
@st.cache_data
def load_data_and_preprocessors():
    bank_marketing = fetch_ucirepo(id=222)
    X, y = bank_marketing.data.features, bank_marketing.data.targets
    
    try:
        with open('preprocessing_params_20251015_111305.json', 'r') as f:
            prep = json.load(f)
        cat_cols = prep.get("column_transformer_info", {}).get("categorical_columns", [])
        num_cols = prep.get("column_transformer_info", {}).get("numerical_columns", [])
        ohe_cats = prep.get("one_hot_encoder", {}).get("categories")
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, 
                           categories=ohe_cats if ohe_cats else 'auto')
    except:
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64']).columns.tolist()
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    ct = ColumnTransformer([
        ("one-hot", ohe, cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ], remainder='passthrough')
    
    X_transformed = ct.fit_transform(X)
    le = LabelEncoder()
    y_transformed = le.fit_transform(y.values.ravel())
    
    return X, y, ct, le, cat_cols, num_cols

@st.cache_data
def load_model_info():
    default_info = {
        "model_architecture": {"num_inputs": 51, "num_hidden": 100, "num_outputs": 2, 
                              "hidden_sizes": [100], "total_parameters": 5402},
        "hyperparameters": {"beta": 0.4, "threshold": 1.4, "num_steps": 25, "batch_size": 32, 
                          "learning_rate": 0.001, "num_epochs": 5, "optimizer": "Adam", 
                          "loss_function": "CrossEntropyLoss"},
        "best_results": {"best_dev_accuracy": 88.58, "best_epoch": 4, 
                       "final_test_accuracy": 88.79, "final_test_loss": 0.2201}
    }
    
    try:
        with open('optuna_results_20251015_113406.json', 'r') as f:
            opt = json.load(f)
        
        arch = opt.get("model_architecture", {})
        hidden_sizes = arch.get("hidden_sizes", [arch.get("num_hidden", 100)])
        hp = opt.get("best_hyperparameters", {})
        final_res = opt.get("final_results", {})
        
        return {
            "model_architecture": {
                "num_inputs": arch.get("num_inputs", 51),
                "num_hidden": hidden_sizes[0],
                "hidden_sizes": hidden_sizes,
                "num_outputs": arch.get("num_outputs", 2),
                "total_parameters": arch.get("total_parameters", 0)
            },
            "hyperparameters": {
                "beta": hp.get("beta", 0.4),
                "threshold": hp.get("threshold", 1.4),
                "num_steps": hp.get("num_steps", 25),
                "batch_size": hp.get("batch_size", 32),
                "learning_rate": hp.get("learning_rate", 0.001),
                "adam_beta1": hp.get("adam_beta1"),
                "adam_beta2": hp.get("adam_beta2"),
                "lr_patience": hp.get("lr_patience"),
                "lr_factor": hp.get("lr_factor"),
                "num_epochs": final_res.get("best_epoch", 0),
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss"
            },
            "best_results": {
                "best_dev_accuracy": final_res.get("best_dev_accuracy", 88.58),
                "best_epoch": final_res.get("best_epoch", 4),
                "final_test_accuracy": final_res.get("test_accuracy", final_res.get("final_test_accuracy", 88.79)),
                "final_test_loss": final_res.get("test_loss", final_res.get("final_test_loss", 0.2201))
            },
            "optimization_info": opt.get("optimization_info", {}),
            "data_info": opt.get("data_info", {})
        }
    except Exception as e:
        return default_info

# ===== Model =====
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_sizes, num_outputs, beta, threshold):
        super().__init__()
        hidden_sizes = hidden_sizes or []
        layer_sizes = [num_inputs] + hidden_sizes
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) 
                                     for i in range(len(hidden_sizes))])
        self.lif_layers = nn.ModuleList([snn.Leaky(beta=beta, threshold=threshold) 
                                        for _ in range(len(hidden_sizes))])
        self.fc_out = nn.Linear(layer_sizes[-1], num_outputs)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold)

    def init_state(self, batch_size):
        mems = [torch.zeros(batch_size, lin.out_features) for lin in self.layers]
        spks = [torch.zeros(batch_size, lin.out_features) for lin in self.layers]
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

@st.cache_resource
def load_trained_model():
    model_info = load_model_info()
    model = SpikingNeuralNetwork(
        model_info["model_architecture"]["num_inputs"],
        model_info["model_architecture"].get("hidden_sizes", [model_info["model_architecture"]["num_hidden"]]),
        model_info["model_architecture"]["num_outputs"],
        model_info["hyperparameters"]["beta"],
        model_info["hyperparameters"]["threshold"]
    )
    try:
        try:
            model.load_state_dict(torch.load('best_snn_model_optuna.pth', map_location='cpu'))
        except:
            model.load_state_dict(torch.load('best_snn_model.pth', map_location='cpu'))
        model.eval()
        return model, model_info
    except:
        st.error("Model file not found!")
        return None, model_info

def predict_with_snn(model, input_data, num_steps=25):
    if model is None:
        return None, None, None
    model.eval()
    with torch.no_grad():
        mems, spks, mem_out = model.init_state(batch_size=input_data.shape[0])
        spk_out_rec, mem_out_rec = [], []
        for _ in range(num_steps):
            mems, spks, mem_out, spk_out = model(input_data, mems, spks, mem_out)
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)
        spk2_rec = torch.stack(spk_out_rec)
        mem2_rec = torch.stack(mem_out_rec)
        return spk2_rec.sum(dim=0), mem2_rec, torch.softmax(mem2_rec[-1], dim=1)

# ===== UI Components =====
def create_input_form(cat_cols, num_cols):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Personal Information**")
        age = st.slider("Age", 18, 95, 30)
        job = st.selectbox("Job", ["blue-collar", "entrepreneur", "housemaid", "management", 
                                   "retired", "self-employed", "services", "student", 
                                   "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
        st.write("**Financial Information**")
        balance = st.number_input("Account Balance", -8019, 102127, 0)
        housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
    
    with col2:
        st.write("**Contact Information**")
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        day_of_week = st.slider("Day of Month", 1, 31, 15)
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", 
                                      "jul", "aug", "sep", "oct", "nov", "dec"])
        st.write("**Campaign Information**")
        duration = st.number_input("Last Contact Duration (seconds)", 0, 4918, 200)
        campaign = st.slider("Contacts (this campaign)", 1, 63, 1)
        pdays = st.number_input("Days Since Last Contact", 0, 871, 0)
        previous = st.slider("Previous Contacts", 0, 275, 0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "other", "success", "unknown"])
    
    return {'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
            'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact,
            'day_of_week': day_of_week, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome}

def render_sidebar(model_info):
    st.sidebar.title("Model Information")
    st.sidebar.metric("Total Parameters", f"{model_info['model_architecture']['total_parameters']:,}")
    st.sidebar.subheader("Hyperparameters")
    hp = model_info['hyperparameters']
    st.sidebar.metric("Learning Rate", f"{hp['learning_rate']:.6g}")
    if hp.get("adam_beta1"):
        st.sidebar.metric("Adam Beta1", f"{hp['adam_beta1']:.3f}")
    if hp.get("adam_beta2"):
        st.sidebar.metric("Adam Beta2", f"{hp['adam_beta2']:.3f}")
    st.sidebar.metric("Batch Size", hp["batch_size"])
    st.sidebar.metric("Hidden Neurons", model_info["model_architecture"]["num_hidden"])
    st.sidebar.subheader("SNN Hyperparameters")
    st.sidebar.metric("Beta", f"{hp['beta']:.1f}")
    st.sidebar.metric("Threshold", f"{hp['threshold']:.1f}")
    st.sidebar.metric("Time Steps", hp["num_steps"])
    st.sidebar.subheader("Performance")
    st.sidebar.metric("Test Accuracy", f"{model_info['best_results']['final_test_accuracy']:.2f}%")
    st.sidebar.metric("Dev Accuracy", f"{model_info['best_results']['best_dev_accuracy']:.2f}%")

# ===== Main App =====
def main():
    st.markdown('<h1 class="main-header">BankSpike AI Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Predict term deposit subscriptions with 88% accuracy using spiking neural networks</h2>', unsafe_allow_html=True)
    
    with st.expander("Legal Disclaimer"):
        st.markdown("This tool is for educational purposes only. Not financial advice. Use at your own risk.")
    
    with st.spinner("Loading model and data..."):
        X, y, ct, le, cat_cols, num_cols = load_data_and_preprocessors()
        model, model_info = load_trained_model()
    
    if model is None:
        return
    
    render_sidebar(model_info)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Monitoring", "Model Analysis", "Dataset Overview"])
    
    with tab1:
        st.header("Make a Prediction")
        input_data = create_input_form(cat_cols, num_cols)
        
        if st.button("Predict Subscription Probability", type="primary"):
            with st.spinner("Processing..."):
                input_df = pd.DataFrame([input_data])
                input_transformed = ct.transform(input_df)
                input_tensor = torch.FloatTensor(input_transformed)
                
                spike_count, mem_rec, probs = predict_with_snn(model, input_tensor, 
                                                               model_info["hyperparameters"].get("num_steps", 25))
                
                if spike_count is not None:
                    pred_class = spike_count.argmax().item()
                    conf = probs[0][pred_class].item()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", "Will Subscribe" if pred_class == 1 else "Will Not Subscribe")
                    col2.metric("Confidence", f"{conf:.2%}")
                    
                    fig = go.Figure(data=[go.Bar(
                        x=['Will Not Subscribe', 'Will Subscribe'],
                        y=[probs[0][0].item(), probs[0][1].item()],
                        marker_color=['#ff6b6b', '#4ecdc4']
                    )])
                    fig.update_layout(title="Subscription Probability", yaxis_title="Probability", 
                                    yaxis=dict(range=[0, 1]), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    try:
                        append_prediction_log({
                            "timestamp": datetime.utcnow().isoformat(),
                            "inputs": input_data,
                            "prediction": {"class": int(pred_class), 
                                         "label": "Will Subscribe" if pred_class == 1 else "Will Not Subscribe"},
                            "confidence": float(conf),
                            "probability_no": float(probs[0][0].item()),
                            "probability_yes": float(probs[0][1].item())
                        })
                    except:
                        pass
    
    with tab2:
        st.header("Monitoring")
        show_n = st.slider("Show last N predictions", 10, MAX_RECENT_LOGS, min(200, MAX_RECENT_LOGS))
        logs = load_recent_logs()
        
        if not logs:
            st.info("No predictions logged yet.")
        else:
            df = pd.json_normalize(logs)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp")
            recent_df = df.tail(show_n)
            
            display_df = recent_df.copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Predictions", f"{len(df):,}")
            if "confidence" in recent_df:
                c2.metric("Avg Confidence", f"{recent_df['confidence'].mean():.2%}")
            if "prediction.label" in recent_df:
                c3.metric("Subscribe Rate", f"{(recent_df['prediction.label']=='Will Subscribe').mean():.2%}")
    
    with tab3:
        st.header("Model Analysis")
        st.subheader("Network Architecture")
        st.dataframe(pd.DataFrame({
            'Layer': ['Input', 'Hidden (LIF)', 'Output (LIF)'],
            'Neurons': [model_info["model_architecture"]["num_inputs"],
                       model_info["model_architecture"]["num_hidden"],
                       model_info["model_architecture"]["num_outputs"]]
        }))
        
        st.subheader("Performance Metrics")
        st.dataframe(pd.DataFrame({
            'Metric': ['Test Accuracy', 'Dev Accuracy', 'Test Loss', 'Best Epoch'],
            'Value': [f"{model_info['best_results']['final_test_accuracy']:.2f}%",
                     f"{model_info['best_results']['best_dev_accuracy']:.2f}%",
                     f"{model_info['best_results']['final_test_loss']:.4f}",
                     str(model_info['best_results']['best_epoch'])]
        }))
    
    with tab4:
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", f"{len(X):,}")
        col1.metric("Features", len(X.columns))
        col2.metric("Categorical", len(cat_cols))
        col2.metric("Numerical", len(num_cols))
        target_counts = y.value_counts()
        col3.metric("No Subscription", f"{target_counts.get('no', 0):,}")
        col3.metric("Subscription", f"{target_counts.get('yes', 0):,}")
        
        feature = st.selectbox("Select Feature to Visualize", X.columns)
        if pd.api.types.is_numeric_dtype(X[feature]):
            fig = px.histogram(X, x=feature, title=f"Distribution of {feature}")
        else:
            vc = X[feature].astype(str).value_counts()
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sample Data")
        st.dataframe(X.head(10))

if __name__ == "__main__":
    main()