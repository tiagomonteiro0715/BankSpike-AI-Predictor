#!/bin/sh
set -e

# Start FastAPI (Uvicorn) in background
uvicorn bank_spike_api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Streamlit in foreground
exec streamlit run bank_spike_predictor.py --server.port 8501 --server.address 0.0.0.0

# Cleanup (not usually reached because of exec, but kept for safety)
kill $API_PID 2>/dev/null || true
wait $API_PID 2>/dev/null || true
