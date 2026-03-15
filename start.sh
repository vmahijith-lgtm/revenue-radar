#!/bin/bash
# start.sh — starts the full Attribution Engine stack
# Usage: ./start.sh [port]
#
# Starts:
#   1. FastAPI budget allocation API  on port 8000 (background)
#   2. Streamlit attribution dashboard on port 8501 (foreground)

set -e

PORT=${PORT:-8501}
API_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "🚀 Starting Attribution Engine..."

# Kill any existing processes on these ports
lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$PORT     | xargs kill -9 2>/dev/null || true

# Start FastAPI in background
echo "   ▶ FastAPI API on port $API_PORT"
uvicorn backend.main:app --host 0.0.0.0 --port $API_PORT --workers 1 &
API_PID=$!

# Wait for API to be ready
sleep 2

# Start Streamlit (foreground)
echo "   ▶ Streamlit dashboard on port $PORT"
echo ""
streamlit run dashboard.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true

# Cleanup on exit
trap "kill $API_PID 2>/dev/null; exit" INT TERM
