#!/bin/bash

# Medical AI Platform - Start Script
echo "🏥 Starting Medical AI Platform..."
echo "=================================="

# Check if local medical_env exists
# if [ ! -d "./medical_env" ]; then
#     echo "❌ Local 'medical_env' not found. Please run setup first."
#     exit 1
# fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    exit 1
fi

echo "✓ Python Environment found (tf_gpu_env)"
echo "✓ Node found: $(node --version)"
echo ""

# Start Flask API in background with local environment
echo "🚀 Starting Flask API Server (port 5000)..."
# Using nohup to prevent it from closing if shell closes, but trapping SIGINT to kill it
bash -c "source ~/tf_gpu_env/bin/activate && python api_server.py" &
API_PID=$!
echo "   API PID: $API_PID"

# Wait for API to start
sleep 3

# Start Next.js frontend
echo "🚀 Starting Next.js Frontend (port 3000)..."
cd landeros-clone
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "=================================="
echo "✅ Both servers started!"
echo "=================================="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:5000"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping servers...'; kill $API_PID $FRONTEND_PID; exit" INT
wait
