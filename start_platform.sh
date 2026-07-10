#!/bin/bash

echo "Starting LangSplat Platform..."

# Kill existing processes on ports if they exist
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start Backend
echo "Starting FastAPI Backend on port 8000..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Vite Frontend on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Platform is running!"
echo "- Frontend Dashboard: http://localhost:5173"
echo "- Backend API Docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop both servers."

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
