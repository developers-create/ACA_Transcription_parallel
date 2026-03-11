#!/bin/bash
echo "Starting Call Transcription System..."
echo "================================"
echo "Starting scheduler (background process)..."
python scheduler.py &
SCHEDULER_PID=$!
echo "Scheduler started with PID: $SCHEDULER_PID"
echo "Starting FastAPI API server..."
echo "API will be available at: http://0.0.0.0:$PORT"
uvicorn manual_transcriptor:app --host 0.0.0.0 --port $PORT
