#!/bin/bash
# OpenEnv Agent Validation Script 🚀
# This script runs the local environment server, executes the AI agent,
# and then grades the performance.

# 1. Start Server
echo "Starting Factory Environment Server..."
export PYTHONPATH=.
source .venv/bin/activate
uvicorn factory_env.server.app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for health check..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health | grep -q "ok"; then
        echo "Server is UP! ✅"
        break
    fi
    sleep 2
done

# 2. Run Agent Inference
echo "Running Agent Simulation (Task: Medium)..."
# We'll pipe the output to a log file for grading
python inference.py > agent_trajectory.log

# 3. Grade Results
echo "Grading Performance..."
# Note: In a real run, you'd parse the [END] line or the JSON logs
# Here we'll just check if the log contains the [END] line
if grep -q "\[END\]" agent_trajectory.log; then
    echo "Agent run completed successfully. Summary:"
    grep "\[END\]" agent_trajectory.log
else
    echo "Agent run FAILED or crashed. Check agent_trajectory.log"
fi

# 4. Cleanup
echo "Cleaning up server..."
kill $SERVER_PID
echo "Validation complete."
