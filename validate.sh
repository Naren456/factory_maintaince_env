#!/bin/bash
set -e

echo "🔍 Starting Factory Maintenance Environment Validation..."

# 1. Check for required files
echo "📁 Checking file structure..."
for f in client.py models.py inference.py pyproject.toml openenv.yaml server/app.py factory_env/__init__.py; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing required file: $f"
        exit 1
    fi
done
echo "✅ File structure look good."

# 2. Run OpenEnv validation
echo "🤖 Running openenv validate..."
if ! openenv validate; then
    echo "❌ OpenEnv validation failed."
    exit 1
fi
echo "✅ OpenEnv validation passed."

# 3. Check imports
echo "🐍 Verifying Python imports..."
export PYTHONPATH=.
python3 -c "from client import FactoryEnv; from models import FactoryAction; from factory_env.environment import FactoryEnvironment"
echo "✅ Python imports verified."

# 4. Check server start/stop
echo "🚀 Testing server startup..."
python3 -m server.app > server_test.log 2>&1 &
SERVER_PID=$!
sleep 5
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ Server started successfully and is healthy."
else
    echo "❌ Server health check failed. See server_test.log"
    kill $SERVER_PID
    exit 1
fi
kill $SERVER_PID
echo "✅ Server shutdown test passed."

echo "🏁 Validation complete! Your environment is ready for submission."
