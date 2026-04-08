# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment code
COPY . /app

# Install project and dependencies
RUN pip install --no-cache-dir .

# Set PYTHONPATH so internal imports work
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server directly
# Note: we use python -m uvicorn or just uvicorn since it's in the bin path after pip install
CMD ["python", "-m", "uvicorn", "factory_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
