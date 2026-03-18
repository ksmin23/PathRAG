#!/bin/bash
# Script to start only the UI on port 3000

# Navigate to project root (two levels up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting PathRAG UI on port 3000..."
cd web_app/frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
    echo "Frontend dependencies installed."
fi

# Set environment variables
export PORT=3000
export REACT_APP_API_URL=http://localhost:8000

# Start the UI
echo "Starting React application on port 3000..."
npx cross-env PORT=3000 npm start

echo "UI server stopped."
