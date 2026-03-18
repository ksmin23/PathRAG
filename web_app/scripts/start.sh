#!/bin/bash
# Start script for PathRAG application (both API and UI)

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PathRAG Application...${NC}"

# Navigate to project root (two levels up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Python virtual environment not found. Creating one...${NC}"
    python -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating Python virtual environment...${NC}"
source .venv/bin/activate

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}Backend dependencies installed.${NC}"

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd web_app/frontend
npm install
echo -e "${GREEN}Frontend dependencies installed.${NC}"
cd "$PROJECT_ROOT"

# Start backend in background
echo -e "${BLUE}Starting backend API on port 8000...${NC}"
cd web_app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
sleep 5

# Start frontend
echo -e "${BLUE}Starting frontend UI on port 3000...${NC}"
cd web_app/frontend
PORT=3000 REACT_APP_API_URL=http://localhost:8000 npm start &
FRONTEND_PID=$!

echo -e "${GREEN}Both services are running.${NC}"
echo -e "  Backend API: http://localhost:8000"
echo -e "  Frontend UI: http://localhost:3000"
echo -e "${YELLOW}Press Ctrl+C to stop both services.${NC}"

# Trap Ctrl+C to stop both services
trap "echo -e '\n${GREEN}Stopping services...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Wait for either process to exit
wait
