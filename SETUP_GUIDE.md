# 🚀 Fortune AI Local Development Setup Guide

This guide will help you set up the complete local development environment for the Fortune AI project, including both the FastAPI backend and Next.js frontend.

## 📋 Prerequisites

### System Requirements
- **Operating System**: macOS, Windows, or Linux
- **Python**: 3.11.x (required)
- **Node.js**: 18.x or higher
- **Git**: Latest version
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: At least 5GB free space

### Required Accounts & API Keys
- **OpenAI API Key**: [Get from OpenAI](https://platform.openai.com/api-keys)
- **Google Gemini API Key**: [Get from Google AI Studio](https://makersuite.google.com/app/apikey)
- **Tavily API Key** (optional): [Get from Tavily](https://tavily.com/)

## 🛠️ Installation Steps

### Step 1: Clone the Repository
```bash
# Clone the repository
git clone <your-repository-url>
cd Fortune

# Verify the structure
ls -la
```

Expected structure:
```
Fortune/
├── front/                 # Next.js frontend
├── fastapi/              # FastAPI backend
├── pyproject.toml        # Python dependencies
├── requirements.txt      # Alternative Python dependencies
├── main.py              # Main Python entry point
└── README.md
```

### Step 2: Backend Setup (Python/FastAPI)

#### 2.1 Install Python Dependencies

**Option A: Using Poetry (Recommended)**
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

**Option B: Using pip**
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2.2 Environment Configuration
```bash
# Create environment file
cp .env.example .env  # if exists, or create manually
```

Create `.env` file in the root directory:
```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional API Keys
TAVILY_API_KEY=your_tavily_api_key_here

# Development Settings
DEBUG_MODE=true
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

#### 2.3 Verify Backend Installation
```bash
# Test Python imports
python -c "
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
print('✅ Backend dependencies installed successfully')
"

# Test FastAPI server
cd fastapi
python main.py
```

### Step 3: Frontend Setup (Next.js)

#### 3.1 Install Node.js Dependencies
```bash
# Navigate to frontend directory
cd front

# Install dependencies
npm install

# Verify installation
npm run build
```

#### 3.2 Frontend Environment Configuration
Create `.env.local` file in the `front/` directory:
```env
# Development server
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Optional: Analytics or other services
NEXT_PUBLIC_APP_NAME=Fortune AI
```

#### 3.3 Verify Frontend Installation
```bash
# Test development server
npm run dev

# Should start on http://localhost:3000
```

## 🚀 Running the Application

### Option 1: Separate Terminals (Recommended for Development)

#### Terminal 1: Backend Server
```bash
# Activate Python environment
poetry shell  # or source venv/bin/activate

# Start FastAPI server
cd fastapi
python main.py

# Server should start on http://localhost:8000
```

#### Terminal 2: Frontend Server
```bash
# Navigate to frontend
cd front

# Start Next.js development server
npm run dev

# Frontend should start on http://localhost:3000
```

### Option 2: Single Command (Production-like)
```bash
# Create a startup script
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Fortune AI Development Environment..."

# Start backend in background
echo "📡 Starting FastAPI backend..."
cd fastapi && python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🖥️ Starting Next.js frontend..."
cd ../front && npm run dev &
FRONTEND_PID=$!

echo "✅ Both servers started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop both servers"

# Wait for interrupt
wait
EOF

# Make executable and run
chmod +x start_dev.sh
./start_dev.sh
```

## 🧪 Testing the Setup

### 1. Backend Health Check
```bash
# Test FastAPI endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "system_loaded": {
    "compiled_graph": true,
    "memory": true,
    "rag_system": true,
    "tarot_compiled_graph": true
  }
}
```

### 2. Frontend Health Check
- Open browser to `http://localhost:3000`
- Should see Fortune AI homepage
- Navigate to `/saju` or `/tarot` pages

### 3. WebSocket Connection Test
```bash
# Test WebSocket connection (using wscat if available)
npm install -g wscat
wscat -c ws://localhost:8000/ws/chat/saju/test-session
```

### 4. Full Integration Test
1. Open `http://localhost:3000/saju`
2. Enter a test message: "1996년 12월 13일 남자 운세봐줘"
3. Should receive AI response via WebSocket

## 🔧 Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version  # Should be 3.11.x

# If wrong version, install Python 3.11
# macOS:
brew install python@3.11
# Ubuntu:
sudo apt install python3.11
```

#### 2. Poetry Installation Issues
```bash
# Reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clear Poetry cache
poetry cache clear --all pypi
```

#### 3. Node.js Version Issues
```bash
# Check Node.js version
node --version  # Should be 18.x or higher

# Install/update Node.js
# Using nvm (recommended):
nvm install 18
nvm use 18
```

#### 4. Port Conflicts
```bash
# Check if ports are in use
lsof -i :8000  # Backend port
lsof -i :3000  # Frontend port

# Kill processes if needed
kill -9 <PID>
```

#### 5. API Key Issues
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Test API connectivity
python -c "
import openai
openai.api_key = 'your_key'
print('OpenAI connection test')
"
```

#### 6. Memory Issues
```bash
# Check available memory
free -h  # Linux
vm_stat   # macOS

# Increase swap if needed
# Linux:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debug Mode

#### Backend Debug
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
export DEBUG_MODE=true

# Run with debug output
python main.py --debug
```

#### Frontend Debug
```bash
# Enable Next.js debug
DEBUG=* npm run dev

# Check browser console for errors
# Open Developer Tools (F12)
```

## 📁 Project Structure Overview

```
Fortune/
├── front/                          # Next.js Frontend
│   ├── src/
│   │   ├── app/                   # App Router pages
│   │   ├── components/            # React components
│   │   ├── store/                 # Zustand state management
│   │   └── lib/                   # Utility functions
│   ├── public/                    # Static assets
│   └── package.json
├── fastapi/                       # FastAPI Backend
│   └── main.py                   # Main server file
├── parser/                        # Data processing
│   ├── tarot_agent/              # Tarot card system
│   └── embedding.py              # Vector embeddings
├── faiss_saju/                   # Vector database
├── main.py                       # CLI entry point
├── graph.py                      # LangGraph workflow
├── agents.py                     # AI agents
├── nodes.py                      # Workflow nodes
├── state.py                      # State management
├── tools.py                      # Agent tools
├── pyproject.toml               # Python dependencies
└── requirements.txt             # Alternative dependencies
```

## 🔄 Development Workflow

### 1. Making Changes
```bash
# Backend changes
# Edit Python files in root or fastapi/
# Restart backend server

# Frontend changes
# Edit files in front/src/
# Changes auto-reload in browser
```

### 2. Adding Dependencies
```bash
# Python dependencies
poetry add package-name
# or
pip install package-name && pip freeze > requirements.txt

# Node.js dependencies
cd front
npm install package-name
```

### 3. Testing Changes
```bash
# Backend testing
poetry run python -m pytest

# Frontend testing
cd front
npm test
```

## 🚀 Production Deployment

### Backend Deployment
```bash
# Build production image
docker build -t fortune-backend .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  fortune-backend
```

### Frontend Deployment
```bash
# Build for production
cd front
npm run build

# Start production server
npm start
```

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in both terminal windows
3. Verify all API keys are correctly set
4. Ensure all dependencies are installed
5. Check system requirements are met

For additional support, please refer to the project's README.md or create an issue in the repository.

---

**Happy coding! 🎉**

Your Fortune AI development environment should now be ready for local testing and development. 