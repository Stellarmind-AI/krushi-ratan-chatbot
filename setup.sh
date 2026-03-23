#!/bin/bash

# AI Chatbot Backend - Setup Script
# This script automates the initial setup process

echo "=================================="
echo "AI Chatbot Backend - Setup Script"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Python is not installed. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment...${NC}"
python -m venv venv

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${RED}✗ Failed to create virtual environment${NC}"
    exit 1
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Check if .env exists
echo ""
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please update .env with your actual API keys and database credentials${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create necessary directories
echo ""
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p app/schemas/tools
mkdir -p streamlit_app
echo -e "${GREEN}✓ Directories created${NC}"

# Summary
echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Update .env file with your credentials"
echo "2. Import MySQL database"
echo "3. Run the backend: python app/main.py"
echo "4. Run the dashboard: streamlit run streamlit_app/dashboard.py"
echo ""
echo "For more information, see README.md"
echo ""
