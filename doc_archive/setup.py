#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    if sys.version_info < (3, 9):
        print("Python 3.9 or higher is required")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    dirs = [
        "documents",
        "temp",
        "vectorstore",
        "logs"
    ]
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_name}")

def setup_virtual_environment():
    """Set up virtual environment and install dependencies"""
    if not Path("venv").exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Created virtual environment")

    # Get the pip path in the virtual environment
    if os.name == "nt":  # Windows
        pip_path = "venv/Scripts/pip"
    else:  # Unix/Linux/MacOS
        pip_path = "venv/bin/pip"

    # Upgrade pip
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("Installed dependencies")

def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        print("Created .env file from .env.example")
        print("Please update the .env file with your actual configuration")

def main():
    """Main setup function"""
    print("Starting setup...")
    check_python_version()
    create_directories()
    setup_virtual_environment()
    create_env_file()
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your API keys and configuration")
    print("2. Activate the virtual environment:")
    if os.name == "nt":  # Windows
        print("   .\\venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("   source venv/bin/activate")
    print("3. Start the server:")
    print("   uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()