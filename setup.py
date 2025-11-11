"""
Setup script to initialize the project environment and download initial data.
Run this script first after cloning the repository.
"""

import subprocess
import sys
import os
from pathlib import Path


def create_venv():
    """Create a virtual environment."""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    print("✓ Virtual environment created")


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    
    # Determine the python executable in venv
    if sys.platform == "win32":
        python_exe = Path(".venv/Scripts/python.exe")
        pip_exe = Path(".venv/Scripts/pip.exe")
    else:
        python_exe = Path(".venv/bin/python")
        pip_exe = Path(".venv/bin/pip")
    
    # Upgrade pip
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    print("✓ Requirements installed")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("\nCreating .env file from .env.example...")
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("✓ .env file created. Please update it with your configuration.")
        else:
            print("⚠ .env.example not found")
    else:
        print("\n✓ .env file already exists")


def verify_directories():
    """Verify all necessary directories exist."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "mlruns",
        "notebooks",
        "config",
        "src"
    ]
    
    print("\nVerifying directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure verified")


def main():
    """Main setup function."""
    print("=" * 60)
    print("DL Trading with MLOps - Project Setup")
    print("=" * 60)
    
    try:
        # Check if venv already exists
        if not os.path.exists(".venv"):
            create_venv()
        else:
            print("✓ Virtual environment already exists")
        
        install_requirements()
        create_env_file()
        verify_directories()
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if sys.platform == "win32":
            print("   .venv\\Scripts\\Activate.ps1  (PowerShell)")
            print("   .venv\\Scripts\\activate.bat  (CMD)")
        else:
            print("   source .venv/bin/activate")
        print("\n2. Update config/config.yaml with your asset symbol and dates")
        print("\n3. Run data collection:")
        print("   python src/data_collection.py")
        print("\n4. Follow the project workflow in the README")
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
