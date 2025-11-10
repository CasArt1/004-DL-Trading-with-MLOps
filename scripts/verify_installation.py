"""Script to verify installation and basic functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    errors = []
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        errors.append(f"✗ pandas: {str(e)}")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {str(e)}")
    
    try:
        import yfinance
        print("✓ yfinance")
    except ImportError as e:
        errors.append(f"✗ yfinance: {str(e)}")
    
    try:
        import tensorflow
        print("✓ tensorflow")
    except ImportError as e:
        errors.append(f"✗ tensorflow: {str(e)}")
    
    try:
        import mlflow
        print("✓ mlflow")
    except ImportError as e:
        errors.append(f"✗ mlflow: {str(e)}")
    
    try:
        import fastapi
        print("✓ fastapi")
    except ImportError as e:
        errors.append(f"✗ fastapi: {str(e)}")
    
    try:
        import uvicorn
        print("✓ uvicorn")
    except ImportError as e:
        errors.append(f"✗ uvicorn: {str(e)}")
    
    try:
        import evidently
        print("✓ evidently")
    except ImportError as e:
        errors.append(f"✗ evidently: {str(e)}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {str(e)}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {str(e)}")
    
    return errors


def check_project_modules():
    """Check if project modules can be imported."""
    print("\nChecking project modules...")
    errors = []
    
    try:
        from src.feature_engineering import TechnicalIndicators, MultiTimeframeFeatures
        print("✓ feature_engineering module")
    except ImportError as e:
        errors.append(f"✗ feature_engineering: {str(e)}")
    
    try:
        from src.models import CNNModel, ModelTrainer
        print("✓ models module")
    except ImportError as e:
        errors.append(f"✗ models: {str(e)}")
    
    try:
        from src.api import app
        print("✓ api module")
    except ImportError as e:
        errors.append(f"✗ api: {str(e)}")
    
    try:
        from src.monitoring import DriftDetector
        print("✓ monitoring module")
    except ImportError as e:
        errors.append(f"✗ monitoring: {str(e)}")
    
    try:
        from src.backtesting import BacktestEngine, TradingStrategy
        print("✓ backtesting module")
    except ImportError as e:
        errors.append(f"✗ backtesting: {str(e)}")
    
    return errors


def check_directory_structure():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    required_dirs = [
        'src',
        'src/feature_engineering',
        'src/models',
        'src/api',
        'src/monitoring',
        'src/backtesting',
        'scripts',
        'configs',
        'notebooks'
    ]
    
    errors = []
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}/")
        else:
            errors.append(f"✗ {dir_path}/ not found")
    
    return errors


def main():
    """Main verification function."""
    print("="*60)
    print("DL Trading with MLOps - Installation Verification")
    print("="*60)
    
    all_errors = []
    
    # Check imports
    import_errors = check_imports()
    all_errors.extend(import_errors)
    
    # Check project modules
    module_errors = check_project_modules()
    all_errors.extend(module_errors)
    
    # Check directory structure
    dir_errors = check_directory_structure()
    all_errors.extend(dir_errors)
    
    # Summary
    print("\n" + "="*60)
    if all_errors:
        print("VERIFICATION FAILED")
        print("="*60)
        print("\nErrors found:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("VERIFICATION SUCCESSFUL")
        print("="*60)
        print("\nAll checks passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python scripts/01_fetch_data.py")
        print("  2. Run: python scripts/02_engineer_features.py")
        print("  3. Run: python scripts/03_train_model.py")
        print("  4. Run: python scripts/04_run_api.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
