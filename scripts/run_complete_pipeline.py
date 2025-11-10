"""Script to run the complete ML pipeline end-to-end."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import datetime


def print_step(step_num, step_name):
    """Print a formatted step header."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {step_name}")
    print("="*70 + "\n")


def run_step(step_num, step_name, script_path):
    """Run a pipeline step."""
    print_step(step_num, step_name)
    
    start_time = time.time()
    
    try:
        # Import and run the script
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            module.main()
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Step {step_num} completed in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Step {step_num} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return False


def main():
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("DL TRADING WITH MLOPS - COMPLETE PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    steps = [
        (1, "Fetch Market Data", os.path.join(base_dir, "01_fetch_data.py")),
        (2, "Engineer Features", os.path.join(base_dir, "02_engineer_features.py")),
        (3, "Train CNN Model", os.path.join(base_dir, "03_train_model.py")),
        (4, "Monitor Data Drift", os.path.join(base_dir, "05_monitor_drift.py")),
        (5, "Run Backtesting", os.path.join(base_dir, "06_run_backtest.py")),
    ]
    
    results = []
    
    for step_num, step_name, script_path in steps:
        success = run_step(step_num, step_name, script_path)
        results.append((step_num, step_name, success))
        
        if not success:
            print("\n⚠️  Pipeline stopped due to error")
            break
    
    # Print summary
    pipeline_time = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for step_num, step_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Step {step_num} - {step_name}: {status}")
    
    total_passed = sum(1 for _, _, success in results if success)
    total_steps = len(steps)
    
    print(f"\nCompleted: {total_passed}/{total_steps} steps")
    print(f"Total time: {pipeline_time:.2f} seconds ({pipeline_time/60:.2f} minutes)")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if total_passed == total_steps:
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. View MLFlow experiments: mlflow ui")
        print("  2. Check reports in: reports/")
        print("  3. Start API server: python scripts/04_run_api.py")
    else:
        print("\n⚠️  Pipeline completed with errors")
        print("Please check the error messages above")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
