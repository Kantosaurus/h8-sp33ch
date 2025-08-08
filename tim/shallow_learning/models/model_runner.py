"""
Model Runner - Execute all hate speech classification models
Run individual models or all models at once for comparison
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the models directory to the path
models_dir = Path(__file__).parent
sys.path.append(str(models_dir))


def run_model(model_name):
    """Run a specific model"""
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} Model")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        if model_name == "extra_trees":
            from extra_trees_classifier import main

            main()
        elif model_name == "random_forest":
            from random_forest_classifier import main

            main()
        elif model_name == "xgboost":
            from xgboost_classifier import main

            main()
        elif model_name == "catboost":
            from catboost_classifier import main

            main()
        elif model_name == "qda":
            from qda_classifier import main

            main()
        elif model_name == "lightgbm":
            from lightgbm_classifier import main

            main()
        elif model_name == "decision_tree":
            from decision_tree_classifier import main

            main()
        elif model_name == "lda":
            from lda_classifier import main

            main()
        elif model_name == "gradient_boosting":
            from gradient_boosting_classifier import main

            main()
        elif model_name == "ridge":
            from ridge_classifier import main

            main()
        elif model_name == "knn":
            from knn_classifier import main

            main()
        else:
            print(f"Unknown model: {model_name}")
            return False

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n{model_name.upper()} completed in {execution_time:.2f} seconds")
        return True

    except Exception as e:
        print(f"Error running {model_name}: {e}")
        return False


def run_all_models():
    """Run all models sequentially"""
    models = [
        "extra_trees",
        "random_forest",
        "xgboost",
        "catboost",
        "qda",
        "lightgbm",
        "decision_tree",
        "lda",
        "gradient_boosting",
        "ridge",
        "knn",
    ]

    print("Running all machine learning models for hate speech classification...")
    print(f"Total models to run: {len(models)}")

    start_time = time.time()
    successful_runs = []
    failed_runs = []

    for model in models:
        success = run_model(model)
        if success:
            successful_runs.append(model)
        else:
            failed_runs.append(model)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(
        f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"Successful runs: {len(successful_runs)}/{len(models)}")

    if successful_runs:
        print(f"\nSuccessful models:")
        for model in successful_runs:
            print(f"  ✓ {model}")

    if failed_runs:
        print(f"\nFailed models:")
        for model in failed_runs:
            print(f"  ✗ {model}")

    print(f"\nSubmission files should be available in the current directory.")


def list_available_models():
    """List all available models"""
    models = [
        ("extra_trees", "Extra Trees Classifier - 83.84% accuracy"),
        ("random_forest", "Random Forest Classifier - 83.49% accuracy"),
        ("xgboost", "XGBoost Classifier - 81.44% accuracy"),
        ("catboost", "CatBoost Classifier - 78.99% accuracy"),
        ("qda", "Quadratic Discriminant Analysis - 78.70% accuracy"),
        ("lightgbm", "LightGBM Classifier - 78.43% accuracy"),
        ("decision_tree", "Decision Tree Classifier - 75.67% accuracy"),
        ("lda", "Linear Discriminant Analysis - 74.60% accuracy"),
        ("gradient_boosting", "Gradient Boosting Classifier - 72.88% accuracy"),
        ("ridge", "Ridge Classifier - 71.39% accuracy"),
        ("knn", "K-Nearest Neighbors Classifier - Performance TBD"),
    ]

    print("Available models:")
    print("=" * 80)
    for name, description in models:
        print(f"{name:<20} : {description}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Run hate speech classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_runner.py --all                    # Run all models
  python model_runner.py --model extra_trees      # Run specific model
  python model_runner.py --list                   # List available models
        """,
    )

    parser.add_argument("--model", type=str, help="Name of specific model to run")

    parser.add_argument("--all", action="store_true", help="Run all models")

    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        list_available_models()
    elif args.all:
        run_all_models()
    elif args.model:
        run_model(args.model)
    else:
        print("Please specify --model <name>, --all, or --list")
        parser.print_help()


if __name__ == "__main__":
    main()
