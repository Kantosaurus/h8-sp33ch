"""
Model Runner - Execute all hate speech classification models
Run individual models or all models at once for comparison
"""

import os
import sys
import time
import argparse
import json
import pandas as pd
from pathlib import Path

# Add the models directory to the path
models_dir = Path(__file__).parent
sys.path.append(str(models_dir))


def run_model(model_name):
    """Run a specific model and return performance metrics"""
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
            return False, None

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n{model_name.upper()} completed in {execution_time:.2f} seconds")

        # Extract metrics from model output directory
        metrics = extract_model_metrics(model_name, execution_time)

        return True, metrics

    except Exception as e:
        print(f"Error running {model_name}: {e}")
        return False, None


def extract_model_metrics(model_name, execution_time):
    """Extract performance metrics from model outputs"""
    # Define output directory mapping
    output_dirs = {
        "extra_trees": "extra_trees_outputs",
        "random_forest": "random_forest_outputs",
        "xgboost": "xgboost_outputs",
        "catboost": "catboost_outputs",
        "qda": "qda_outputs",
        "lightgbm": "lightgbm_outputs",
        "decision_tree": "decision_tree_outputs",
        "lda": "lda_outputs",
        "gradient_boosting": "gradient_boosting_outputs",
        "ridge": "ridge_outputs",
        "knn": "knn_outputs",
    }

    metrics = {
        "Model": model_name,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "TT (Sec)": execution_time,
    }

    # Try to read metrics from a metrics file if it exists
    output_dir = output_dirs.get(model_name, f"{model_name}_outputs")
    metrics_file = os.path.join(output_dir, "metrics.json")

    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                stored_metrics = json.load(f)
                metrics.update(stored_metrics)
        except Exception as e:
            print(f"Could not read metrics file for {model_name}: {e}")
    else:
        # Return None if no metrics file exists and no execution time provided
        if execution_time is None:
            return None

    return metrics


def run_all_models():
    """Run all models sequentially and generate comparison report"""
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
    all_metrics = []

    for model in models:
        success, metrics = run_model(model)
        if success:
            successful_runs.append(model)
            if metrics:
                all_metrics.append(metrics)
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

    # Generate comparison report
    if all_metrics:
        generate_comparison_report(all_metrics)

    print(
        f"\nSubmission files are organized in individual model directories (e.g., extratrees_outputs/, randomforest_outputs/, etc.)."
    )


def generate_comparison_report(all_metrics):
    """Generate a comprehensive comparison report of all models"""
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*80}")

    if not all_metrics:
        print("No metrics available for comparison.")
        return

    # Create DataFrame for better formatting
    try:
        import pandas as pd

        df = pd.DataFrame(all_metrics)

        # Sort by accuracy (if available) or by model name
        if "Accuracy" in df.columns and df["Accuracy"].notna().any():
            df = df.sort_values("Accuracy", ascending=False)
        else:
            df = df.sort_values("Model")

        # Format the table
        print("\nPerformance Summary:")
        print("-" * 80)

        # Print header
        print(
            f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time (s)':<10}"
        )
        print("-" * 80)

        # Print data rows
        for _, row in df.iterrows():
            model = row.get("Model", "N/A")[:18]
            accuracy = (
                f"{row.get('Accuracy', 0):.4f}"
                if row.get("Accuracy") is not None
                else "N/A"
            )
            precision = (
                f"{row.get('Precision', 0):.4f}"
                if row.get("Precision") is not None
                else "N/A"
            )
            recall = (
                f"{row.get('Recall', 0):.4f}"
                if row.get("Recall") is not None
                else "N/A"
            )
            f1 = f"{row.get('F1', 0):.4f}" if row.get("F1") is not None else "N/A"
            time_taken = (
                f"{row.get('TT (Sec)', 0):.2f}"
                if row.get("TT (Sec)") is not None
                else "N/A"
            )

            print(
                f"{model:<20} {accuracy:<10} {precision:<10} {recall:<10} {f1:<10} {time_taken:<10}"
            )

        # Save to CSV file
        report_file = "model_comparison_report.csv"
        df.to_csv(report_file, index=False)
        print(f"\nDetailed comparison report saved to: {report_file}")

        # Print top performers
        if "Accuracy" in df.columns and df["Accuracy"].notna().any():
            best_accuracy = df.loc[df["Accuracy"].idxmax()]
            print(
                f"\n🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})"
            )

        if "F1" in df.columns and df["F1"].notna().any():
            best_f1 = df.loc[df["F1"].idxmax()]
            print(f"🎯 Best F1 Score: {best_f1['Model']} ({best_f1['F1']:.4f})")

        if "TT (Sec)" in df.columns and df["TT (Sec)"].notna().any():
            fastest = df.loc[df["TT (Sec)"].idxmin()]
            print(f"⚡ Fastest Model: {fastest['Model']} ({fastest['TT (Sec)']:.2f}s)")

    except ImportError:
        # Fallback to simple text formatting if pandas not available
        print("\nPerformance Summary (Simple Format):")
        print("-" * 60)
        for metrics in all_metrics:
            model = metrics.get("Model", "Unknown")
            accuracy = metrics.get("Accuracy", "N/A")
            f1 = metrics.get("F1", "N/A")
            time_taken = metrics.get("TT (Sec)", "N/A")

            print(f"{model}: Accuracy={accuracy}, F1={f1}, Time={time_taken}s")


def compare_models(model_names=None):
    """Compare specific models or all available models"""
    if model_names is None:
        model_names = [
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

    print(f"\n{'='*60}")
    print("MODEL COMPARISON MODE")
    print(f"{'='*60}")
    print("Loading metrics from existing model outputs...")

    all_metrics = []

    for model_name in model_names:
        # Try to get metrics from existing outputs without re-running
        metrics = extract_model_metrics(model_name, None)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        generate_comparison_report(all_metrics)
    else:
        print(
            "No model metrics found. Please run models first using --all or individual --model commands."
        )


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
  python model_runner.py --compare                # Compare model performance
        """,
    )

    parser.add_argument("--model", type=str, help="Name of specific model to run")

    parser.add_argument("--all", action="store_true", help="Run all models")

    parser.add_argument("--list", action="store_true", help="List available models")

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare model performance from existing outputs",
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
    elif args.compare:
        compare_models()
    elif args.all:
        run_all_models()
    elif args.model:
        success, metrics = run_model(args.model)
        if success and metrics:
            print(f"\nModel Metrics Summary:")
            for key, value in metrics.items():
                if value is not None:
                    print(f"  {key}: {value}")
    else:
        print("Please specify --model <name>, --all, --compare, or --list")
        parser.print_help()


if __name__ == "__main__":
    main()
