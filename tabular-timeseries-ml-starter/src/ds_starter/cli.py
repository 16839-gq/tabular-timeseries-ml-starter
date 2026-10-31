"""Command line interface for experiments and future predictions."""

import argparse

from ds_starter.runner import predict_from_bundle, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular / time-series ML starter")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Train, cross-validate and evaluate an experiment")
    run.add_argument("--config", required=True, help="YAML file relative to project root")
    predict = sub.add_parser("predict", help="Predict with a trusted saved pipeline")
    predict.add_argument("--model", required=True, help="Trained .joblib path")
    predict.add_argument("--input", required=True, help="CSV of new unlabeled observations")
    predict.add_argument("--output", required=True, help="Output prediction CSV")
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(args.config)
        print(f"Selected model: {result['chosen_model']}")
        print("Cross-validation summary:\n", result["cv_summary"].round(3).to_string())
        print("Holdout scores:\n", result["holdout_scores"].round(3).to_string())
        print(f"Reports: {result['report_dir']}")
        print(f"Saved pipeline: {result['model_dir']}/trained_pipeline.joblib")
    else:
        output = predict_from_bundle(args.model, args.input, args.output)
        print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
