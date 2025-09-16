import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"])
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--where", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)   # for version / rollback
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set env vars for consistency
    os.environ["PROJECT_ID"] = args.project_id
    os.environ["DATASET"] = args.dataset
    os.environ["TABLE"] = args.table
    os.environ["BUCKET_NAME"] = args.bucket_name
    if args.where:
        os.environ["WHERE"] = args.where

    if args.mode == "train":
        print("🔹 Running training...")
        from app.train import train_tft_model
        model_dir = train_tft_model(
            project_id=args.project_id,
            dataset=args.dataset,
            table=args.table,
            bucket_name=args.bucket_name,
            where=args.where,
            model_dir=args.model_dir,
        )
        print(f"🔹 Model stored at: {model_dir}")

    elif args.mode == "predict":
        print("🔹 Running batch prediction...")
        from app.batch_predict import predict
        forecasts = predict(
            bucket_name=args.bucket_name,
            model_dir=args.model_dir,   # explicit version
            project_id=args.project_id,
            dataset=args.dataset,
            table=args.table,
            where=args.where,
        )

        # Save forecasts to CSV
        import pandas as pd
        all_forecasts = []
        for idx, ts in enumerate(forecasts):
            df_pred = ts.pd_dataframe()
            df_pred["series_idx"] = idx
            all_forecasts.append(df_pred)

        pd.concat(all_forecasts).to_csv(args.output_path, index=False)
        print(f"✅ Predictions saved to {args.output_path}")
