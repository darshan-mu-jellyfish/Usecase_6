import argparse
import os
from train import train_tft_model
from batch_predict import predict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"])
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--where", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["PROJECT_ID"] = args.project_id
    os.environ["DATASET"] = args.dataset
    os.environ["TABLE"] = args.table
    os.environ["BUCKET_NAME"] = args.bucket_name
    if args.where:
        os.environ["WHERE"] = args.where

    if args.mode == "train":
        print("Running training...")
        train_tft_model(args.project_id, args.dataset, args.table, args.bucket_name, args.where, args.model_dir)
    elif args.mode == "predict":
        print("Running batch prediction...")
        forecasts = predict(args.bucket_name, args.model_dir, args.project_id, args.dataset, args.table, args.where)
        print("Predictions done.")
