import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

import optuna

import xg_lstm


TRAIN_PATH = xg_lstm.resolve_path("train.parquet")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"]= str(seed)
    np.random.seed(seed)


def evaluate_params(params: Dict) -> float:
    seed_everything(42)
    df = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    ap_scores = []
    for train_idx, val_idx in splits.split(df, df['clicked']):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        params_ = params.copy()
        params_.update({
            "num_cols": xg_lstm.num_cols,
            "cat_cols": xg_lstm.cat_cols,
            "seq_col": xg_lstm.SEQ_COL,
            "target_col": xg_lstm.TARGET_COL,
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "lr": params["lr"],
            "device": xg_lstm.DEVICE,
        })
        model = xg_lstm.train_model(train_fold, **params_)
        val = df.iloc[val_idx].reset_index(drop=True)
        val_dataset = xg_lstm.ClickDataset(val, xg_lstm.num_cols, xg_lstm.cat_cols, xg_lstm.SEQ_COL)
        loader = xg_lstm.DataLoader(val_dataset, batch_size=1024, shuffle=False, collate_fn=xg_lstm.collate_fn_infer, pin_memory=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for num_x, cat_x, seqs, lens in loader:
                num_x, cat_x, seqs, lens = num_x.to(xg_lstm.DEVICE), cat_x.to(xg_lstm.DEVICE), seqs.to(xg_lstm.DEVICE), lens.to(xg_lstm.DEVICE)
                prob = torch.sigmoid(model(num_x, cat_x, seqs, lens)).cpu().numpy()
                preds.append(prob)
        preds = np.concatenate(preds)
        ap_scores.append(average_precision_score(val['clicked'], preds))
    return float(np.mean(ap_scores))


def objective(trial: optuna.Trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024]),
        "epochs": trial.suggest_int("epochs", 3, 6),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "model_hidden": trial.suggest_int("hidden_size", 32, 128, step=32),
    }
    score = evaluate_params(params)
    trial.set_user_attr("ap", score)
    return score


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)
    print("Best AP:", study.best_value)


if __name__ == "__main__":
    main()

