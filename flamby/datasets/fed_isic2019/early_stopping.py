import torch
import numpy as np
from pathlib import Path
import os


class EarlyStopping:
    # credits: https://github.com/Bjarten/early-stopping-pytorch
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path, preds_df, df_path, args):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(epoch_score, model, model_path)
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.save_df(epoch_score, preds_df, df_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

    def save_df(self, epoch_score, preds_df, df_path):
        df_path = Path(df_path)
        parent = df_path.parent
        os.makedirs(parent, exist_ok=True)
        preds_df.to_csv(df_path, index=False)
        print(
                "Validation score improved ({} --> {}). Validation predictions saved at at {}!".format(
                    self.val_score, epoch_score, df_path
                )
        )