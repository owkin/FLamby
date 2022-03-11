import os
import pandas as pd 
from sklearn import model_selection
import numpy as np


if __name__ == '__main__':
    input_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    kf = model_selection.StratifiedKFold(n_splits=5)
    df = pd.read_csv(os.path.join(input_path, 'ISIC_2019_Training_GroundTruth.csv'))
    onehot = df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
    df['target'] = [np.where(r==1)[0][0] for r in onehot]
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.target.values
    for fold, (train_index, test_index) in enumerate(kf.split(X=df[['image']], y=targets)):
        df.loc[test_index, 'kfold'] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
