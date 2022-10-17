import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_heart_disease import BATCH_SIZE, NUM_CLIENTS, FedHeartDisease


def test_normalization():
    # Test normalization in pooled mode
    d_pooled_train = FedHeartDisease(pooled=True, train=True)
    dl_pooled_train = dl(
        d_pooled_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    Xtrain_pooled = torch.zeros(0, 13, dtype=torch.float32)
    for X, y in dl_pooled_train:
        Xtrain_pooled = torch.cat((Xtrain_pooled, X), axis=0)
    assert torch.allclose(Xtrain_pooled.mean(axis=0), torch.zeros((13)), atol=1e-7)
    assert torch.allclose(Xtrain_pooled.std(axis=0), torch.ones((13)), atol=1e-7)
    # Test normalization per center
    training_dls = [
        dl(
            FedHeartDisease(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        for i in range(NUM_CLIENTS)
    ]
    for train_dl in training_dls:
        Xtrain_center = torch.zeros(0, 13, dtype=torch.float32)
        for X, y in dl_pooled_train:
            Xtrain_center = torch.cat((Xtrain_center, X), axis=0)
        assert torch.allclose(Xtrain_center.mean(axis=0), torch.zeros((13)), atol=1e-7)
        assert torch.allclose(Xtrain_center.std(axis=0), torch.ones((13)), atol=1e-7)
