# Adapted from the ooriginal DeepMIL implementation from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        # As per the article
        self.O = 2048 # Original dimension of the input embeddings
        self.M = 128 # New dimension of the input embedding

        self.L = 128 # Dimension of the new features after query and value projections
        self.K = 10000 # Number of elements in each bag

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(self.O, self.M),
        )
        # The Gated Attention using tanh and sigmoid from Eq 9 from https://arxiv.org/abs/1802.04712

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.L, 1)

        # The classifier follows https://arxiv.org/pdf/2012.03583.pdf Supplementary Table 3
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = self.feature_extractor_part1(x) # BxKxM

        # Computing the attention mask A
        A_V = self.attention_V(H)  # BxKxL
        A_U = self.attention_U(H)  # BxKxL
        A = self.attention_weights(A_V * A_U) # element wise multiplication # BxKx1
        A = F.softmax(A, dim=1)  # softmax over the bag dimension K to normalize 

        # We multiply the features by the attention mask
        M = torch.matmul(A.transpose(1, 2), H).squeeze(1) # BxL
        # We further build
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A