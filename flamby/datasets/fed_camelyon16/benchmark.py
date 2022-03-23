from flamby.datasets.fed_camelyon16 import FedCamelyon16, Baseline, BaselineLoss, metric, collate_fn
from torch.utils.data import DataLoader as dl
import torch.optim as optim
from flamby.utils import evaluate_model_on_tests
import torch


BATCH_SIZE = 64
NUM_WORKERS_TORCH = 10
NUM_EPOCHS = 0
DEBUG = False

training_dl = dl(FedCamelyon16(train=True, pooled=True, debug=DEBUG), num_workers=NUM_WORKERS_TORCH, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
test_dl = dl(FedCamelyon16(train=False, pooled=True, debug=DEBUG), num_workers=NUM_WORKERS_TORCH, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

m = Baseline()
loss = BaselineLoss()
optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

for e in range(NUM_EPOCHS):
    for X, y in training_dl:
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        
        optimizer.zero_grad()
        y_pred = m(X)
        breakpoint()
        l = loss(y_pred, y)
        l.backward()
        otimizer.step()

print(evaluate_model_on_tests(m, [test_dl], metric))



        







