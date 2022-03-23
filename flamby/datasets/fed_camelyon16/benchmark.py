from flamby.datasets.fed_camelyon16 import FedCamelyon16, Baseline, BaselineLoss, metric, collate_fn
from torch.utils.data import DataLoader as dl
import torch.optim as optim
from flamby.utils import evaluate_model_on_tests
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

torch.use_deterministic_algorithms(True)


BATCH_SIZE = 32
NUM_WORKERS_TORCH = 20
NUM_EPOCHS = 40
DEBUG = False
LOG = True
LOG_PERIOD = 10 
LR = 0.001

metrics_dict = {"AUC": metric}
training_dl = dl(FedCamelyon16(train=True, pooled=True, debug=DEBUG), num_workers=NUM_WORKERS_TORCH, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
test_dl = dl(FedCamelyon16(train=False, pooled=True, debug=DEBUG), num_workers=NUM_WORKERS_TORCH, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)


if LOG:
    # We compute the number of batches per epoch
    num_local_steps_per_epoch = len(training_dl.dataset) // BATCH_SIZE 
    num_local_steps_per_epoch += int((len(training_dl.dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0)


results = []
for seed in range(42, 47):
    # At each new seed we re-initialize the model
    # and training_dl is shuffled as well  
    torch.manual_seed(seed) 
    m = Baseline()
    # We put the model on GPU whenever it is possible
    if torch.cuda.is_available():
        m = m.cuda()
    loss = BaselineLoss()
    optimizer = optim.Adam(m.parameters(), lr=LR)
    if LOG:
        # We create one summarywriter for each seed in order to overlay the plots
        writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

    for e in tqdm(range(NUM_EPOCHS)):
        if LOG:
            # At each epoch we look at the histograms of all the network's parameters
            for name, p in m.named_parameters():
                writer.add_histogram(f"client_0/{name}", p, e)
        for s, (X, y) in enumerate(training_dl):
            # traditional training loop with optional GPU transfer
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
    
            optimizer.zero_grad()
            y_pred = m(X)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            if LOG:
                current_step = s + num_local_steps_per_epoch * e
                if (current_step % LOG_PERIOD) == 0:
                    writer.add_scalar(f"Loss/train/client", l.item(), s + num_local_steps_per_epoch * e)
                    for k, v in metrics_dict.items():
                        train_batch_metric = v(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                        writer.add_scalar(f"{k}/train/client", train_batch_metric, s + num_local_steps_per_epoch * e)

    current_results_dict = evaluate_model_on_tests(m, [test_dl], metric)
    print(current_results_dict)
    results.append(current_results_dict["client_test_0"])

results = np.array(results)

if LOG:
    writer = SummaryWriter(log_dir=f"./runs/tests")
    for i in range(results.shape[0]):
        writer.add_scalar(f"AUC/client_test", results[i], 0)


print("Benchmark Results on Camelyon16 pooled:")
print(f"mAUC on 5 runs: {results.mean(): .2%} \pm {results.std(): .2%}")
