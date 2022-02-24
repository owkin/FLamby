import torch
from torch.utils.data import DataLoader as dl
import numpy as np
import copy
from tqdm import tqdm
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

def evaluate_model_on_tests(model, test_dataloaders, metric, use_gpu=True):
    """This function takes a pytorch model and evaluate it on a list of dataloaders using the
    provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. Defaults to True.
    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to client_test_{len(test_dataloaders) - 1} and associated scalar metrics as leaves.
    """
    results_dict = {}
    with torch.inference_mode():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final = []
            y_true_final = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                    model = model.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())
            y_true_final = np.vstack(y_true_final)
            y_pred_final = np.vstack(y_pred_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
    return results_dict