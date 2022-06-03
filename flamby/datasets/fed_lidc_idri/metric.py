import numpy as np
import torch
from tqdm import tqdm

torch.manual_seed(42)


def metric(y_true, y_pred):
    """
    Soft Dice coefficient
    """
    intersection = np.sum(y_pred * y_true, axis=tuple(range(1, y_true.ndim)))
    union = np.sum(0.5 * (y_pred + y_true), axis=tuple(range(1, y_true.ndim)))
    dice = intersection / np.maximum(union, 1.0e-7)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1

    return np.mean(dice)


def evaluate_dice_on_tests_by_chunks(model, test_dataloaders, use_gpu=True, nchunks=4):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the dice coefficient. The dice coefficient is computed by splitting
    the list of patches in chunks, to fit in memory.
    WARNING : assumes batches of size, i.e. input patches come from a single image.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available.
        Defaults to True.
    nchunks: int, default = 4
        Number of chunks to split the list of patches from the dataloader
    Returns
    -------
    dict
        A dictionary with keys client_test_{0} to
        client_test_{len(test_dataloaders) - 1} and associated dice as leaves.
    """
    results_dict = {}
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            dices = []
            for (X, y) in test_dataloader_iterator:
                intersection = 0
                union = 0
                X = torch.chunk(X, nchunks)
                y = torch.chunk(y, nchunks)
                for ii, X_ in enumerate(X):
                    y_ = y[ii]
                    if torch.cuda.is_available() and use_gpu:
                        X_ = X_.cuda()
                        model = model.cuda()
                    y_pred = model(X_).detach().cpu().numpy()
                    y_ = y_.detach().cpu().numpy()
                    intersection += np.sum(y_pred * y_)
                    union += np.sum(0.5 * (y_pred + y_))
                dice = 1 if np.abs(union) < 1e-7 else intersection / union
                dices.append(dice)
            results_dict[f"client_test_{i}"] = np.mean(dices)
    return results_dict
