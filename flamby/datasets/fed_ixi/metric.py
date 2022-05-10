import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def metric(y_pred, y_true):
    """
    Soft Dice coefficient
    """
    SPATIAL_DIMENSIONS = 2, 3, 4
    intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
    union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
    dice = intersection / (union + 1.0e-7)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1
    return np.mean(dice)

def evaluate_dice_on_tests(model, test_dataloaders, use_gpu=True):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the dice coefficient.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available.
        Defaults to True.
    Returns
    -------
    dict
        A dictionary with keys client_test_{0} to
        client_test_{len(test_dataloaders) - 1} and associated dice as leaves.
    """
    CHANNELS_DIMENSION = 1
    results_dict = {}
    with torch.inference_mode():
        model.eval()
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            dices = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X)
                probabilities = F.softmax(y_pred, dim=CHANNELS_DIMENSION).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                dice = metric(probabilities, y)
                dices.append(dice)
            results_dict[f"client_test_{i}"] = np.mean(dices)
    return results_dict

if __name__ == '__main__':
    print(metric(np.ones((10,1,10,10,10)),np.ones((10,1,10,10,10))))
    print(metric((np.random.rand(10,1,10,10,10) > 0.5),np.ones((10,1,10,10,10))))