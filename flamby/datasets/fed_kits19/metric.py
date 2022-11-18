import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def softmax_helper(x):
    """This function computes the softmax using torch functionnal on the 1-axis.

    Parameters
    ----------
    x : torch.Tensor
        The input.

    Returns
    -------
    torch.Tensor
        Output
    """
    return F.softmax(x, 1)


def Dice_coef(output, target, eps=1e-5):  # dice score used for evaluation
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return num / den, den, num


def metric(predictions, gt):
    gt = gt.float()
    predictions = predictions.float()
    # Compute tumor+kidney Dice >0 (1+2)
    tk_pd = torch.gt(predictions, 0)
    tk_gt = torch.gt(gt, 0)
    tk_dice, denom, num = Dice_coef(tk_pd.float(), tk_gt.float())  # Composite
    tu_dice, denom, num = Dice_coef((predictions == 2).float(), (gt == 2).float())

    return (tk_dice + tu_dice) / 2


def evaluate_dice_on_tests(model, test_dataloaders, metric, use_gpu=True):

    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature:\
            (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. \
        Defaults to True.
#     Returns
#     -------
#     dict
#         A dictionnary with keys client_test_{0} to \
#         client_test_{len(test_dataloaders) - 1} and associated scalar metrics \
#         as leaves.
#     """
    results_dict = {}
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(len(test_dataloaders))):
            dice_list = []
            test_dataloader_iterator = iter(test_dataloaders[i])
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X).detach().cpu()
                preds_softmax = softmax_helper(y_pred)
                preds = preds_softmax.argmax(1)
                y = y.detach().cpu()
                dice_score = metric(preds, y)
                dice_list.append(dice_score)
            results_dict[f"client_test_{i}"] = np.mean(dice_list)
    return results_dict
