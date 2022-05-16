import torch
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)
def Dice_coef(output, target, eps=1e-5):  # dice score used for evaluation
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return num / den, den, num

def metrics(predictions, gt):
    gt = gt.float()
    predictions = predictions.float()
    # Compute tumor+kidney Dice >0 (1+2)
    tk_pd = torch.gt(predictions, 0)
    tk_gt = torch.gt(gt, 0)
    tk_dice, denom, num = Dice_coef(tk_pd.float(), tk_gt.float())  # Composite
    tu_kid_dice, denom, num = Dice_coef((predictions == 1).float(), (gt == 1).float())
    tu_dice, denom, num = Dice_coef((predictions == 2).float(), (gt == 2).float())

    return tk_dice, tu_kid_dice, tu_dice
