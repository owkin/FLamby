from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss


class BaselineLoss(DC_and_CE_loss):
    def __init__(
        self,
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False},
        ce_kwargs={},
        aggregate="sum",
        square_dice=False,
        weight_ce=1,
        weight_dice=1,
        log_dice=False,
        ignore_label=None,
    ):
        super(BaselineLoss, self).__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate,
            square_dice=square_dice,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            log_dice=log_dice,
            ignore_label=ignore_label,
        )
