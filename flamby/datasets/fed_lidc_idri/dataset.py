from torch.utils.data import Dataset


class LIDC_IDRIRaw(Dataset):
    """
    Pytorch dataset containing all the features, labels and
    metadata for LIDC-IDRI without any discrimination.

    Attributes
    ----------
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class FedLIDC_IDRI(LIDC_IDRIRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for LIDC-IDRI federated classification.
    """

    def __init__(self, center=0, train=True, pooled=False):
        super().__init__()
        pass
