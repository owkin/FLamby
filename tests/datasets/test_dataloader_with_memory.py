import torch

# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_tcga_brca import BATCH_SIZE, NUM_CLIENTS
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset


class DataLoaderWithMemory:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(self, dataloader):
        """This initialization takes a dataloader and creates an iterator object
        from it.

        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            A dataloader object built from one of the datasets of this repository.
        """
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.

        Returns
        -------
        tuple
            a batch from the iterator
        """
        try:
            X, y = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            X, y = next(self._iterator)
        return X, y


# We loop on all the clients of the distributed dataset
# and instantiate associated data loaders
train_dataloaders = [
    torch.utils.data.DataLoader(
        FedDataset(center=i, train=True, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    for i in range(NUM_CLIENTS)
]


def test_next_function():
    # this code worked in torch 1.12
    dataloader = train_dataloaders[0]
    iterator = iter(dataloader)
    try:
        _ = iterator.next()
        assert False
    except AttributeError:
        assert True

    # This one does work in both :
    try:
        _ = next(iterator)
        assert True
    except:  # noqa:E722 Handled
        assert False
