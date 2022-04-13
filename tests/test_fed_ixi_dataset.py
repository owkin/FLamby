import pytest


def dataset_test_routine(dataset):
    # Validate subject id verification
    dataset._validate_subject_id('002')
    dataset._validate_subject_id(2)

    # Test len dataset
    print('Dataset length:', len(dataset))

    # Test subject search in TAR file
    img, metadata = dataset[0]
    print(img.shape)

    # %% Test using Dataloader
    from torch.utils.data import DataLoader
    BATCH_SIZE = 10
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = iter(data_loader).next()
    print(len(batch))

    # %% Test visualization
    dataset.simple_visualization()


def test_generic_interface_instantiation():
    # %%
    from flamby.datasets.fed_ixi import IXIDataset

    # Instantiate generic interface (no errors should be raised)
    dataset = IXIDataset(root='~/Downloads/ixi-test')


def test_t1_images_ixi_dataset():
    # %%
    from flamby.datasets.fed_ixi import T1ImagesIXIDataset
    dataset = T1ImagesIXIDataset(root='~/Downloads/')
    assert dataset.modality == 'T1'
    dataset_test_routine(dataset)


def test_t2_images_ixi_dataset():
    # %%
    from flamby.datasets.fed_ixi import T2ImagesIXIDataset
    dataset = T2ImagesIXIDataset(root='~/Downloads/')
    assert dataset.modality == 'T2'
    dataset_test_routine(dataset)


def test_pd_images_ixi_dataset():
    # %%
    from flamby.datasets.fed_ixi import PDImagesIXIDataset
    dataset = PDImagesIXIDataset(root='~/Downloads/')
    assert dataset.modality == 'PD'
    dataset_test_routine(dataset)


def test_mra_images_ixi_dataset():
    # %%
    from flamby.datasets.fed_ixi import MRAImagesIXIDataset
    dataset = MRAImagesIXIDataset(root='~/Downloads/')
    assert dataset.modality == 'MRA'
    dataset_test_routine(dataset)


def test_dti_images_ixi_dataset():
    # %%
    from flamby.datasets.fed_ixi import DTIImagesIXIDataset
    dataset = DTIImagesIXIDataset(root='~/Downloads/')
    assert dataset.modality == 'DTI'
    dataset_test_routine(dataset)


@pytest.mark.skip(reason="Dataset is too large. Meat to be run when necessary.")
def test_dataset_download():
    import shutil
    from pathlib import Path
    from flamby.datasets.fed_ixi import IXIDataset, \
                                        T1ImagesIXIDataset, T2ImagesIXIDataset, \
                                        PDImagesIXIDataset, MRAImagesIXIDataset, DTIImagesIXIDataset

    # Clean previous downloads
    root = Path('~/Downloads/ixi-test/').expanduser()
    shutil.rmtree(root)
    root.mkdir()

    kwargs = dict(root='~/Downloads/ixi-test/', download=True)
    dataset = IXIDataset(**kwargs)  # As parent class downloads only demographics
    dataset = T1ImagesIXIDataset(**kwargs)  # Downloads T1 images (~4.5GiB)
    dataset = T2ImagesIXIDataset(**kwargs)  # Downloads T2 images (~3.6GiB)
    dataset = PDImagesIXIDataset(**kwargs)  # Downloads PD images (~3.8GiB)
    dataset = MRAImagesIXIDataset(**kwargs)  # Downloads MRA images (~12GiB)
    dataset = DTIImagesIXIDataset(**kwargs)  # Downloads DTI images (~4GiB)
