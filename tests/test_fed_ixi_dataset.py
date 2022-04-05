def test_generic_interface_instantiation():
    # %%
    from flamby.datasets.fed_ixi import IXIDataset

    # Instantiate generic interface (no errors should be raised)
    dataset = IXIDataset(root='~/Downloads/')

    # Validate subject id verification
    dataset._validate_subject_id('002')
    dataset._validate_subject_id(2)

    # Test len dataset
    print('Dataset length:', len(dataset))

    # Test subject search in TAR file
    img = dataset[0]
    print(img)


