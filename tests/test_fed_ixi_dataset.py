def test_general_interface_instantiation():
    from flamby.datasets.fed_ixi import IXIDataset

    # Instantiate generic interface (no errors should be raised)
    dataset = IXIDataset(root='~/.data/')
