from flamby.datasets.fed_kits19.dataset_creation_scripts.nnunet_library.set_environment_variables import (
    set_environment_variables,
)
from flamby.utils import get_config_file_path, write_value_in_config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # set_environment_variables should be called before importing nnunet
    set_environment_variables()
    from nnunet.experiment_planning.nnUNet_plan_and_preprocess import main

    main()
    path_to_config_file = get_config_file_path("fed_kits19", False)
    write_value_in_config(path_to_config_file, "preprocessing_complete", True)
