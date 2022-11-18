#    Copyright 2020 Division of Medical Image Computing, German Cancer Research
# Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Part of this file comes from https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet
# See flamby/datasets/fed_kits19/dataset_creation_scripts/LICENSE/README.md for more
# information
import sys

from flamby.datasets.fed_kits19.dataset_creation_scripts.utils import (
    set_environment_variables,
)
from flamby.utils import get_config_file_path, write_value_in_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Specify if debug mode (True) or not (False)",
    )
    parser.add_argument("--num_threads", default="1", help="Number of threads used.")
    args = parser.parse_args()

    # set_environment_variables should be called before importing nnunet
    set_environment_variables(args.debug)
    from nnunet.experiment_planning.nnUNet_plan_and_preprocess import main

    # We need to remove --debug and '--num_threads' from sys.argv as it is not listed in
    # the CLI of nnunet.experiment_planning.nnUNet_plan_and_preprocess.main()
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
    if "--num_threads" in sys.argv:
        index_num_threads = sys.argv.index("--num_threads")
        for _ in range(2):
            sys.argv.pop(index_num_threads)

    sys.argv = sys.argv + ["-t", "064", "-tf", args.num_threads, "-tl", args.num_threads]

    main()
    path_to_config_file = get_config_file_path("fed_kits19", False)
    write_value_in_config(path_to_config_file, "preprocessing_complete", True)
