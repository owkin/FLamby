import argparse
import os

import numpy as np

from flamby.datasets.fed_synthetic.synthetic_generator import generate_synthetic_dataset
from flamby.utils import create_config, get_config_file_path, write_value_in_config


def main(output_folder, debug=False, **kwargs):
    """Generates a synthetic dataset.

    Parameters
    ----------
    output_folder : str
        The folder where to download the dataset.
    """

    if (
        kwargs["sample_repartition"] is not None
        and len(kwargs["sample_repartition"][0]) == 1
    ):
        kwargs["sample_repartition"] = np.array(kwargs["sample_repartition"]).item()

    os.makedirs(output_folder, exist_ok=True)

    # Erase existing config file
    if os.path.exists(get_config_file_path("fed_synthetic", debug)):
        os.remove(get_config_file_path("fed_synthetic", debug))

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_synthetic")

    # Generate data
    df_full, indices = generate_synthetic_dataset(**kwargs)

    # save each center's dataset
    for center in range(kwargs["n_centers"]):
        fname = str(center) + ".data"

        df_loc = df_full.iloc[indices[center]]
        df_loc.to_csv(output_folder + "/" + fname, index=False, header=None)

    # save arguments
    write_value_in_config(config_file, "download_complete", True)
    write_value_in_config(config_file, "preprocessing_complete", True)
    write_value_in_config(config_file, "n_centers", kwargs["n_centers"])
    write_value_in_config(config_file, "n_samples", kwargs["n_samples"])
    write_value_in_config(config_file, "n_features", kwargs["n_features"])
    write_value_in_config(config_file, "seed", kwargs["seed"])
    write_value_in_config(config_file, "snr", kwargs["snr"])
    write_value_in_config(config_file, "classification", kwargs["classification"])
    write_value_in_config(
        config_file, "sample_repartition", kwargs["sample_repartition"]
    )
    write_value_in_config(
        config_file, "noise_heterogeneity", kwargs["noise_heterogeneity"]
    )
    write_value_in_config(
        config_file, "features_heterogeneity", kwargs["features_heterogeneity"]
    )
    write_value_in_config(
        config_file, "label_heterogeneity", kwargs["label_heterogeneity"]
    )
    write_value_in_config(config_file, "n_clusters", kwargs["n_clusters"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centers",
        type=int,
        default=6,
        help="Number of centers in the generated dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Total number of samples in the generated dataset.",
        default=1000,
    )
    parser.add_argument(
        "--features",
        type=int,
        help="Number of features in the generated dataset.",
        default=10,
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for the random number generator.", default=42
    )
    parser.add_argument("--snr", type=float, help="Signal to noise ratio.", default=3)
    parser.add_argument(
        "--sample-repartition",
        nargs="+",
        action="append",
        type=float,
        help="Sample repartition.",
        default=None,
    )
    parser.add_argument(
        "--noise-heterogeneity", type=float, help="Sample repartition.", default=None
    )
    parser.add_argument(
        "--features-heterogeneity",
        type=float,
        help="If regression, centers of the features per center."
        "If classification, ratio between intra/extra cluster distances.",
        default=None,
    )
    parser.add_argument(
        "--label-heterogeneity",
        type=float,
        help="Parameter of the Dirichlet law for label heterogeneity"
        " (used for classification).",
        default=None,
    )
    parser.add_argument(
        "--clusters",
        type=int,
        help="Number of clusters (for classification).",
        default=3,
    )

    classif_parser = parser.add_mutually_exclusive_group(required=False)
    classif_parser.add_argument(
        "--classification",
        dest="classification",
        action="store_true",
        help="Generate a classification dataset.",
    )
    classif_parser.add_argument(
        "--regression",
        dest="classification",
        action="store_false",
        help="Generate a regression dataset. (Default)",
    )
    parser.set_defaults(classification=True)

    parser.add_argument(
        "--output-folder",
        type=str,
        help="Where to store the downloaded data.",
        required=True,
    )

    args = parser.parse_args()
    main(
        args.output_folder,
        n_centers=args.centers,
        n_samples=args.samples,
        n_features=args.features,
        seed=args.seed,
        snr=args.snr,
        classification=args.classification,
        sample_repartition=args.sample_repartition,
        noise_heterogeneity=args.noise_heterogeneity,
        features_heterogeneity=args.features_heterogeneity,
        label_heterogeneity=args.label_heterogeneity,
        n_clusters=args.clusters,
    )
