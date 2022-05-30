import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    n_centers=6, n_samples=1e3, n_features=10, seed=42, snr=3, sample_repartition=None
):
    """
    Generates a synthetic dataset
    Parameters
    ---------
    n_centers: int, default=6
    sample_repartition: float or list of float or None. default=None
        Number of sample per centers. If it is a float, then it is the exponent
        of the power law. If it is a list, it must be of the same size as
        n_centers and contains the weights of being in a given center. If None,
        samples are equally splitted to each of the centers.
    n_samples: int. default=1e3
        Global number of samples.
    n_features: int, default=10
        Dimension of a sample
    classification: boolean
        If we generate a classification problem or a regression one
    label_heterogeneity: float, used for classification problem
        Dirichlet law for classification
    noise_heterogeneity: float or list or None
    features_heterogeneity:
    Returns
    -------
    df_full: pandas dataframe
        Generated data
    indices: list of numpy arrays
        Repartition of the data in the different centers. Each element of the
        list is an array with the indices of the corresponding centers.
    """

    rng = np.random.default_rng(seed=42)

    if sample_repartition is None:
        n_samples_locs = [
            n_samples // n_centers + (i < n_samples % n_centers)
            for i in range(n_centers)
        ]
    elif type(sample_repartition) in [list, np.array]:
        sample_repartition = np.array(sample_repartition)
        assert len(sample_repartition) == n_centers

        positions = np.random.choice(
            n_centers, size=n_samples, replace=True, p=sample_repartition
        )

        n_samples_locs = [np.sum(positions == i) for i in range(n_centers)]

    elif type(sample_repartition) == float:
        assert sample_repartition > 0

        positions = np.random.choice(
            n_centers,
            size=n_samples,
            replace=True,
            p=[i ** (-sample_repartition) for i in range(n_centers)],
        )

        n_samples_locs = [np.sum(positions == i) for i in range(n_centers)]

    else:
        raise ValueError(
            "Incorrect value sample repartition. It must be either a list of\
            weights, or a float, or None."
        )

    indices = []
    start_index = 0
    for n_samples_loc in n_samples_locs:
        indices.append(np.arange(start_index, start_index + n_samples_loc))
        start_index += n_samples_loc

    # generate the true coefficients
    w = rng.lognormal(sigma=1, size=n_features)

    # generate the data for each center
    df_full = pd.DataFrame()
    cov = np.eye(n_features)

    for n_samples_loc in n_samples_locs:
        # generate features
        X = rng.multivariate_normal(
            mean=np.zeros(n_features), cov=cov, size=n_samples_loc
        )
        # generate labels
        y_raw = np.dot(X, w)
        noise_stdev = np.sqrt(np.mean(y_raw**2) / snr)
        noise = rng.normal(scale=noise_stdev, size=n_samples_loc)
        y = np.dot(X, w) + noise

        Xy = np.hstack((X, y[:, None]))
        df = pd.DataFrame(Xy)

        df_full = pd.concat((df_full, df), ignore_index=True)

    return df_full, indices


if __name__ == "__main__":
    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=10, seed=42, snr=3
    )
    print(type(df_full))
    print(type(indices[0]))
