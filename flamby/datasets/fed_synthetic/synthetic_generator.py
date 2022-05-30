import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    n_centers=6,
    n_samples=1e3,
    n_features=10,
    seed=42,
    snr=3,
    sample_repartition=None,
    noise_heterogeneity=None,
    features_heterogeneity=None,
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
    features_heterogeneity: float or list or None
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
        sample_repartition = sample_repartition / np.sum(sample_repartition)
        assert len(sample_repartition) == n_centers
        assert (sample_repartition >= 0).all()

        positions = np.random.choice(
            n_centers, size=n_samples, replace=True, p=sample_repartition
        )

        n_samples_locs = [np.sum(positions == i) for i in range(n_centers)]

    elif type(sample_repartition) == float:
        assert sample_repartition > 1

        weights = [(i+1) ** (-sample_repartition+1) for i in range(n_centers)]
        prob = weights / np.sum(weights)

        positions = np.random.choice(
            n_centers,
            size=n_samples,
            replace=True,
            p=prob,
        )

        n_samples_locs = [np.sum(positions == i) for i in range(n_centers)]

    else:
        raise ValueError(
            "Incorrect value sample repartition. It must be either a list of\
            weights, or a float, or None."
        )
    
    if features_heterogeneity is None:
        features_locs = np.zeros((n_centers, n_features))
    elif type(features_heterogeneity) == float:
        # TODO: discuss about what does we really want here with the other
        features_locs = np.random.random((n_centers, n_features)) * features_heterogeneity
    else:
        raise ValueError(
            "Incorrect value features_heterogeneity. It must be either a float, or None."
        )
    
    if noise_heterogeneity is None:
        snr_locs = np.ones(n_centers) * snr
    elif type(noise_heterogeneity) in [list, np.array]:
        assert snr == 3, "Option snr is incompatible with noise_heterogeneity as a list."
        snr_locs = np.array(noise_heterogeneity)
    else:
        raise ValueError(
            "Incorrect value noise_heterogeneity. It must be either a list of signal to noise ratio, or None for a constant signal to noise ratio."
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

    for i in range(n_centers):
        X = rng.multivariate_normal(
            mean=features_locs[i], cov=cov, size=n_samples_locs[i]
        )

        # generate labels
        signal = np.dot(X, w)
        noise_stdev = np.sqrt(np.mean(signal**2) / snr_locs[i])
        noise = rng.normal(scale=noise_stdev, size=n_samples_locs[i])
        y = signal + noise

        Xy = np.hstack((X, y[:, None]))
        df = pd.DataFrame(Xy)

        df_full = pd.concat((df_full, df), ignore_index=True)

    return df_full, indices


if __name__ == "__main__":
    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=100, seed=42, snr=3,
        features_heterogeneity=3.0,
        sample_repartition=2.0
    )

    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=100, seed=42, snr=3,
        features_heterogeneity=3.0,
        sample_repartition=[3.0, np.pi, np.pi**2/6, 42, 1, 0]
    )

    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=100, seed=42, snr=3,
        features_heterogeneity=3.0,
        sample_repartition=None
    )

    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=100, seed=42, snr=3,
        features_heterogeneity=3.0,
        sample_repartition=2.0,
        noise_heterogeneity=[3.0, np.pi, np.pi**2/6, 42, 1, 9]
    )

    df_full, indices = generate_synthetic_dataset(
        n_centers=6, n_samples=100, seed=42, snr=3,
        sample_repartition=2.0,
        noise_heterogeneity=[3.0, np.pi, np.pi**2/6, 42, 1, 9]
    )
    print(type(df_full))

    print((df_full))
    print([len(i) for i in indices])
    print(type(indices[0]))
