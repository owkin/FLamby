import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_centers, n_samples, n_features, seed=42, snr=3):

    rng = np.random.default_rng(seed=42)

    if isinstance(n_samples, int):
        n_samples = [n_samples for i in range(n_centers)]

    # generate the true coefficients
    w = rng.lognormal(sigma=1, size=n_features)

    # generate the data for each center
    df_full = pd.DataFrame()
    cov = np.eye(n_features)

    indices = []

    start_index = 0
    for n_samples_loc in n_samples:
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
        indices.append(np.arange(start_index, start_index + n_samples_loc))

        start_index += n_samples_loc

    return df_full, indices
