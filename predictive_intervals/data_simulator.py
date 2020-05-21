import numpy as np

from typing import Dict


class DataSimulator:
    """Class encapsulating process of simulating data from a linear model.
    Might be expanded in the future by other data generating processes.

    """

    @staticmethod
    def generate_lm_data(
        *,
        n_points: int = 10_000,
        pct_train: float = 0.5,
        alpha: float = 1.0,
        beta: float = 2.0,
        sigma: float = np.sqrt(3.0),
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Method to generate data according to following linear model:
            y = alpha + beta * x + N(0, sigma^2)
        Once data is generated, split it into training set and test set.

        Parameters
        ----------
        n_points: int, default is 10_000
            Number of points we want to generate
        pct_train: float, default is 0.5, i.e. 50%
            Percentage of data points to be allocated to the training set
        alpha: float, optional, default = 1.
            Intercept parameter of the linear model
        beta: float, optional, default = 2.
            Explanatory variable parameter of the linear model
        sigma: float, optional, default = sqrt(3.)
            Standard deviation of the normal error

        Returns
        -------
        data: Dict[str, Dict[str, np.ndarray]]
            Dictionary containing dictionary training and test data
        """
        assert 0 < pct_train < 1.0, "`pct_train` needs to be between 0. and 1.!"

        # Sample data/realization
        x = np.random.randn(n_points)
        y = alpha + beta * x + sigma * np.random.randn(n_points)

        # Split data into training and test set
        n_train = int(pct_train * n_points)

        # Create training data and test data
        y_train = y[:n_train]
        x_train = x[:n_train]

        y_test = y[n_train:]
        x_test = x[n_train:]

        # Stack columns together
        data_train = {"y": y_train, "x": x_train}
        data_test = {"y": y_test, "x": x_test}

        data = {"train": data_train, "test": data_test}
        return data
