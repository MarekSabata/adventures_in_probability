import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from typing import Dict, Optional, NoReturn, Tuple
from predictive_intervals.models.predictive_intervals_model import PredictiveIntervalModel


class QuantileRegression(PredictiveIntervalModel):
    """Class encapsulating estimation of predictive intervals using quantile regression

    """

    def __init__(
        self,
        params_lb: Optional[np.ndarray] = None,
        params_ub: Optional[np.ndarray] = None,
        alpha: Optional[float] = 0.1,
    ):
        """
        Parameters
        ----------
        params_lb: Optional np.ndarray or None
            Parameters of the lower bound quantile regression, i.e. alpha level
        params_ub: Optional np.ndarray or None
            Parameters of the upper bound quantile regression, i.e. 1-alpha level
        alpha: float in (0,1), optional, default value = 0.1
            Quantile we want to estimate. Regression will esimate alpha/2-th and (1-alpha/2)-th quantile
            to create predictive interval at 1-alpha level
        """
        super().__init__(alpha=alpha)
        self.params_lb = params_lb
        self.params_ub = params_ub

    def fit(self, data: Dict[str, np.ndarray]) -> NoReturn:
        """Method that fits lower (1-alpha/2) and upper (alpha/2) quantile regression based on the
        provided data.

        Parameters
        ----------
        data: Dictionary
            Data dictionary with keys `x` and `y` corresponding to endogenous and exogenous variables.
        """
        assert "x" and "y" in data.keys(), "Data dictionary must have `x` and `y` keys!"

        # Specify quantile regression model
        mod = smf.quantreg("y ~ x", pd.DataFrame(data))

        # Fit the lower and upper quantile regression
        res_lb = mod.fit(q=self.alpha / 2)
        res_ub = mod.fit(q=(1 - self.alpha / 2))

        # Save the fitted parameters into self
        self.params_lb = res_lb.params.values
        self.params_ub = res_ub.params.values

    def predict(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Method that predicts

        Parameters
        ----------
        data: Dictionary
            Data dictionary with keys `x` and `y` corresponding to endogenous and exogenous variables.

        Returns
        -------
        y_lb, y_up: Tuple[np.ndarray, np.ndarray]
            Tuple of numpy array's for lower and upper bound predictions for y
        """
        # Assert that `x` and `y` are in the data dict keys
        assert "x" and "y" in data.keys(), "Data dictionary must have `x` and `y` keys!"

        # Assert that x and y have correct lengths
        assert len(data["x"]) == len(
            data["y"]
        ), "`x` data and `y` data must be of the same length!"

        # Extract the new x and y data (have to add intercept to the `x` data)
        x_new = np.vstack([np.ones(len(data["x"])), data["x"]])

        # Predict the lower and upper bound for y
        y_lb = np.dot(self.params_lb, x_new)
        y_ub = np.dot(self.params_ub, x_new)

        return y_lb, y_ub

    def get_predictive_intervals(
        self, data: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Method that encapsulates process of creating predictive intervals based on train and
        test data provided in the data dictionary.

        Parameters
        ----------
        data: Dict
            Data dictionary containing train and test data under `train` and `test` keys.

        Returns
        -------
        pred_ints: np.ndarray
            2-d numpy array with the dimensions being lower and upper bounds of the created predictive intervals
        """
        # Check if we already have model parameters, if not, fit the models
        if self.params_lb is None or self.params_ub is None:
            self.fit(data=data["train"])

        # Get lower and upper bound of the intervals
        y_lb, y_ub = self.predict(data["test"])

        # Calculate hit ratio
        y_real = data["test"]["y"]
        self.hit_ratio = np.mean((y_real >= y_lb) & (y_real <= y_ub))

        # Calculate average interval length
        self.avg_length = np.mean(y_ub - y_lb)

        # Print stats
        self.print_stats()

        # Stack the lower and upper bound to create predictive intervals
        pred_ints = np.vstack([y_lb, y_ub])

        return pred_ints

    def __repr__(self) -> str:
        return "Quantile Regression"
