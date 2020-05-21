import numpy as np
import statsmodels.api as sm

from typing import Optional, NoReturn, Dict, Tuple
from predictive_intervals.models.predictive_intervals_model import PredictiveIntervalModel


class LinearModel(PredictiveIntervalModel):
    """
    Class encapsulating methods for related to linear model

        y = alpha + beta_1 * x_1 + ... + beta_p * x_p + eps

    where eps comes from Normal distribution with zero mean and fixed variance sigma.
    """

    def __init__(
        self, params: Optional[np.ndarray] = None, alpha: Optional[float] = 0.1
    ):
        """
        Parameters
        ----------
        alpha: float, in interval (0,1)
            Probability level at which we want the predictive interval to be, i.e. alpha = 0.1 corresponds
            to 90% probability interval.
        params: np.ndarray
            Parameters of the linear model we want to use,
        """
        # Init super
        super().__init__(alpha=alpha)

        # Set parameters if some were provided
        self.params = params

    def fit(self, data: Dict[str, np.ndarray]) -> NoReturn:
        """
        Method that fits linear model based on the provided data.

        Parameters
        ----------
        data: Dict[str, np.ndarray]
            Data dictionary with keys `x` and `y` representing exogenous and engenous data, respectively
        """
        lm = sm.OLS(
            endog=data["y"], exog=sm.add_constant(data["x"]), has_intercept=True
        )
        lm_fit = lm.fit()
        self.params = lm_fit.params

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """
        Method that predicts new values of the target variable using the provided data and fitted model.

        Parameters
        ----------
        x_new: np.ndarray
            Array with new data we want to create prediction for

        Returns
        -------
        y_fit: np.ndarray
            Array with predicted values based on x_new
        """
        assert (
            self.params is not None
        ), "Model has not been fitted yet! Run `.fit` method first."
        assert (
            x_new.ndim == len(self.params) - 1
        ), "Dimension of new data does not correspond to fitted parameters!"

        # Create y_fit
        if x_new.ndim == 1:
            y_fit = self.params[0] + self.params[1] * x_new
        else:
            y_fit = self.params[0] + np.dot(
                np.expand_dims(self.params[:1], 0), x_new.transpose()
            )

        # Assert y_fit has the same length as the input data
        assert len(y_fit) == len(
            x_new
        ), "Wrong predict calculation, lengths don't match!"

        return y_fit

    def predict_with_residuals(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Method that predicts new values of the target variable and also calculates respective residuals
        based on the actual response variable values.

        Parameters
        ----------
        data: Dict
            Dictionary with new regressor data we want to use to create predictions, should contain keys `x` and `y`

        Returns
        -------
        y_hat, res: Tuple[np.ndarray, np.ndarray]
            Fitted values of y and residuals w.r.t. to actual y
        """
        # Assert that `x` and `y` are in the data dict keys
        assert "x" and "y" in data.keys(), "Data dictionary must have `x` and `y` keys!"

        # Assert that x and y have correct lengths
        assert len(data["x"]) == len(
            data["y"]
        ), "`x` and `y` data must have the same length!"

        # Extract the new x and y data
        x_new = data["x"]
        y_new = data["y"]

        # Predict the new y
        y_hat = self.predict(x_new=x_new)

        # Calculate residuals
        res = y_hat - y_new

        return y_hat, res

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
        # Fit the model if no parameters are available
        if self.params is None:
            self.fit(data=data["train"])

        # Create new predictions
        y_hat, res = self.predict_with_residuals(data=data["test"])

        # Get (1-alpha)-th quantile of absolute values of residuals
        q_alpha = np.quantile(np.abs(res), 1 - self.alpha)

        # Create lower and upper bound of the predictive interval
        y_lb = y_hat - q_alpha
        y_ub = y_hat + q_alpha

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
        return "Linear Regression"
