import numpy as np
import statsmodels.api as sm

from typing import Dict, Optional, NoReturn
from predictive_intervals.models.predictive_intervals_model import PredictiveIntervalModel


class JacknifePlus(PredictiveIntervalModel):
    """Class encapsulating implementation of the Jacknife+ method
    Jacknife+ paper on arxiv: https://arxiv.org/pdf/1905.02928.pdf

    """

    def __init__(self, alpha: Optional[float] = 0.1):
        """
        Parameters
        ----------
        alpha: float in (0,1), optional, default value = 0.1
            Quantile we want to estimate. Regression will esimate alpha/2-th and (1-alpha/2)-th quantile
            to create predictive interval at 1-alpha level
        """

        super().__init__(alpha=alpha)

        # Attributes for storing residuals and fitted models
        self.residuals = None
        self.fitted_lms = None

    @staticmethod
    def fit_linear_model(data: Dict[str, np.ndarray]):
        """
        Method that fits linear model based on the provided data.

        Parameters
        ----------
        data: Dict[str, np.ndarray]
            Data dictionary with keys `x` and `y` corresponding to exogenous and endogenous variable, respectively

        Returns
        -------
        lm_fit:
        """
        lm = sm.OLS(data['y'], sm.add_constant(data['x']), has_intercept=True)
        lm_fit = lm.fit()
        return lm_fit

    def fit(self, data: Dict[str, np.ndarray]) -> NoReturn:
        """
        Method that fits linear model on every leave-one-out subset of the data and collects
        residuals for the leave-one-out point based on the trained model.

        Parameters
        ----------
        data: Dict[str, np.ndarray]
        """
        # Extract training data
        y_train = data['y']
        x_train = data['x']

        # Get number of data points in the data
        n = len(y_train)

        # Store residuals and fitted models
        res = np.zeros(n)
        fitted_lms = []

        for i in range(n):
            # Single out the point we're not using in this estimation
            y_out = y_train[i]
            if x_train.ndim == 1:
                x_out = x_train[i]
            else:
                x_out = x_train[i, :]

            # Estimate linear regression leaving out the point
            data = {'x': np.delete(x_train, i, axis=0),
                    'y': np.delete(y_train, i, axis=0)}
            lm_fit = self.fit_linear_model(data=data)

            # Estimate the point we left out using the fitted linear model
            y_hat = lm_fit.predict(exog=np.array((1, x_out)))

            # Calculate the residual
            res[i] = y_hat - y_out

            # Append fitted model
            fitted_lms.append(lm_fit)

        # Save to self
        self.residuals = res
        self.fitted_lms = fitted_lms

    def get_predictive_intervals(self,
                                 data: Dict[str, Dict[str, np.ndarray]],
                                 pct_sample: float = 1.) -> np.ndarray:
        """

        Parameters
        -----------
        data:
        pct_sample: float
            Percentage of data points to use when calculating predictive intervals using Jacknife+
        """
        assert 0. < pct_sample <= 1., "`pct_sample` has to be in the (0, 1] interval!"

        if self.fitted_lms or self.residuals is None:
            self.fit(data=data['train'])

        # Add constant term for intercept
        x_new = sm.add_constant(data['test']['x'])

        # Get number of points and number of fitted models
        n_points = int(pct_sample * len(x_new))
        n_models = len(self.fitted_lms)

        # Array for predictive intervals
        y_lb = np.zeros(n_points)
        y_ub = np.zeros(n_points)

        # Idx sample to subset data
        idx_data = np.random.choice(len(x_new), size=n_points, replace=False)

        # Loop over all points in the new data
        for i in range(n_points):

            # Dictionary with upper and lower bounds created by k-th model
            y_jacknife = {'lb': np.zeros_like(self.residuals),
                          'ub': np.zeros_like(self.residuals)}

            # Loop over all fitted models
            for k in range(n_models):
                # Get the jacknife+ point estimate using the k-th fitted model
                y_hat_k = self.fitted_lms[k].predict(exog=x_new[idx_data[i], :])

                # Construct the jacknife+ intervals
                y_jacknife['lb'][k] = y_hat_k - self.residuals[k]
                y_jacknife['ub'][k] = y_hat_k + self.residuals[k]

            # Calculate the quantile of the jacknife+ quantities
            y_lb[i] = np.quantile(y_jacknife['lb'], self.alpha / 2)
            y_ub[i] = np.quantile(y_jacknife['ub'], 1 - self.alpha / 2)

        # Calculate hit ratio
        y_real = data['test']['y'][idx_data]
        self.hit_ratio = np.mean((y_real >= y_lb) & (y_real <= y_ub))

        # Calculate average interval length
        self.avg_length = np.mean(y_ub - y_lb)

        # Print stats
        self.print_stats()

        # Stack the lower and upper bound to create predictive intervals
        pred_ints = np.vstack([y_lb, y_ub])

        return pred_ints

    def __repr__(self) -> str:
        return "Jacknife+ Regression"
