import numpy as np
import pystan as ps

from typing import Optional, Dict
from predictive_intervals.models.predictive_intervals_model import PredictiveIntervalModel


class BayesianRegression(PredictiveIntervalModel):
    """Bayesian regression implemented in PyStan

    """

    def __init__(self, alpha: Optional[float] = 0.1):
        """
        Parameters
        ----------
        alpha: float
        """
        super().__init__(alpha=alpha)

    @staticmethod
    def get_stan_model() -> str:
        """
        Returns
        -------
        model: str
            String specification of Stan model
        """

        model = """
        data {
            int<lower=0> N;
            int<lower=0> N_test;
            vector[N] x_train;
            vector[N] y_train;
            vector[N] x_test;
        }
        parameters {
            real alpha;
            real beta;
            real<lower=0> sigma;
            vector[N_test] y_hat;
        }
        model {
            y_train ~ normal(alpha + beta * x_train, sigma);
            y_hat ~ normal(alpha + beta * x_test, sigma);
        }

        """
        return model

    def get_predictive_intervals(self, data: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """

        Notes
        -----
        Implementation inspired by the following article:
        https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53

        and the following pystan tutorial:
        https://mc-stan.org/docs/2_21/stan-users-guide/prediction-forecasting-and-backcasting.html
        """
        # Specify Stan model
        model = self.get_stan_model()

        # Put model data into dictionary
        data_stan = {'N': len(data['train']['x']),
                     'x_train': data['train']['x'],
                     'y_train': data['train']['y'],
                     'N_test': len(data['test']['x']),
                     'x_test': data['test']['x']}

        # Compile PyStan model
        sm = ps.StanModel(model_code=model)

        # Fit the model and generate fitted predictions
        fit = sm.sampling(data=data_stan, iter=1000, chains=4, warmup=500, thin=1, seed=101).to_dataframe()

        # Extract real y, prepare arrays for lower and upper bound
        y_real = data['test']['y']
        y_lb = np.zeros_like(y_real)
        y_ub = np.zeros_like(y_real)

        # Loop over the generated samples and extract their distributional property
        for i in range(data_stan['N_test']):
            y_hat_name = "y_hat[{}]".format(i + 1)
            y_lb[i] = np.quantile(fit[y_hat_name], self.alpha / 2)
            y_ub[i] = np.quantile(fit[y_hat_name], 1 - self.alpha / 2)

        # Calculate hit ratio
        self.hit_ratio = np.mean((y_real >= y_lb) & (y_real <= y_ub))

        # Calculate average interval length
        self.avg_length = np.mean(y_ub - y_lb)

        # Print stats
        self.print_stats()

        # Stack the lower and upper bound to create predictive intervals
        pred_ints = np.vstack([y_lb, y_ub])

        return pred_ints

    def __repr__(self) -> str:
        return "Bayesian Regression"
