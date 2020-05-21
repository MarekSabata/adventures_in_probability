import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, NoReturn, Optional


class PredictiveIntervalModel(ABC):
    """Abstract class encapsulating methods related to child classes that implement
    different ways of getting predictive intervals.
    
    """

    def __init__(self, alpha: Optional[float] = 0.1, verbose: bool = False):
        """
        Parameters
        ----------
        alpha: float, in interval (0,1)
            Probability level at which we want the predictive interval to be, i.e. alpha = 0.1 corresponds
            to 90% probability interval.
        verbose: bool
            Print out statistics once done if set to True, else it won't print them out.
            
        """
        self.check_alpha(alpha=alpha)
        self.alpha = alpha
        self.verbose = verbose

        # Store statistics from the latest experiment
        self.hit_ratio = None
        self.avg_length = None

    @abstractmethod
    def get_predictive_intervals(
        self, data: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """ Method for that encapsulates the creation and evaluation of predictive interval.
        Should print out statistics
        
        Parameters
        ----------
        data: Dictionary
            Data dictionary with keys 'train' and 'test' corresponding to data we want to use for
            training the model and for testing, aka creating predictive intervals for
            
        Returns
        -------
        pred_ints: np.ndarray
            2-d numpy array with the dimensions being lower and upper bounds of the created predictive intervals
        """
        pass

    @staticmethod
    def check_alpha(alpha: float) -> bool:
        """Method for asserting alpha is a float between 0 and 1
        
        Parameters
        ----------
        alpha: float
            Level 
            
        Returns
        -------
        bool
            True if alpha is between 0 and 1
            
        Raises
        ------
        Exception if alpha is outside of (0, 1) interval or not float
        """

        if isinstance(alpha, float) and (0 < alpha < 1.0):
            return True
        else:
            raise Exception("`alpha` must be float between 0. and 1.!")

    def print_stats(self) -> NoReturn:
        """Method for printing statistics for the created predictive intervals. Prints out the hit ratio and
        average length of the predictive interval.

        """
        if self.verbose and (self.hit_ratio and self.avg_length is not None):
            print(
                "The predictive intervals achieved hit ratio of {:.2f}% and have average length of {:.3f}.".format(
                    100 * self.hit_ratio, self.avg_length
                )
            )

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns
        -------
            String description of the model.
        """
        pass
