"""Disentanglement, Completeness and Informativeness score from Eastwood et al., 2018."""
from collections import namedtuple

from sklearn.ensemble import RandomForestRegressor

DCIScore = namedtuple(
    "DCIScore", ["disentanglement", "completeness", "informativeness"]
)


class DCIMetric:
    """Class calculating DCI score"""

    def __init__(self, data, n_factors):
        self.data = data
        self.n_factors = n_factors
        self.regressor = RandomForestRegressor()

    def compute(self, latents, factors) -> DCIScore:
        """Calculate the DCI score.
        Args:
            latents: Latent representation of the data.
            factors: Ground truth factors of the data.
        Returns:
            dci_score: DCI score.
        """
        coefficients = self._get_regression_coefficients(latents, factors)
        disenanglement = self._compute_disentanglement(coefficients)
        completeness = self._compute_completeness(coefficients)
        informativeness = self._compute_informativeness(coefficients)
        return DCIScore(disenanglement, completeness, informativeness)

    def _get_regression_coefficients(self, latents, factors):
        """Compute regression coefficients of predicting factors from latents."""
        raise NotImplementedError

    def _compute_disentanglement(self, coefficients):
        raise NotImplementedError

    def _compute_completeness(self, coefficients):
        raise NotImplementedError

    def _compute_informativeness(self, coefficients):
        raise NotImplementedError
