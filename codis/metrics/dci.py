"""Disentanglement, Completeness and Informativeness score from Eastwood et al., 2018."""


class DCI:
    """DCI score"""

    def __init__(self, data, n_factors):
        raise NotImplementedError

    def compute(self):
        """Calculate the DCI score."""
        raise NotImplementedError

    def _compute_disenablement(self):
        raise NotImplementedError

    def _compute_completeness(self):
        raise NotImplementedError

    def _compute_informativeness(self):
        raise NotImplementedError
