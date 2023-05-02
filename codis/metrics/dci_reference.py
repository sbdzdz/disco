class DCIMetrics:
    def __init__(
        self, data, n_factors, regressor="lasso", regressoR_coeffkwargs=None, infer=True
    ):
        kwargs = {"cv": 5, "selection": "random", "alphas": [0.02]}

        if regressoR_coeffkwargs is not None:
            kwargs.update(regressoR_coeffkwargs)

        if regressor == "lasso":
            regressor = LassoCV(**kwargs)
        elif regressor == "random-forest":
            regressor = RandomForestRegressor(**kwargs)
        else:
            raise ValueError()

        self.data = data
        self.n_factors = n_factors
        self.regressor = regressor

    def _get_regressoR_coeffscores(self, X, y):
        """
        Compute R_coeff{dk} for each code dimension D and each
        generative factor K
        """
        R = []

        for k in range(self.n_factors):
            y_k = y[:, k]
            if len(np.unique(y_k)) > 1:
                self.regressor.fit(X, y_k)
                R.append(np.abs(self.regressor.coef_))
            else:
                R.append(np.zeros(10))

        return np.stack(R).T

    def _disentanglement(self, R_coeff):
        """

        Disentanglement score as in Eastwood et al., 2018.
        """

        # Normalizing factors wrt to generative factors for each latent var
        sums_k = R_coeff.sum(axis=1, keepdims=True) + EPS
        weights = (sums_k / sums_k.sum()).squeeze()

        # Compute probabilities and entropy
        probs = R_coeff / sums_k
        log_probs = np.log(probs + EPS) / np.log(self.n_factors)
        entropy = -(probs * log_probs).sum(axis=1)

        # Compute scores
        di = 1 - entropy
        total_di = (di * weights).sum()

        return di, total_di

    def _completness(self, R_coeff):
        """
        Completness score as in Eastwood et al., 2018
        """

        # Normalizing factors along each latent for each generative factor
        sums_d = R_coeff.sum(axis=0) + EPS

        # Probabilities and entropy
        probs = R_coeff / sums_d
        log_probs = np.log(probs + EPS) / np.log(R_coeff.shape[0])
        entropy = -(probs * log_probs).sum(axis=0)

        # return completness scores
        return 1 - entropy

    def _informativeness(self, z_p, z):
        if isinstance(self.regressor, LassoCV):
            regressor = MultiTaskLassoCV(
                cv=self.regressor.cv, max_iter=2000, selection="random"
            )

        regressor.fit(z_p, z)
        return self.regressor.score(z_p)

    def compute_score(self, model, model_zs=None):
        if model_zs is None:
            X, y = infer(model, self.data)
            X, y = X.numpy(), y.numpy()
        else:
            X, y = model_zs

        X = X - X.mean(axis=0)  # / (X.std(axis=0) + EPS)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + EPS)

        R_coeff = self._get_regressoR_coeffscores(X, y)

        # compute metrics
        d_scores, total_d_score = self._disentanglement(R_coeff)
        c_scores = self._completness(R_coeff)
        # info_score = self._informativeness(X, y)

        return DCIResults(R_coeff, d_scores, total_d_score, c_scores)

    def __call__(self, model, model_zs=None):
        return self.compute_score(model, model_zs)


def compute_dci_metrics(models, model_names, factors, data):
    """
    Convenience function to compute the DCI metrics for a set of models
    in a given dataset.
    """
    n_factors = data.n_gen_factors

    loader = DataLoader(data, batch_size=64, num_workers=4, pin_memory=True)

    eastwood = DCIMetrics(loader, n_factors=n_factors)

    results = [eastwood(vae) for vae in models]

    return results
