"""Portable nonlinear regressors without loading executable pickle artifacts."""

import numpy as np


def fit_nonlinear(
    kind,
    x,
    y,
    seed,
    hidden_size=128,
    max_iter=300,
    learning_rate=0.001,
    kernel_alpha=0.001,
    kernel_gamma=0.01,
    kernel_max_samples=2000,
):
    try:
        import sklearn
        from sklearn.neural_network import MLPRegressor
        from sklearn.kernel_ridge import KernelRidge
    except ImportError as exc:
        raise ValueError(
            "Install chemsmart-ml[nonlinear] for nonlinear models"
        ) from exc
    if kind == "mlp":
        # Scale residual targets on the training split for stable optimization.
        center, scale = float(np.mean(y)), float(np.std(y))
        if scale == 0:
            scale = 1.0
        estimator = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size),
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            random_state=seed,
            early_stopping=False,
        )
        estimator.fit(x, (y - center) / scale)
        artifact = dict(
            kind="mlp",
            coefs=[a.tolist() for a in estimator.coefs_],
            intercepts=[a.tolist() for a in estimator.intercepts_],
            target_mean=center,
            target_scale=scale,
            hidden_size=hidden_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            iterations=int(estimator.n_iter_),
        )
        expected = estimator.predict(x[:128]) * scale + center
    elif kind == "kernel-ridge":
        if kernel_max_samples < 1 or len(x) > kernel_max_samples:
            raise ValueError(
                f"Kernel ridge has {len(x)} training samples; limit is {kernel_max_samples}. "
                "Use a smaller validated dataset; kernel memory scales quadratically."
            )
        estimator = KernelRidge(alpha=kernel_alpha, kernel="rbf", gamma=kernel_gamma)
        estimator.fit(x, y)
        artifact = dict(
            kind="kernel-ridge",
            support_vectors=x.tolist(),
            dual_coef=estimator.dual_coef_.tolist(),
            alpha=kernel_alpha,
            gamma=kernel_gamma,
            max_samples=kernel_max_samples,
        )
        expected = estimator.predict(x[:128])
    else:
        raise ValueError("Unknown nonlinear model")
    artifact["sklearn_version"] = sklearn.__version__
    if not np.allclose(
        predict_nonlinear(x[:128], artifact), expected, rtol=1e-8, atol=1e-8
    ):
        raise ValueError("Nonlinear model export verification failed")
    return artifact


def predict_nonlinear(x, artifact):
    x = np.asarray(x, dtype=float)
    if artifact["kind"] == "mlp":
        z = x
        coefs, intercepts = artifact["coefs"], artifact["intercepts"]
        for i, (coef, bias) in enumerate(zip(coefs, intercepts)):
            z = z @ np.asarray(coef) + np.asarray(bias)
            if i < len(coefs) - 1:
                z = np.maximum(z, 0)
        return z.ravel() * artifact["target_scale"] + artifact["target_mean"]
    if artifact["kind"] == "kernel-ridge":
        vectors = np.asarray(artifact["support_vectors"])
        dual = np.asarray(artifact["dual_coef"])
        # Bound the temporary kernel matrix on inference.
        values = []
        for start in range(0, len(x), 512):
            block = x[start : start + 512]
            dist = np.maximum(
                np.sum(block * block, axis=1)[:, None]
                + np.sum(vectors * vectors, axis=1)[None, :]
                - 2 * block @ vectors.T,
                0,
            )
            values.append(np.exp(-artifact["gamma"] * dist) @ dual)
        return np.concatenate(values)
    raise ValueError("Unsupported nonlinear model")
