"""Fit histogram gradient boosting and export predictions as safe JSON trees."""

import numpy as np


def fit_tree(x, y, seed, max_iter, learning_rate, max_leaf_nodes):
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ValueError(
            "Tree training requires scikit-learn; install chemsmart-ml[trees]"
        ) from exc
    if (
        max_iter < 1
        or max_leaf_nodes < 2
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid tree parameters")
    estimator = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        early_stopping=False,
    )
    estimator.fit(x, y)
    trees = []
    for stage in estimator._predictors:
        nodes = stage[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("Categorical splits are not supported in JSON export")
        trees.append(
            [
                [
                    int(n["feature_idx"]),
                    float(n["num_threshold"]),
                    int(n["left"]),
                    int(n["right"]),
                    float(n["value"]),
                    bool(n["is_leaf"]),
                ]
                for n in nodes
            ]
        )
    artifact = dict(
        model="hist-gradient-boosting-delta",
        baseline=float(estimator._baseline_prediction[0, 0]),
        trees=trees,
        scikit_learn_version=sklearn.__version__,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
    )
    check = x[: min(len(x), 1024)]
    if not np.allclose(
        predict_tree(check, artifact), estimator.predict(check), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Tree export does not match scikit-learn predictions")
    return artifact


def predict_tree(x, model):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Tree inputs must be finite")
    result = np.full(len(x), float(model["baseline"]))
    for tree in model["trees"]:
        nodes = np.asarray(tree, dtype=float)
        ids = np.zeros(len(x), dtype=int)
        while True:
            active = nodes[ids, 5] == 0
            if not active.any():
                break
            ix = np.flatnonzero(active)
            selected = nodes[ids[ix]]
            features = selected[:, 0].astype(int)
            go_left = x[ix, features] <= selected[:, 1]
            ids[ix] = np.where(go_left, selected[:, 2], selected[:, 3]).astype(int)
        result += nodes[ids, 4]
    return result
