import json

import numpy as np
import pytest

from chemsmart_ml.models import (
    DeltaModel,
    GradientBoostingTrees,
    LinearDeltaModel,
    RBFKernelRidge,
    RidgeDeltaModel,
)

pytest.importorskip("sklearn")


def test_tree_json_matches_sklearn_and_non_linearity():
    x = np.linspace(-2, 2, 300)[:, None]
    y = x[:, 0] ** 2
    trees = GradientBoostingTrees(60, 0.1, 15, seed=42).fit(x, y)
    restored = GradientBoostingTrees.from_dict(
        json.loads(json.dumps(trees.to_dict()))
    )
    assert np.mean(abs(restored.predict(x) - y)) < 0.15


def test_kernel_limit():
    x = np.arange(30).reshape(15, 2).astype(float)
    with pytest.raises(ValueError, match="memory scales quadratically"):
        RBFKernelRidge(max_samples=10).fit(x, np.arange(15))


def test_ridge_penalises_slope_not_intercept():
    design = np.column_stack([np.ones(50), np.linspace(-1, 1, 50)])
    delta = 3 + 2 * design[:, 1]
    rows = np.arange(50)
    ols = LinearDeltaModel().fit(design, delta, rows)
    ridge = RidgeDeltaModel(alpha=10.0).fit(design, delta, rows)
    np.testing.assert_allclose(ols.coefficients, [3, 2])
    assert ridge.coefficients[0] == pytest.approx(3)
    assert 0 < ridge.coefficients[1] < 2


def test_unsupported_artifact_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model format"):
        DeltaModel.from_artifact(dict(model="random-forest-delta"))
