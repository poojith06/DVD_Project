"""The command-line key of each model type mapped to its class and folder.

To add a model: create a folder with a model.py holding a class with the shared
interface and a unique `name`, then add it here.

Shared interface (X = scaled features, df = the raw rows, y = log10|I| of id, ig, is, ib):
    Model(case, seed, params)
    fit(X_train, y_train, X_val, y_val, df_train, df_val) -> self
    predict(X, df) -> array (rows, 4) of log10|I|
    predict_signed(X, df) -> array (rows, 4) of signed amps   (optional; then no sign model is used)
"""
from components_mlp.model import ComponentsMLPModel
from gpr.model import GPRModel
from hybrid_analytic_mlp.model import HybridAnalyticMLPModel
from resnet_mlp.model import ResNetMLPModel

MODEL_TYPES = {
    cls.name: cls
    for cls in (ComponentsMLPModel, GPRModel, HybridAnalyticMLPModel, ResNetMLPModel)
}

# Each model's folder name, which is where its train.py writes results/ and saved_models/.
MODEL_FOLDERS = {
    "components_mlp": "components_mlp",
    "gpr": "gpr",
    "hybrid_analytic_mlp": "hybrid_analytic_mlp",
    "resnet_mlp": "resnet_mlp",
}
