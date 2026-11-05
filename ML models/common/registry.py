"""The command-line key of each model type mapped to its class and folder.

To add a model: create a folder with a model.py holding a class with the shared
fit/predict interface and a unique `name`, then add it here.
"""
from mlp.model import MLPModel
from random_forest.model import RandomForestModel
from ridge_polynomial.model import RidgePolynomialModel
from xgboost_regressor.model import XGBoostModel

MODEL_TYPES = {
    cls.name: cls
    for cls in (MLPModel, XGBoostModel, RandomForestModel, RidgePolynomialModel)
}

# Each model's folder name, which is where its train.py writes results/ and saved_models/.
MODEL_FOLDERS = {
    "mlp": "mlp",
    "xgb": "xgboost_regressor",
    "rf": "random_forest",
    "ridge": "ridge_polynomial",
}
