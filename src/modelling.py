import joblib
import warnings
import numpy as np
import pandas as pd

from typing import List
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from logger import get_logger

warnings.filterwarnings("ignore")


"""
This module is used to train the models and save them in the models folder:
Here we are training both baseline models and fine-tuned models.
The hyperparameters have been taken from the phase 2 notebooks.
"""

score_data_baseline = pd.DataFrame(
    index=[
        "R2_Score",
        "Accuracy",
        "Mean_Squared_Error",
        "Mean_Absolute_Error",
        "Root_MSE"
    ]
)

score_data_tuned = pd.DataFrame(
    index=[
        "R2_Score",
        "Accuracy",
        "Mean_Squared_Error",
        "Mean_Absolute_Error",
        "Root_MSE"
    ]
)

logger = get_logger(module_name="modelling")


def save_trained_model_in_local(model, file_name_and_path: str) -> None:
    joblib.dump(model, file_name_and_path)


def get_train_test_split_data(
        df: pd.DataFrame,
        test_size=0.2,
        random_state=22
):
    features = df.loc[:, df.columns != "price"]
    target = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def get_result_metrics(y_test, y_pred) -> List[str]:
    results = [
        "%.5f" % float(r2_score(y_test, y_pred)),
        "%.5f" % float(r2_score(y_test, y_pred) * 100),
        "%.5f" % float(mean_squared_error(y_test, y_pred)),
        "%.5f" % float(mean_absolute_error(y_test, y_pred)),
        "%.5f" % float(np.sqrt(mean_squared_error(y_test, y_pred)))
    ]

    return results


def create_baseline_decision_tree_regressor(X_train, y_train, X_test, y_test) -> None:
    dt_baseline_model = DecisionTreeRegressor()
    dt_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=dt_baseline_model,
        file_name_and_path="models/baseline/decision_tree_regressor.pkl"
    )

    dt_baseline_pred = dt_baseline_model.predict(X_test)

    dt_baseline_scores = get_result_metrics(y_test, dt_baseline_pred)

    score_data_baseline["Decision Tree Regressor"] = dt_baseline_scores

    print("R2 score - Decision Tree Regression (Baseline):", dt_baseline_scores[0])
    print("Mean squared error - Decision Tree Regression (Baseline):", dt_baseline_scores[2])
    print("Mean absolute error - Decision Tree Regression (Baseline):", dt_baseline_scores[3])
    print("Root mean squared error - Decision Tree Regression (Baseline):", dt_baseline_scores[4])


def create_fine_tuned_decision_tree_regressor(X_train, y_train, X_test, y_test) -> None:
    """
    After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
    {
        "max_depth": None,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_samples_leaf": 5,
        "splitter": "best"
    }
    """
    dt_fine_tuned_model = DecisionTreeRegressor(
        max_depth=None,
        max_features=None,
        max_leaf_nodes=None,
        min_samples_leaf=5,
        splitter="best"
    )
    dt_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=dt_fine_tuned_model,
        file_name_and_path="models/fine_tuned/decision_tree_regressor.pkl"
    )

    dt_fine_tuned_pred = dt_fine_tuned_model.predict(X_test)

    dt_fine_tuned_scores = get_result_metrics(y_test, dt_fine_tuned_pred)

    score_data_tuned["Decision Tree Regressor"] = dt_fine_tuned_scores

    logger.info(f"\nR2 score - Decision Tree Regression (Fine Tuned): {dt_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - Decision Tree Regression (Fine Tuned): {dt_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - Decision Tree Regression (Fine Tuned): {dt_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - Decision Tree Regression (Fine Tuned): {dt_fine_tuned_scores[4]}")


def create_baseline_random_forest_regressor(X_train, y_train, X_test, y_test) -> None:
    rf_baseline_model = RandomForestRegressor()
    rf_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=rf_baseline_model,
        file_name_and_path="models/baseline/random_forest_regressor.pkl"
    )

    rf_baseline_pred = rf_baseline_model.predict(X_test)

    rf_baseline_scores = get_result_metrics(y_test, rf_baseline_pred)

    score_data_baseline["Random Forest Regressor"] = rf_baseline_scores

    logger.info(f"\nR2 score - Random Forest Regression (Baseline): {rf_baseline_scores[0]}")
    logger.info(f"Mean squared error - Random Forest Regression (Baseline): {rf_baseline_scores[2]}")
    logger.info(f"Mean absolute error - Random Forest Regression (Baseline): {rf_baseline_scores[3]}")
    logger.info(f"Root mean squared error - Random Forest Regression (Baseline): {rf_baseline_scores[4]}")


def create_fine_tuned_random_forest_regressor(X_train, y_train, X_test, y_test) -> None:
    """
        After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
            "max_depth": None,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_samples_leaf": 1,
            "n_estimators": 150,
            "oob_score": True
        }
    """
    rf_fine_tuned_model = RandomForestRegressor(
        max_depth=None,
        max_leaf_nodes=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_estimators=150,
        oob_score=True
    )
    rf_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=rf_fine_tuned_model,
        file_name_and_path="models/fine_tuned/random_forest_regressor.pkl"
    )

    rf_fine_tuned_pred = rf_fine_tuned_model.predict(X_test)

    rf_fine_tuned_scores = get_result_metrics(y_test, rf_fine_tuned_pred)

    score_data_tuned["Random Forest Regressor"] = rf_fine_tuned_scores

    logger.info(f"\nR2 score - Random Forest Regression (Fine Tuned): {rf_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - Random Forest Regression (Fine Tuned): {rf_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - Random Forest Regression (Fine Tuned): {rf_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - Random Forest Regression (Fine Tuned): {rf_fine_tuned_scores[4]}")


def create_baseline_knn_regressor(X_train, y_train, X_test, y_test) -> None:
    knn_baseline_model = KNeighborsRegressor(n_neighbors=5)
    knn_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=knn_baseline_model,
        file_name_and_path="models/baseline/knn_regressor.pkl"
    )

    knn_baseline_pred = knn_baseline_model.predict(X_test)

    knn_baseline_scores = get_result_metrics(y_test, knn_baseline_pred)

    score_data_baseline["KNN Regressor"] = knn_baseline_scores

    logger.info(f"\nR2 score - KNN Regression (Baseline): {knn_baseline_scores[0]}")
    logger.info(f"Mean squared error - KNN Regression (Baseline): {knn_baseline_scores[2]}")
    logger.info(f"Mean absolute error - KNN Regression (Baseline): {knn_baseline_scores[3]}")
    logger.info(f"Root mean squared error - KNN Regression (Baseline): {knn_baseline_scores[4]}")


def create_fine_tuned_knn_regressor(X_train, y_train, X_test, y_test) -> None:
    """
        After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
          "algorithm": "brute",
          "n_neighbors": 15,
          "p": 1,
          "weights": "distance"
        }
    """
    knn_fine_tuned_model = KNeighborsRegressor(
        algorithm="brute",
        n_neighbors=15,
        p=1,
        weights="distance"
    )
    knn_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=knn_fine_tuned_model,
        file_name_and_path="models/fine_tuned/knn_regressor.pkl"
    )

    knn_fine_tuned_pred = knn_fine_tuned_model.predict(X_test)

    knn_fine_tuned_scores = get_result_metrics(y_test, knn_fine_tuned_pred)

    score_data_tuned["KNN Regressor"] = knn_fine_tuned_scores

    logger.info(f"\nR2 score - KNN Regression (Fine Tuned): {knn_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - KNN Regression (Fine Tuned): {knn_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - KNN Regression (Fine Tuned): {knn_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - KNN Regression (Fine Tuned): {knn_fine_tuned_scores[4]}")


def create_baseline_xg_boost_regressor(X_train, y_train, X_test, y_test) -> None:
    xgb_baseline_model = XGBRegressor()
    xgb_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=xgb_baseline_model,
        file_name_and_path="models/baseline/xgb_regressor.pkl"
    )

    xgb_baseline_pred = xgb_baseline_model.predict(X_test)

    xgb_baseline_scores = get_result_metrics(y_test, xgb_baseline_pred)

    score_data_baseline["XG Boost Regressor"] = xgb_baseline_scores

    logger.info(f"\nR2 score - XG Boost Regression (Baseline): {xgb_baseline_scores[0]}")
    logger.info(f"Mean squared error - XG Boost Regression (Baseline): {xgb_baseline_scores[2]}")
    logger.info(f"Mean absolute error - XG Boost Regression (Baseline): {xgb_baseline_scores[3]}")
    logger.info(f"Root mean squared error - XG Boost Regression (Baseline): {xgb_baseline_scores[4]}")


def create_fine_tuned_xg_boost_regressor(X_train, y_train, X_test, y_test) -> None:
    """
    After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
            "alpha": 0,
            "learning_rate": 0.3,
            "max_depth": 9,
            "n_estimators": 150,
            "objective": "reg:squarederror"
        }
    """
    xgb_fine_tuned_model = XGBRegressor(
        alpha=0,
        learning_rate=0.3,
        max_depth=9,
        objective="reg:squarederror",
        n_estimators=150
    )
    xgb_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=xgb_fine_tuned_model,
        file_name_and_path="models/fine_tuned/xgb_regressor.pkl"
    )

    xgb_fine_tuned_pred = xgb_fine_tuned_model.predict(X_test)

    xgb_fine_tuned_scores = get_result_metrics(y_test, xgb_fine_tuned_pred)

    score_data_tuned["XG Boost Regressor"] = xgb_fine_tuned_scores

    logger.info(f"\nR2 score - XG Boost Regression (Baseline): {xgb_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - XG Boost Regression (Baseline): {xgb_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - XG Boost Regression (Baseline): {xgb_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - XG Boost Regression (Baseline): {xgb_fine_tuned_scores[4]}")


def create_baseline_light_gbm_regressor(X_train, y_train, X_test, y_test) -> None:
    light_gbm_baseline_model = LGBMRegressor()
    light_gbm_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=light_gbm_baseline_model,
        file_name_and_path="models/baseline/light_gbm_regressor.pkl"
    )

    light_gbm_baseline_pred = light_gbm_baseline_model.predict(X_test)

    light_gbm_baseline_scores = get_result_metrics(y_test, light_gbm_baseline_pred)

    score_data_baseline["Light GBM Regressor"] = light_gbm_baseline_scores

    logger.info(f"\nR2 score - Light GBM Regression (Baseline): {light_gbm_baseline_scores[0]}")
    logger.info(f"Mean squared error - Light GBM Regression (Baseline): {light_gbm_baseline_scores[2]}")
    logger.info(f"Mean absolute error - Light GBM Regression (Baseline): {light_gbm_baseline_scores[3]}")
    logger.info(f"Root mean squared error - Light GBM Regression (Baseline): {light_gbm_baseline_scores[4]}")


def create_fine_tuned_light_gbm_regressor(X_train, y_train, X_test, y_test) -> None:
    """
        After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
            "learning_rate": 0.1,
            "max_depth": 9,
            "n_estimators": 125
        }
    """
    light_gbm_fine_tuned_model = LGBMRegressor(
        learning_rate=0.1,
        max_depth=9,
        n_estimators=125
    )
    light_gbm_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=light_gbm_fine_tuned_model,
        file_name_and_path="models/fine_tuned/light_gbm_regressor.pkl"
    )

    light_gbm_fine_tuned_pred = light_gbm_fine_tuned_model.predict(X_test)

    light_gbm_fine_tuned_scores = get_result_metrics(y_test, light_gbm_fine_tuned_pred)

    score_data_tuned["Light GBM Regressor"] = light_gbm_fine_tuned_scores

    logger.info(f"\nR2 score - Light GBM Regression (Fine Tuned): {light_gbm_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - Light GBM Regression (Fine Tuned): {light_gbm_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - Light GBM Regression (Fine Tuned): {light_gbm_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - Light GBM Regression (Fine Tuned): {light_gbm_fine_tuned_scores[4]}")


def create_baseline_lasso_regressor(X_train, y_train, X_test, y_test) -> None:
    lasso_baseline_model = Lasso()
    lasso_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=lasso_baseline_model,
        file_name_and_path="models/baseline/lasso_regressor.pkl"
    )

    lasso_baseline_pred = lasso_baseline_model.predict(X_test)

    lasso_baseline_scores = get_result_metrics(y_test, lasso_baseline_pred)

    score_data_baseline["Lasso Regressor"] = lasso_baseline_scores

    logger.info(f"\nR2 score - Lasso Regression (Baseline): {lasso_baseline_scores[0]}")
    logger.info(f"Mean squared error - Lasso Regression (Baseline): {lasso_baseline_scores[2]}")
    logger.info(f"Mean absolute error - Lasso Regression (Baseline): {lasso_baseline_scores[3]}")
    logger.info(f"Root mean squared error - Lasso Regression (Baseline): {lasso_baseline_scores[4]}")


def create_fine_tuned_lasso_regressor(X_train, y_train, X_test, y_test) -> None:
    """
    After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
           "alpha": 0.8
        }

    """
    lasso_fine_tuned_model = Lasso(alpha=0.8)

    lasso_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=lasso_fine_tuned_model,
        file_name_and_path="models/fine_tuned/lasso.pkl"
    )

    lasso_fine_tuned_pred = lasso_fine_tuned_model.predict(X_test)

    lasso_fine_tuned_scores = get_result_metrics(y_test, lasso_fine_tuned_pred)

    score_data_tuned["Lasso Regressor"] = lasso_fine_tuned_scores

    logger.info(f"\nR2 score - Lasso Regression (Fine Tuned): {lasso_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - Lasso Regression (Fine Tuned): {lasso_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - Lasso Regression (Fine Tuned): {lasso_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - Lasso Regression (Fine Tuned): {lasso_fine_tuned_scores[4]}")


def create_baseline_gbr(X_train, y_train, X_test, y_test) -> None:
    gbr_baseline_model = GradientBoostingRegressor(random_state=42)
    gbr_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=gbr_baseline_model,
        file_name_and_path="models/baseline/gbr.pkl"
    )

    gbr_baseline_pred = gbr_baseline_model.predict(X_test)

    gbr_baseline_scores = get_result_metrics(y_test, gbr_baseline_pred)

    score_data_baseline["GBR Regressor"] = gbr_baseline_scores

    logger.info(f"\nR2 score - GBR Regression (Baseline): {gbr_baseline_scores[0]}")
    logger.info(f"Mean squared error - GBR Regression (Baseline): {gbr_baseline_scores[2]}")
    logger.info(f"Mean absolute error - GBR Regression (Baseline): {gbr_baseline_scores[3]}")
    logger.info(f"Root mean squared error - GBR Regression (Baseline): {gbr_baseline_scores[4]}")


def create_fine_tuned_gbr(X_train, y_train, X_test, y_test) -> None:
    """
    After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
            "subsample": 0.9,
            "n_estimators": 800,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_depth": 7,
            "learning_rate": 0.1
        }
    """
    gbr_fine_tuned_model = GradientBoostingRegressor(
        subsample=0.9,
        n_estimators=800,
        min_samples_split=10,
        min_samples_leaf=4,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    )
    gbr_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=gbr_fine_tuned_model,
        file_name_and_path="models/fine_tuned/gbr.pkl"
    )

    gbr_fine_tuned_pred = gbr_fine_tuned_model.predict(X_test)

    gbr_fine_tuned_scores = get_result_metrics(y_test, gbr_fine_tuned_pred)

    score_data_tuned["GBR Regressor"] = gbr_fine_tuned_scores

    logger.info(f"\nR2 score - GBR Regression (Fine Tuned): {gbr_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - GBR Regression (Fine Tuned): {gbr_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - GBR Regression (Fine Tuned): {gbr_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - GBR Regression (Fine Tuned): {gbr_fine_tuned_scores[4]}")


def create_baseline_svr(X_train, y_train, X_test, y_test) -> None:
    svr_baseline_model = SVR()
    svr_baseline_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=svr_baseline_model,
        file_name_and_path="models/baseline/svr.pkl"
    )

    svr_baseline_pred = svr_baseline_model.predict(X_test)

    svr_baseline_scores = get_result_metrics(y_test, svr_baseline_pred)

    score_data_baseline["SVR Regressor"] = svr_baseline_scores

    logger.info(f"\nR2 score - SVR Regression (Baseline): {svr_baseline_scores[0]}")
    logger.info(f"Mean squared error - SVR Regression (Baseline): {svr_baseline_scores[2]}")
    logger.info(f"Mean absolute error - SVR Regression (Baseline): {svr_baseline_scores[3]}")
    logger.info(f"Root mean squared error - SVR Regression (Baseline): {svr_baseline_scores[4]}")


def create_fine_tuned_svr(X_train, y_train, X_test, y_test) -> None:
    """
        After doing fine-tuning, we can observe the best hyperparameters in the phase 2 notebook -
        {
            "C": 10,
            "epsilon": 0.1,
            "gamma": "scale",
            "kernel": "rbf"
        }
    """
    svr_fine_tuned_model = SVR(
        C=10,
        epsilon=0.1,
        gamma="scale",
        kernel="rbf"
    )
    svr_fine_tuned_model.fit(X_train, y_train)

    save_trained_model_in_local(
        model=svr_fine_tuned_model,
        file_name_and_path="models/fine_tuned/svr.pkl"
    )

    svr_fine_tuned_pred = svr_fine_tuned_model.predict(X_test)

    svr_fine_tuned_scores = get_result_metrics(y_test, svr_fine_tuned_pred)

    score_data_tuned["SVR Regressor"] = svr_fine_tuned_scores

    logger.info(f"\nR2 score - SVR Regression (Fine Tuned): {svr_fine_tuned_scores[0]}")
    logger.info(f"Mean squared error - SVR Regression (Fine Tuned): {svr_fine_tuned_scores[2]}")
    logger.info(f"Mean absolute error - SVR Regression (Fine Tuned): {svr_fine_tuned_scores[3]}")
    logger.info(f"Root mean squared error - SVR Regression (Fine Tuned): {svr_fine_tuned_scores[4]}")


def train_models(df: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = get_train_test_split_data(df=df)
    print(x_train.shape)
    print(y_train.shape)

    create_baseline_decision_tree_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_decision_tree_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_random_forest_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_random_forest_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_knn_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_knn_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_xg_boost_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_xg_boost_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_light_gbm_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_light_gbm_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_lasso_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_lasso_regressor(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_gbr(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_gbr(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_baseline_svr(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )

    create_fine_tuned_svr(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test
    )
