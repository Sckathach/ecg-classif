import argparse
import json
import pickle
import sys

import mlflow
import numpy as np
import optuna
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load Data
DATA_PATH = "data/processed/features.pkl"


def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    return data


def get_pipeline(params):
    steps = []

    # Scaling
    steps.append(("scaler", StandardScaler()))

    # Feature Selection
    selector_type = params["selector_type"]
    n_features = params["n_features"]

    if selector_type == "ANOVA":
        steps.append(("selector", SelectKBest(f_classif, k=n_features)))
    elif selector_type == "RFE":
        # Use a lightweight estimator for RFE to ensure compatibility and speed
        rfe_estimator = DecisionTreeClassifier(random_state=42)
        steps.append(
            ("selector", RFE(estimator=rfe_estimator, n_features_to_select=n_features))
        )
    else:  # None
        pass  # No selection

    # Classifier
    clf_type = params["classifier"]
    if clf_type == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            random_state=42,
            n_jobs=1,
        )
    elif clf_type == "SVM":
        clf = SVC(
            C=params["svm_C"],
            gamma=params["svm_gamma"],
            kernel=params["svm_kernel"],
            probability=True,  # Needed if we want probas, but for acc it's ok.
            random_state=42,
        )
    elif clf_type == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=params["gb_n_estimators"],
            learning_rate=params["gb_learning_rate"],
            max_depth=params["gb_max_depth"],
            random_state=42,
        )
    elif clf_type == "LogisticRegression":
        clf = LogisticRegression(
            C=params["lr_C"], solver="lbfgs", max_iter=1000, random_state=42
        )

    steps.append(("classifier", clf))

    return Pipeline(steps)


def objective(trial, X, y):
    params = {}

    # 1. Feature Selection
    params["selector_type"] = trial.suggest_categorical(
        "selector_type", ["ANOVA", "RFE", "None"]
    )
    if params["selector_type"] != "None":
        max_features = min(50, X.shape[1])
        params["n_features"] = trial.suggest_int("n_features", 5, max_features, step=5)
    else:
        params["n_features"] = 0  # Dummy

    # 2. Classifier
    params["classifier"] = trial.suggest_categorical(
        "classifier", ["RandomForest", "SVM", "GradientBoosting", "LogisticRegression"]
    )

    if params["classifier"] == "RandomForest":
        params["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 200)
        params["rf_max_depth"] = trial.suggest_int("rf_max_depth", 2, 20)

    elif params["classifier"] == "SVM":
        params["svm_C"] = trial.suggest_float("svm_C", 1e-3, 1e2, log=True)
        params["svm_gamma"] = trial.suggest_categorical("svm_gamma", ["scale", "auto"])
        params["svm_kernel"] = trial.suggest_categorical(
            "svm_kernel", ["linear", "rbf", "poly"]
        )

    elif params["classifier"] == "GradientBoosting":
        params["gb_n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 200)
        params["gb_learning_rate"] = trial.suggest_float(
            "gb_learning_rate", 1e-4, 1.0, log=True
        )
        params["gb_max_depth"] = trial.suggest_int("gb_max_depth", 2, 10)

    elif params["classifier"] == "LogisticRegression":
        params["lr_C"] = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)

    # Instantiate Pipeline
    pipeline = get_pipeline(params)

    # Cross Validation (LeaveOneOut)
    loo = LeaveOneOut()

    # We can use cross_val_score for efficiency/cleanliness
    scores = cross_val_score(pipeline, X, y, cv=loo, scoring="accuracy", n_jobs=-1)
    accuracy = scores.mean()

    return accuracy


def run_sweep(n_trials):
    data = load_data()
    bands = ["ALPHA", "BETA", "THETA", "DELTA"]

    best_configs = {}

    # Set Optuna logging to avoid spam
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    for band in bands:
        logger.info(f"Starting sweep for Band: {band}")

        def band_objective(trial):
            # Select Mode
            mode = trial.suggest_categorical("mode", ["regional", "vector"])

            # Get Data
            X = data[mode][band]["X"]
            y = data[mode][band]["y"]

            # Ensure y is 1D
            if y.ndim > 1:
                y = y.ravel()

            # Log params
            with mlflow.start_run(nested=True):
                acc = objective(trial, X, y)
                mlflow.log_params(trial.params)
                mlflow.log_metric("accuracy", acc)

            return acc

        study_name = f"study_{band}"
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(band_objective, n_trials=n_trials)

        logger.info(f"Band {band} Best Accuracy: {study.best_value}")
        logger.info(f"Band {band} Best Params: {study.best_params}")

        best_configs[band] = study.best_params

    return best_configs


def run_ensemble(best_configs, data):
    logger.info("Starting Ensemble Phase...")

    bands = ["ALPHA", "BETA", "THETA", "DELTA"]

    # Store predictions for meta-model
    # Shape: (n_samples, n_bands)
    # We assume y is consistent across bands/modes for the same subject order.
    # We'll take y from one of them to verify.
    first_y = data["regional"]["ALPHA"]["y"]  # Assuming this exists and is standard
    n_samples = len(first_y)

    X_meta = np.zeros((n_samples, len(bands)))

    for i, band in enumerate(bands):
        logger.info(f"Generating OOF predictions for {band}...")
        params = best_configs[band]
        mode = params["mode"]

        X = data[mode][band]["X"]
        y = data[mode][band]["y"]
        if y.ndim > 1:
            y = y.ravel()

        # Reconstruct pipeline from params
        # Note: params contains 'mode' which is not for get_pipeline, keys need to match
        # get_pipeline expects specific keys. params has them flat.

        pipeline = get_pipeline(params)

        # Generate OOF predictions
        # cross_val_predict with cv=LeaveOneOut returns the prediction for each sample when it was in test set
        loo = LeaveOneOut()
        preds = cross_val_predict(pipeline, X, y, cv=loo, n_jobs=-1)

        X_meta[:, i] = preds

    logger.info("Training Meta-Classifier...")

    # Meta Classifier
    # We can use a simple LogisticRegression or check majority vote.
    # Since inputs are class labels (predictions), LR might need one-hot if classes are categorical?
    # Or if predictions are 0/1 (binary), LR is fine.
    # If multiclass, maybe use mode (majority voting) or a meta-learner.

    # Let's check if predictions are binary or multiclass from data?
    # Assuming binary for now based on 'ECG Classif' context usually, but could be multi.
    # Let's use a robust meta learner: Logistic Regression on predictions.
    # If the base models output 0/1, X_meta is binary.

    meta_clf = LogisticRegression()
    loo = LeaveOneOut()

    # Evaluate Meta Classifier
    final_scores = cross_val_score(
        meta_clf, X_meta, first_y, cv=loo, scoring="accuracy"
    )
    final_acc = final_scores.mean()

    logger.info(f"Final Ensemble Accuracy: {final_acc}")
    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials", type=int, default=5, help="Number of Optuna trials per band"
    )
    args = parser.parse_args()

    mlflow.set_experiment("ECG_Band_Sweep")

    best_configs = run_sweep(args.n_trials)

    logger.info("Best Configurations found:")
    print(json.dumps(best_configs, indent=2))

    # Load data again or pass it (it's loaded in run_sweep but we need it for ensemble)
    # Better to load globally or pass.
    data = load_data()
    run_ensemble(best_configs, data)
