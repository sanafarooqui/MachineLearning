"""Helpers for training baseline models and tuning hyperparameters."""

import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def get_baseline_model_stats(models, preprocessor, X_train, y_train, X_test, y_test):
    """Fit each model with default hyperparameters and collect basic scores.

    ``models`` maps a name to an unfitted estimator. Returns a list of dicts
    (one per model) suitable for ``pd.DataFrame``.
    """
    results_score = []
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (name, model),
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time

        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)

        y_pred = pipeline.predict(X_test)

        results_score.append({
            "Model": name,
            "Train time": fit_time,
            "Train accuracy": train_score,
            "Test accuracy": test_score,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        })

    return results_score


def get_model_tuning_stats(models, preprocessor, X_train, y_train, X_test, y_test,
                           sample_strategy="SMOTE", cv=CV):
    """Grid-search each model (optionally with resampling) and score on the test set.

    ``models`` maps a name to ``(estimator, param_grid)``. ``sample_strategy``
    is one of ``"SMOTE"``, ``"UNDER_SAMPLE"``, or anything else for no
    resampling. Returns a list of dicts suitable for ``pd.DataFrame``.
    """
    results_score = []
    for name, (model, params) in models.items():
        if sample_strategy == "SMOTE":
            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                (name, model),
            ])
        elif sample_strategy == "UNDER_SAMPLE":
            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("undersample", RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
                (name, model),
            ])
        else:
            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                (name, model),
            ])

        prefixed_params = {
            f"{name.lower().replace(' ', '')}__{param_name}": param_values
            for param_name, param_values in params.items()
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed_params,
            cv=cv,
            n_jobs=-1,
            scoring="f1",
            refit=True,
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        fit_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        print(grid_search.best_params_)

        y_pred = best_model.predict(X_test)
        roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        results_score.append({
            "Model": name,
            "Sampling strategy": sample_strategy,
            "Train time": fit_time,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc,
        })

    return results_score
