"""Train, compare, tune and evaluate models for Titanic survival prediction.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

from features import build_pipeline

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Titanic-Dataset.csv"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"


def eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    sns.barplot(data=df, x="Sex", y="Survived", ax=ax[0, 0])
    sns.barplot(data=df, x="Pclass", y="Survived", ax=ax[0, 1])
    sns.barplot(data=df, x="Embarked", y="Survived", ax=ax[0, 2])
    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, ax=ax[1, 0])
    sns.boxplot(data=df, x="Survived", y="Fare", ax=ax[1, 1])
    ax[1, 1].set_yscale("log")
    sns.barplot(data=df, x="Sex", y="Survived", hue="Pclass", ax=ax[1, 2])
    fig.suptitle("Titanic: survival rate by key features", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "eda.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    missing = df.isna().sum()
    print("Missing values:\n", missing[missing > 0], "\n")
    eda_plots(df)

    X = df.drop(columns=["Survived", "PassengerId", "Ticket"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ---- 1. Compare candidate models with 5-fold CV on the training set
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    print("5-fold CV accuracy (train set):")
    for name, model in candidates.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train,
                                 cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name:20s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # ---- 2. Tune a Random Forest
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 3, 5],
    }
    grid = GridSearchCV(
        build_pipeline(RandomForestClassifier(random_state=SEED)),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest RF params: {grid.best_params_}")
    print(f"Best RF CV accuracy: {grid.best_score_:.4f}")
    best = grid.best_estimator_

    # ---- 3. Final evaluation on the untouched test set
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_results": cv_results,
        "best_params": grid.best_params_,
    }
    print("\nTest-set results:")
    print(classification_report(y_test, pred, target_names=["Died", "Survived"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # ---- 4. Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Died", "Survived"], cmap="Blues", ax=ax[0])
    ax[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[1].set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(FIG / "evaluation.png", dpi=130)
    plt.close(fig)

    names = best.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Top 15 feature importances (Random Forest)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=130)
    plt.close(fig)

    # ---- 5. Save artifacts
    joblib.dump(best, MODELS / "titanic_model.joblib")
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved model to {MODELS / 'titanic_model.joblib'}")


if __name__ == "__main__":
    main()
