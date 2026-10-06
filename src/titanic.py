"""
Titanic Survival Prediction - single-file version (works on Windows/Mac/Linux).

Setup:
    pip install pandas scikit-learn matplotlib seaborn joblib
Run:
    python titanic.py
The script looks for the dataset automatically (see find_dataset below).
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
FALLBACK_URL = ("https://raw.githubusercontent.com/datasciencedojo/"
                "datasets/master/titanic.csv")


# ------------------------------------------------------------ 0. Find the data
def find_dataset():
    """Return a path/URL to the Titanic CSV, trying several places."""
    # 1) any CSV with 'titanic' in the name next to the script, in ./data,
    #    in the current folder, or in the user's Downloads folder
    folders = [BASE_DIR, BASE_DIR / "data", Path.cwd(), Path.home() / "Downloads"]
    for folder in folders:
        if folder.is_dir():
            for f in sorted(folder.glob("*.csv")):
                if "titanic" in f.name.lower():
                    return f

    # 2) ask the user to pick the file (opens a small window)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Select the Titanic CSV file", filetypes=[("CSV files", "*.csv")])
        root.destroy()
        if chosen:
            return Path(chosen)
    except Exception:
        pass

    # 3) last resort: download the public copy of the dataset
    print("Local file not found - downloading the dataset from GitHub...")
    return FALLBACK_URL


data_source = find_dataset()
print(f"Using dataset: {data_source}\n")
try:
    df = pd.read_csv(data_source)
except Exception as e:
    raise SystemExit(
        f"\nCould not load the dataset ({e}).\n"
        f"Copy Titanic-Dataset.csv into this folder:\n  {BASE_DIR}\nand run again.")

# ---------------------------------------------------------------- 1. Overview
print("Shape:", df.shape)
missing = df.isna().sum()
print("Missing values:\n", missing[missing > 0], "\n")

# ------------------------------------------------------ 2. Feature engineering
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("U")
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

numeric = ["Age", "Fare", "FamilySize", "FarePerPerson"]
categorical = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin", "Deck"]
X = df[numeric + categorical]
y = df["Survived"]

# ------------------------------------------------------------- 3. Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), numeric),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])


def make_pipe(model):
    return Pipeline([("prep", preprocess), ("model", model)])


# ------------------------------------------------------------------- 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ------------------------------------------------------ 5. Compare the models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
print("5-fold CV accuracy:")
for name, m in models.items():
    s = cross_val_score(make_pipe(m), X_train, y_train, cv=cv)
    print(f"  {name:20s} {s.mean():.3f} +/- {s.std():.3f}")

# ----------------------------------------------------------- 6. Tune the model
grid = GridSearchCV(
    make_pipe(RandomForestClassifier(random_state=SEED)),
    {"model__n_estimators": [200, 400],
     "model__max_depth": [4, 6, 8, None],
     "model__min_samples_leaf": [1, 3, 5]},
    cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.3f}")

# ------------------------------------------------------------- 7. Test results
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
print("\nTest set results:")
print(classification_report(y_test, pred, target_names=["Died", "Survived"]))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

# ----------------------------------------------------------------- 8. Plots
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Died", "Survived"], cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()

names = [n.split("__", 1)[1] for n in best.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
plt.figure(figsize=(7, 6))
imp.sort_values().tail(15).plot.barh(color="steelblue")
plt.title("Top 15 feature importances")
plt.savefig(BASE_DIR / "feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- 9. Save model
joblib.dump(best, BASE_DIR / "titanic_model.joblib")
print(f"\nSaved model and plots to: {BASE_DIR}")

# ------------------------------------------------------ 10. Example prediction
passenger = pd.DataFrame([{
    "Age": 29, "Fare": 80.0, "FamilySize": 2, "FarePerPerson": 40.0,
    "Pclass": 1, "Sex": "female", "Embarked": "S", "Title": "Mrs",
    "IsAlone": 0, "HasCabin": 1, "Deck": "B",
}])
print(f"Example passenger survival probability: "
      f"{best.predict_proba(passenger)[0, 1]:.1%}")