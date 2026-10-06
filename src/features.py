"""Feature engineering and preprocessing pipeline for the Titanic dataset."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona",
}
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

NUMERIC = ["Age", "Fare", "FamilySize", "FarePerPerson"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin", "Deck"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived columns: Title, FamilySize, IsAlone, HasCabin, Deck, FarePerPerson."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Title"] = (
            df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        )
        df["Title"] = df["Title"].replace(TITLE_MAP)
        df["Title"] = df["Title"].where(~df["Title"].isin(RARE_TITLES), "Rare")
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].str[0].fillna("U")
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def build_pipeline(model) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> model."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
