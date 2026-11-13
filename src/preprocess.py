import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

def correct_feature(df):
    condition = df["Social_Media_Hours"] > df["Screen_Time_Hours"]

    df.loc[condition, "Screen_Time_Hours"] = df.loc[condition, "Social_Media_Hours"]

    return df


def target_isolation(df):
    TARGET = "Mental_Health_Risk"

    X = df.drop(columns=["Student_ID", TARGET])
    y = df[TARGET]

    return X, y

def feats_engineering(df):
    df["Study_Hours"] = df["Study_Hours_Per_Week"] / 7
    df = df.drop(columns="Study_Hours_Per_Week")
    return df

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def feats_categorization():
    numerical_features = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate"
]

    nominal_features = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_features = [
        "Physical_Activity",
        "Diet_Quality"
    ]

    return numerical_features, nominal_features, ordinal_features

def numericals_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return numeric_pipeline

def nominals_pipeline():
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return nominal_pipeline

def ordinals_pipeline():
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    return ordinal_pipeline

def preprocessor_col_transformer(numeric_pipeline, numerical_features, nominal_pipeline, nominal_features, ordinal_pipeline, ordinal_features):
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("nom", nominal_pipeline, nominal_features),
        ("ord", ordinal_pipeline, ordinal_features)
    ])
    return preprocessor

def preprocess(df):
    df = correct_feature(df)
    df = feats_engineering(df)
    X, y = target_isolation(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    numerical_features, nominal_features, ordinal_features = feats_categorization()
    numeric_pipeline = numericals_pipeline()
    nominal_pipeline = nominals_pipeline()
    ordinal_pipeline = ordinals_pipeline()
    preprocessor = preprocessor_col_transformer(
        numeric_pipeline,
        numerical_features,
        nominal_pipeline,
        nominal_features,
        ordinal_pipeline,
        ordinal_features,
    )

    return X_train, X_test, y_train, y_test, preprocessor