# TASK 3 - NETFLIX AUDIENCE RATING CLASSIFICATION
# Machine Learning Internship Project
#
# Objective:
# Predict the audience rating category of Netflix content
# using available content attributes.
#
# Workflow:
# 1. Analyze rating categories
# 2. Prepare training dataset
# 3. Train classification models
# 4. Optimize model performance
# 5. Evaluate prediction accuracy
#
# Models:
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - Hyperparameter Tuning

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


print("=" * 70)
print("TASK 3 - NETFLIX AUDIENCE RATING CLASSIFICATION")
print("=" * 70)

# 2. LOAD DATASET

# Make sure Dataset.csv is uploaded to Google Colab

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# 3. BASIC DATA EXPLORATION

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
display(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 4. CLEAN DATA

# Remove duplicate rows

df = df.drop_duplicates().copy()

# Replace missing values in target rating

df["rating"] = df["rating"].fillna("Not Given")

# Convert selected columns to appropriate format

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

# Fill missing release years

df["release_year"] = df["release_year"].fillna(
    df["release_year"].median()
)

print("\nAfter cleaning:")
print("Dataset shape:", df.shape)

# 5. ANALYZE RATING CATEGORIES

print("\n" + "=" * 70)
print("RATING CATEGORY ANALYSIS")
print("=" * 70)

rating_counts = df["rating"].value_counts()

print("\nRating categories:")
print(rating_counts)


# Plot rating distribution

plt.figure(figsize=(12, 6))

rating_counts.plot(kind="bar")

plt.title("Netflix Audience Rating Distribution")

plt.xlabel("Rating Category")

plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 6. PREPARE FEATURES AND TARGET

# Target variable

y = df["rating"]


# Features selected from available Netflix content attributes

features = [
    "type",
    "director",
    "country",
    "release_year",
    "duration",
    "listed_in"
]

X = df[features].copy()


print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("\nFeatures used:")
print(features)

print("\nTarget:")
print("rating")

print("\nFeature shape:", X.shape)

print("Target shape:", y.shape)

# 7. REMOVE VERY RARE RATING CATEGORIES
# Some Netflix rating categories have very few samples.
# Extremely rare classes can cause problems during stratified
# train/test splitting.
#
# We keep categories having at least 10 samples.

rating_frequency = y.value_counts()

valid_ratings = rating_frequency[
    rating_frequency >= 10
].index

df_model = df[
    df["rating"].isin(valid_ratings)
].copy()

X = df_model[features].copy()

y = df_model["rating"].copy()

print("\n" + "=" * 70)
print("RATINGS USED FOR MODELING")
print("=" * 70)

print(y.value_counts())

print("\nNumber of rating categories:",
      y.nunique())

# 8. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES

categorical_features = [
    "type",
    "director",
    "country",
    "duration",
    "listed_in"
]

numerical_features = [
    "release_year"
]

# 9. PREPROCESSING

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# 10. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print("Training samples:", X_train.shape[0])

print("Testing samples:", X_test.shape[0])

# 11. LOGISTIC REGRESSION MODEL

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(
    X_test
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy, 4)
)

# 12. DECISION TREE MODEL

print("\n" + "=" * 70)
print("TRAINING DECISION TREE")
print("=" * 70)


decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        )
    ]
)


decision_tree_model.fit(
    X_train,
    y_train
)


decision_tree_predictions = decision_tree_model.predict(
    X_test
)


decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_predictions
)


print(
    "Decision Tree Accuracy:",
    round(decision_tree_accuracy, 4)
)

# 13. RANDOM FOREST MODEL

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_predictions = random_forest_model.predict(
    X_test
)


random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)


print(
    "Random Forest Accuracy:",
    round(random_forest_accuracy, 4)
)

# 14. MODEL ACCURACY COMPARISON

print("\n" + "=" * 70)
print("MODEL ACCURACY COMPARISON")
print("=" * 70)


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        random_forest_accuracy
    ]
})


results = results.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)


display(results)

# 15. ACCURACY GRAPH

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title(
    "Netflix Rating Classification - Model Accuracy"
)

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy")

plt.ylim(
    max(0, results["Accuracy"].min() - 0.10),
    1.0
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# 16. CLASSIFICATION REPORT - RANDOM FOREST


print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        random_forest_predictions,
        zero_division=0
    )
)

# 17. CONFUSION MATRIX - RANDOM FOREST

print("\n" + "=" * 70)
print("RANDOM FOREST CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    random_forest_predictions,
    labels=random_forest_model.classes_
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=random_forest_model.classes_
)


fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Random Forest - Netflix Rating Confusion Matrix"
)

plt.tight_layout()

plt.show()

# 18. HYPERPARAMETER TUNING

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 70)

print("\nGridSearchCV is being performed...")
print("Please wait...")


tuning_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],
    "classifier__max_depth": [
        10,
        20,
        None
    ],
    "classifier__min_samples_split": [
        2,
        5
    ]
}


grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


print("\nHyperparameter tuning completed!")

# 19. BEST PARAMETERS

print("\n" + "=" * 70)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 70)

print(
    grid_search.best_params_
)

print(
    "\nBest Cross-Validation Accuracy:",
    round(grid_search.best_score_, 4)
)

# 20. OPTIMIZED MODEL EVALUATION

best_model = grid_search.best_estimator_


optimized_predictions = best_model.predict(
    X_test
)


optimized_accuracy = accuracy_score(
    y_test,
    optimized_predictions
)


print("\n" + "=" * 70)
print("OPTIMIZED MODEL PERFORMANCE")
print("=" * 70)

print(
    "Optimized Random Forest Test Accuracy:",
    round(optimized_accuracy, 4)
)

# 21. OPTIMIZED CLASSIFICATION REPORT

print("\n" + "=" * 70)
print("OPTIMIZED RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        y_test,
        optimized_predictions,
        zero_division=0
    )
)

# 22. OPTIMIZED CONFUSION MATRIX
print("\n" + "=" * 70)
print("OPTIMIZED RANDOM FOREST CONFUSION MATRIX")
print("=" * 70)


optimized_cm = confusion_matrix(
    y_test,
    optimized_predictions,
    labels=best_model.classes_
)


optimized_disp = ConfusionMatrixDisplay(
    confusion_matrix=optimized_cm,
    display_labels=best_model.classes_
)


fig, ax = plt.subplots(figsize=(12, 10))

optimized_disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Optimized Random Forest - Netflix Rating Confusion Matrix"
)

plt.tight_layout()

plt.show()

# 23. FINAL MODEL COMPARISON

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Optimized Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        random_forest_accuracy,
        optimized_accuracy
    ]
})


final_results = final_results.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)


display(final_results)

# 24. FINAL ACCURACY GRAPH

plt.figure(figsize=(11, 6))

plt.bar(
    final_results["Model"],
    final_results["Accuracy"]
)

plt.title(
    "Netflix Audience Rating Classification - Final Model Comparison"
)

plt.xlabel("Model")

plt.ylabel("Test Accuracy")

plt.ylim(
    max(0, final_results["Accuracy"].min() - 0.10),
    1.0
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.show()

# 25. CUSTOM RATING PREDICTION

print("\n" + "=" * 70)
print("CUSTOM NETFLIX RATING PREDICTION")
print("=" * 70)


# Example Netflix content

sample_content = pd.DataFrame({
    "type": ["Movie"],
    "director": ["Not Given"],
    "country": ["United States"],
    "release_year": [2020],
    "duration": ["100 min"],
    "listed_in": ["Dramas, International Movies"]
})


predicted_rating = best_model.predict(
    sample_content
)


print("\nSample Content:")

display(sample_content)

print(
    "Predicted Audience Rating:",
    predicted_rating[0]
)

# 26. SAVE MODEL RESULTS

final_results.to_csv(
    "netflix_rating_model_results.csv",
    index=False
)


# Save optimized model

joblib.dump(
    best_model,
    "netflix_rating_classification_model.pkl"
)


# Save test predictions

prediction_output = X_test.copy()

prediction_output["Actual_Rating"] = y_test.values

prediction_output["Predicted_Rating"] = optimized_predictions

prediction_output.to_csv(
    "netflix_rating_predictions.csv",
    index=False
)


print("\nFiles saved successfully:")
print("1. netflix_rating_model_results.csv")
print("2. netflix_rating_classification_model.pkl")
print("3. netflix_rating_predictions.csv")

# 27. FINAL PROJECT SUMMARY

print("\n")
print("=" * 70)
print("TASK 3 - PROJECT SUMMARY")
print("=" * 70)

print("Project: Netflix Audience Rating Classification")

print("\nObjective:")
print(
    "Predict the audience rating category of Netflix content "
    "using content attributes."
)

print("\nNumber of rating categories:",
      y.nunique())

print("\nModels trained:")
print("- Logistic Regression")
print("- Decision Tree")
print("- Random Forest")
print("- Optimized Random Forest")

print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    "\nBest Cross-Validation Accuracy:",
    round(grid_search.best_score_, 4)
)

print(
    "\nOptimized Random Forest Test Accuracy:",
    round(optimized_accuracy, 4)
)

print("\nGenerated files:")
print("- netflix_rating_model_results.csv")
print("- netflix_rating_classification_model.pkl")
print("- netflix_rating_predictions.csv")

print("\n" + "=" * 70)
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)

Output:
======================================================================
TASK 3 - NETFLIX AUDIENCE RATING CLASSIFICATION
======================================================================

Dataset loaded successfully!
Dataset shape: (8790, 10)

======================================================================
DATASET INFORMATION
======================================================================

First 5 rows:
show_id	type	title	director	country	date_added	release_year	rating	duration	listed_in
0	s1	Movie	Dick Johnson Is Dead	Kirsten Johnson	United States	9/25/2021	2020	PG-13	90 min	Documentaries
1	s3	TV Show	Ganglands	Julien Leclercq	France	9/24/2021	2021	TV-MA	1 Season	Crime TV Shows, International TV Shows, TV Act...
2	s6	TV Show	Midnight Mass	Mike Flanagan	United States	9/24/2021	2021	TV-MA	1 Season	TV Dramas, TV Horror, TV Mysteries
3	s14	Movie	Confessions of an Invisible Girl	Bruno Garotti	Brazil	9/22/2021	2021	TV-PG	91 min	Children & Family Movies, Comedies
4	s8	Movie	Sankofa	Haile Gerima	United States	9/24/2021	1993	TV-MA	125 min	Dramas, Independent Movies, International Movies

Column names:
['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']

Dataset information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8790 entries, 0 to 8789
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       8790 non-null   object
 1   type          8790 non-null   object
 2   title         8790 non-null   object
 3   director      8790 non-null   object
 4   country       8790 non-null   object
 5   date_added    8790 non-null   object
 6   release_year  8790 non-null   int64 
 7   rating        8790 non-null   object
 8   duration      8790 non-null   object
 9   listed_in     8790 non-null   object
dtypes: int64(1), object(9)
memory usage: 686.8+ KB

Missing values:
show_id         0
type            0
title           0
director        0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
dtype: int64

Duplicate rows: 0

After cleaning:
Dataset shape: (8790, 10)

======================================================================
RATING CATEGORY ANALYSIS
======================================================================

Rating categories:
rating
TV-MA       3205
TV-14       2157
TV-PG        861
R            799
PG-13        490
TV-Y7        333
TV-Y         306
PG           287
TV-G         220
NR            79
G             41
TV-Y7-FV       6
NC-17          3
UR             3
Name: count, dtype: int64


======================================================================
FEATURES AND TARGET
======================================================================

Features used:
['type', 'director', 'country', 'release_year', 'duration', 'listed_in']

Target:
rating

Feature shape: (8790, 6)
Target shape: (8790,)

======================================================================
RATINGS USED FOR MODELING
======================================================================
rating
TV-MA    3205
TV-14    2157
TV-PG     861
R         799
PG-13     490
TV-Y7     333
TV-Y      306
PG        287
TV-G      220
NR         79
G          41
Name: count, dtype: int64

Number of rating categories: 11

======================================================================
TRAIN-TEST SPLIT
======================================================================
Training samples: 7022
Testing samples: 1756

======================================================================
TRAINING LOGISTIC REGRESSION
======================================================================
/usr/local/lib/python3.13/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Logistic Regression Accuracy: 0.4949

======================================================================
TRAINING DECISION TREE
======================================================================
Decision Tree Accuracy: 0.5

======================================================================
TRAINING RANDOM FOREST
======================================================================
Random Forest Accuracy: 0.4334

======================================================================
MODEL ACCURACY COMPARISON
======================================================================
Model	Accuracy
0	Decision Tree	0.500000
1	Logistic Regression	0.494875
2	Random Forest	0.433371


======================================================================
RANDOM FOREST CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

           G       0.00      0.00      0.00         8
          NR       0.00      0.00      0.00        16
          PG       1.00      0.05      0.10        57
       PG-13       1.00      0.03      0.06        98
           R       0.80      0.10      0.18       160
       TV-14       0.60      0.30      0.40       432
        TV-G       0.00      0.00      0.00        44
       TV-MA       0.40      0.94      0.56       641
       TV-PG       1.00      0.01      0.02       172
        TV-Y       1.00      0.03      0.06        61
       TV-Y7       0.67      0.06      0.11        67

    accuracy                           0.43      1756
   macro avg       0.59      0.14      0.14      1756
weighted avg       0.61      0.43      0.33      1756


======================================================================
RANDOM FOREST CONFUSION MATRIX
======================================================================


======================================================================
HYPERPARAMETER TUNING - RANDOM FOREST
======================================================================

GridSearchCV is being performed...
Please wait...
Fitting 3 folds for each of 12 candidates, totalling 36 fits

Hyperparameter tuning completed!

======================================================================
BEST RANDOM FOREST PARAMETERS
======================================================================
{'classifier__max_depth': None, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}

Best Cross-Validation Accuracy: 0.5393

======================================================================
OPTIMIZED MODEL PERFORMANCE
======================================================================
Optimized Random Forest Test Accuracy: 0.541

======================================================================
OPTIMIZED RANDOM FOREST CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

           G       1.00      0.50      0.67         8
          NR       0.00      0.00      0.00        16
          PG       0.56      0.40      0.47        57
       PG-13       0.51      0.40      0.45        98
           R       0.57      0.53      0.55       160
       TV-14       0.52      0.46      0.49       432
        TV-G       0.30      0.07      0.11        44
       TV-MA       0.57      0.79      0.66       641
       TV-PG       0.33      0.15      0.21       172
        TV-Y       0.50      0.54      0.52        61
       TV-Y7       0.51      0.43      0.47        67

    accuracy                           0.54      1756
   macro avg       0.49      0.39      0.42      1756
weighted avg       0.52      0.54      0.52      1756


======================================================================
OPTIMIZED RANDOM FOREST CONFUSION MATRIX
======================================================================


======================================================================
FINAL MODEL COMPARISON
======================================================================
Model	Accuracy
0	Optimized Random Forest	0.541002
1	Decision Tree	0.500000
2	Logistic Regression	0.494875
3	Random Forest	0.433371


======================================================================
CUSTOM NETFLIX RATING PREDICTION
======================================================================

Sample Content:
type	director	country	release_year	duration	listed_in
0	Movie	Not Given	United States	2020	100 min	Dramas, International Movies
Predicted Audience Rating: TV-MA

Files saved successfully:
1. netflix_rating_model_results.csv
2. netflix_rating_classification_model.pkl
3. netflix_rating_predictions.csv


======================================================================
TASK 3 - PROJECT SUMMARY
======================================================================
Project: Netflix Audience Rating Classification

Objective:
Predict the audience rating category of Netflix content using content attributes.

Number of rating categories: 11

Models trained:
- Logistic Regression
- Decision Tree
- Random Forest
- Optimized Random Forest

Best Parameters:
{'classifier__max_depth': None, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}

Best Cross-Validation Accuracy: 0.5393

Optimized Random Forest Test Accuracy: 0.541

Generated files:
- netflix_rating_model_results.csv
- netflix_rating_classification_model.pkl
- netflix_rating_predictions.csv

======================================================================
TASK 3 COMPLETED SUCCESSFULLY!
======================================================================
