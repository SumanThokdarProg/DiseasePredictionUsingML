# =============================================================================
# model_train.py
# Disease Prediction Using Machine Learning
# This script loads the dataset, trains 4 ML models, evaluates them,
# and saves the trained models into the /models folder using joblib.
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =============================================================================
# STEP 1: Load the dataset
# =============================================================================

print("Loading datasets...")

# Load training and testing CSV files
train_df = pd.read_csv("dataset/Training.csv")
test_df  = pd.read_csv("dataset/Testing.csv")

# Drop the last unnamed column if it exists (some versions of the dataset have it)
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df  = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

print(f"Training samples : {train_df.shape[0]}")
print(f"Testing  samples : {test_df.shape[0]}")
print(f"Total features   : {train_df.shape[1] - 1}")  # -1 for the prognosis column

# =============================================================================
# STEP 2: Separate features (X) and labels (y)
# =============================================================================

# All columns except the last one ('prognosis') are symptom features
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]

X_test  = test_df.drop("prognosis", axis=1)
y_test  = test_df["prognosis"]

# Save the list of symptom column names — needed later by Flask app
symptom_columns = list(X_train.columns)
print(f"\nSymptom columns loaded: {len(symptom_columns)} symptoms")

# =============================================================================
# STEP 3: Define all 4 models
# =============================================================================

models = {
    "RandomForest"  : RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree"  : DecisionTreeClassifier(random_state=42),
    "NaiveBayes"    : GaussianNB(),
    "KNN"           : KNeighborsClassifier(n_neighbors=5)
}

# =============================================================================
# STEP 4: Train, evaluate, and save each model
# =============================================================================

# Make sure the models/ folder exists
os.makedirs("models", exist_ok=True)

print("\n--- Training Models ---\n")

trained_models = {}

for model_name, model in models.items():

    # --- Train ---
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy * 100:.2f}%")

    # --- Save to /models folder ---
    save_path = f"models/{model_name}.pkl"
    joblib.dump(model, save_path)
    print(f"  Saved → {save_path}")

    trained_models[model_name] = model

# =============================================================================
# STEP 5: Save the symptom column list
# This is used by app.py to build the input vector from user-selected symptoms
# =============================================================================

joblib.dump(symptom_columns, "models/symptom_columns.pkl")
print("\nSymptom column list saved → models/symptom_columns.pkl")

# =============================================================================
# STEP 6: Save the list of unique disease names (for reference)
# =============================================================================

disease_list = sorted(y_train.unique().tolist())
joblib.dump(disease_list, "models/disease_list.pkl")
print(f"Disease list saved → models/disease_list.pkl  ({len(disease_list)} diseases)")

# =============================================================================
# DONE
# =============================================================================

print("\n✅ All models trained and saved successfully!")
print("You can now run app.py to start the Flask web application.")