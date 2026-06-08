# =============================================================================
# app.py
# Disease Prediction Using Machine Learning — Flask Backend
# This file handles all routes, loads ML models, processes user symptoms,
# and returns the top 3 predicted diseases with confidence, description,
# and precautions.
# =============================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

# =============================================================================
# STEP 1: Initialize Flask app
# =============================================================================

app = Flask(__name__)

# =============================================================================
# STEP 2: Load all saved models and supporting data at startup
# =============================================================================

print("Loading models...")

# Load the 4 trained ML models
rf_model  = joblib.load("models/RandomForest.pkl")
dt_model  = joblib.load("models/DecisionTree.pkl")
nb_model  = joblib.load("models/NaiveBayes.pkl")
knn_model = joblib.load("models/KNN.pkl")

# Load the symptom column list (132 symptoms in correct order)
symptom_columns = joblib.load("models/symptom_columns.pkl")

# Load the disease list
disease_list = joblib.load("models/disease_list.pkl")

print("Models loaded successfully!")

# =============================================================================
# STEP 3: Load supplementary CSV files (description + precautions)
# =============================================================================

# Disease descriptions
desc_df = pd.read_csv("dataset/symptom_description.csv")
desc_df.columns = desc_df.columns.str.strip()  # Remove any extra spaces

# Disease precautions
prec_df = pd.read_csv("dataset/symptom_precaution.csv")
prec_df.columns = prec_df.columns.str.strip()

print("Supplementary data loaded!")

# =============================================================================
# HELPER FUNCTION: Get disease description
# =============================================================================

def get_description(disease_name):
    """Returns the description of a disease from symptom_description.csv"""
    result = desc_df[desc_df["Disease"].str.strip() == disease_name.strip()]
    if not result.empty:
        return result.iloc[0]["Description"]
    return "No description available."

# =============================================================================
# HELPER FUNCTION: Get disease precautions
# =============================================================================

def get_precautions(disease_name):
    """Returns a list of precautions for a disease from symptom_precaution.csv"""
    result = prec_df[prec_df["Disease"].str.strip() == disease_name.strip()]
    if not result.empty:
        # Precaution columns are Precaution_1, Precaution_2, Precaution_3, Precaution_4
        precautions = []
        for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            val = result.iloc[0].get(col, "")
            if pd.notna(val) and str(val).strip() != "":
                precautions.append(str(val).strip())
        return precautions
    return ["No precautions available."]

# =============================================================================
# HELPER FUNCTION: Build input vector from selected symptoms
# =============================================================================

def build_input_vector(selected_symptoms):
    """
    Creates a binary input array of length 132.
    1 = symptom is present, 0 = symptom is absent.
    This is what the ML models expect as input.
    """
    input_vector = np.zeros(len(symptom_columns))
    for symptom in selected_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_columns:
            index = symptom_columns.index(symptom)
            input_vector[index] = 1
    return input_vector.reshape(1, -1)  # Reshape for model input

# =============================================================================
# HELPER FUNCTION: Get top 3 predictions with confidence
# =============================================================================

def get_top3_predictions(input_vector, model):
    """
    Uses predict_proba to get confidence scores for all diseases,
    then returns the top 3 diseases with their confidence percentages.
    """
    # Get probability scores for all 42 diseases
    probabilities = model.predict_proba(input_vector)[0]

    # Get class labels from the model
    classes = model.classes_

    # Combine class names with their probabilities
    disease_prob_pairs = list(zip(classes, probabilities))

    # Sort by probability (highest first)
    disease_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # Take top 3
    top3 = disease_prob_pairs[:3]

    return top3

# =============================================================================
# ROUTE 1: Home page — shows the symptom selection form
# =============================================================================

@app.route("/")
def index():
    """
    Renders the main page (index.html).
    Passes the list of all 132 symptoms to the template
    so they can be displayed as checkboxes or a dropdown.
    """
    # Clean up symptom names for display (replace underscores with spaces)
    display_symptoms = [s.replace("_", " ") for s in symptom_columns]

    # Create a dict mapping display name → actual column name
    symptom_map = dict(zip(display_symptoms, symptom_columns))

    return render_template("index.html",
                           symptoms=display_symptoms,
                           symptom_map=symptom_map)

# =============================================================================
# ROUTE 2: Predict — processes form submission and returns results
# =============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives selected symptoms from the form,
    runs prediction using Random Forest (primary model),
    and renders result.html with top 3 diseases + details.
    """

    # Get selected symptoms from the form (list of symptom names)
    selected_display = request.form.getlist("symptoms")

    # Convert display names back to column names (underscored)
    selected_symptoms = [s.replace(" ", "_") for s in selected_display]

    # Safety check — user must select at least 1 symptom
    if not selected_symptoms:
        return render_template("index.html",
                               symptoms=[s.replace("_", " ") for s in symptom_columns],
                               error="Please select at least one symptom.")

    # Build the binary input vector
    input_vector = build_input_vector(selected_symptoms)

    # -------------------------------------------------------------------------
    # Run predictions using all 4 models
    # Primary model is Random Forest
    # -------------------------------------------------------------------------

    # Get top 3 predictions from Random Forest (primary)
    top3_rf  = get_top3_predictions(input_vector, rf_model)

    # Single predictions from other models (for comparison display)
    pred_dt  = dt_model.predict(input_vector)[0]
    pred_nb  = nb_model.predict(input_vector)[0]
    pred_knn = knn_model.predict(input_vector)[0]

    # -------------------------------------------------------------------------
    # Build result data for top 3 Random Forest predictions
    # -------------------------------------------------------------------------

    results = []
    for rank, (disease, confidence) in enumerate(top3_rf, start=1):
        results.append({
            "rank"        : rank,
            "disease"     : disease,
            "confidence"  : round(confidence * 100, 2),  # Convert to percentage
            "description" : get_description(disease),
            "precautions" : get_precautions(disease)
        })

    # -------------------------------------------------------------------------
    # Model comparison data (what each model predicted as #1)
    # -------------------------------------------------------------------------

    model_comparison = {
        "Random Forest" : top3_rf[0][0],   # Top prediction from RF
        "Decision Tree" : pred_dt,
        "Naive Bayes"   : pred_nb,
        "KNN"           : pred_knn
    }

    return render_template("result.html",
                           results=results,
                           selected_symptoms=selected_display,
                           model_comparison=model_comparison)

# =============================================================================
# ROUTE 3: API endpoint (optional — useful for testing via browser/Postman)
# =============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON-based API endpoint.
    Send a POST request with: { "symptoms": ["fever", "cough", ...] }
    Returns top 3 predictions as JSON.
    """
    data = request.get_json()
    selected_symptoms = data.get("symptoms", [])

    if not selected_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    input_vector = build_input_vector(selected_symptoms)
    top3 = get_top3_predictions(input_vector, rf_model)

    response = []
    for disease, confidence in top3:
        response.append({
            "disease"    : disease,
            "confidence" : round(confidence * 100, 2),
            "description": get_description(disease),
            "precautions": get_precautions(disease)
        })

    return jsonify({"predictions": response})

# =============================================================================
# Run the Flask app
# =============================================================================

if __name__ == "__main__":
    # debug=True means Flask will auto-reload when you save changes
    # Remove debug=True when deploying to production
    app.run(debug=True)