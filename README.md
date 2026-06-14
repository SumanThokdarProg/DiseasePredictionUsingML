> [!WARNING]
> **Disclaimer:** This __Disease Prediction__ project fully made by ```Claude```.

# Disease Prediction Using Machine Learning

*A Machine Learning Based Web Application for Predicting Diseases from Symptoms*

**Final Year Project Report**

Submitted by: Suman Thokdar

GitHub Repository: [SumanThokdarProg/DiseasePredictionUsingML](https://github.com/SumanThokdarProg/DiseasePredictionUsingML)

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Literature Review
5. Dataset Description
6. System Architecture
7. Machine Learning Models
8. Development Phases
9. Scope and Limitations
10. Conclusion
11. References

---

## 1. Introduction

Healthcare diagnosis is traditionally dependent on a physician's manual assessment of a patient's symptoms, medical history, and clinical tests. While effective, this process can be time-consuming, and early-stage self-assessment is often unavailable to people in remote or under-resourced areas. With the growth of Machine Learning (ML), it has become possible to build predictive systems that analyse patterns in symptom data and estimate the likelihood of a disease with reasonable accuracy.

This project, "Disease Prediction Using Machine Learning", is a web-based application that allows a user to select their symptoms from a structured list and receive a ranked prediction of the most probable diseases, along with descriptions and recommended precautions. The system is built using Python, Flask, and scikit-learn, and is intended purely as an educational demonstration of how supervised classification algorithms can be applied to healthcare data.

---

## 2. Problem Statement

Many people experience mild symptoms and are unsure whether they require medical attention, what condition they might have, or what precautions they should take in the meantime. Searching the internet for symptoms often leads to inconsistent, alarming, or unreliable information.

The problem this project addresses is: "Can a machine learning model, trained on a structured symptom-disease dataset, accurately predict the most likely disease(s) based on a set of user-selected symptoms, and present this information along with relevant descriptions and precautions in an accessible web interface?"

---

## 3. Objectives

The key objectives of this project are as follows:

- To collect and preprocess a labelled dataset mapping symptoms to diseases.
- To train and evaluate multiple machine learning classification models on the dataset.
- To identify the best-performing model for predicting diseases from symptoms.
- To develop a Flask-based web application that allows users to select symptoms interactively.
- To display the top 3 predicted diseases along with confidence percentages.
- To provide descriptions and precautionary measures for each predicted disease.
- To design a clean, responsive, and user-friendly interface for the application.

---

## 4. Literature Review

Several studies have explored the application of machine learning in disease diagnosis and prediction. Classification algorithms such as Decision Trees, Random Forests, Naive Bayes, and K-Nearest Neighbours (KNN) have been widely used for medical diagnosis tasks due to their interpretability and reasonable performance on structured, tabular data.

Random Forest, an ensemble of multiple decision trees, is particularly popular in healthcare-related classification tasks because it reduces overfitting compared to a single decision tree and provides feature importance, which can help explain which symptoms most strongly influence a prediction. Naive Bayes is valued for its simplicity and speed, especially with categorical or binary features, while KNN is useful for capturing local similarity patterns between patient records.

Existing disease prediction systems, including several open-source projects on platforms such as Kaggle and GitHub, commonly use the symptom-disease dataset adopted in this project as a benchmark for demonstrating multi-class classification. This project builds on these established approaches by combining model comparison with a complete, functional web interface and supplementary information (descriptions and precautions) for each predicted disease, which is often missing from purely model-focused implementations.

---

## 5. Dataset Description

### 5.1 Primary Dataset

The primary dataset used for training and testing the models is the "Disease Prediction Using Machine Learning" dataset, sourced from Kaggle. It consists of two CSV files:

- **Training.csv** – used to train the classification models
- **Testing.csv** – used to evaluate model performance on unseen data

Each file contains 133 columns: 132 binary columns representing the presence (1) or absence (0) of a specific symptom, and one final column named 'prognosis' representing the diagnosed disease. The dataset covers 42 distinct diseases.

### 5.2 Supplementary Dataset

To enrich the prediction results with additional medical context, a supplementary dataset – the "Disease Symptom Description Dataset", also from Kaggle – was used. This dataset provides two additional CSV files:

- **symptom_description.csv** – contains a short textual description for each disease
- **symptom_precaution.csv** – contains up to four recommended precautions for each disease

These files are used by the Flask backend to display relevant information alongside each predicted disease, making the results more informative and actionable for the user.

### 5.3 Dataset Sources

| File | Source |
|---|---|
| Training.csv, Testing.csv | [kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| symptom_description.csv, symptom_precaution.csv | [kaggle.com/datasets/itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |

---

## 6. System Architecture

The application follows a simple three-tier architecture consisting of a presentation layer (HTML/CSS frontend), an application layer (Flask backend), and a data/model layer (CSV datasets and trained ML models).

### 6.1 Architecture Overview

- **Frontend (Templates + Static):** `index.html` allows users to search and select symptoms via checkboxes; `result.html` displays the prediction results, descriptions, precautions, and a model comparison table. Styling is handled via a custom stylesheet (`style.css`).
- **Backend (Flask – app.py):** Handles routing, loads the trained models and supporting data at startup, converts selected symptoms into a binary input vector, runs predictions, and renders the results page.
- **Model Layer (models/):** Contains the trained and serialized (.pkl) models – Random Forest, Decision Tree, Naive Bayes, and KNN – along with the symptom column list and disease list, all generated by `model_train.py`.
- **Data Layer (dataset/):** Contains the raw CSV files – `Training.csv`, `Testing.csv`, `symptom_description.csv`, and `symptom_precaution.csv`.

### 6.2 Request Flow

1. The user opens the home page, which displays all 132 symptoms as a searchable checkbox grid.
2. The user selects relevant symptoms and submits the form via a POST request to `/predict`.
3. The Flask backend converts the selected symptoms into a 132-element binary vector (1 = present, 0 = absent).
4. The Random Forest model (primary model) computes prediction probabilities for all 42 diseases.
5. The top 3 diseases by confidence are selected, along with their descriptions and precautions.
6. The other three models (Decision Tree, Naive Bayes, KNN) each generate their own top prediction for comparison.
7. The results page (`result.html`) is rendered, displaying ranked predictions, confidence bars, descriptions, precautions, and a model comparison table.

---

## 7. Machine Learning Models

Four supervised classification algorithms were trained and evaluated on the dataset. Random Forest was selected as the primary model for generating the top 3 ranked predictions with confidence scores, while the remaining three models are used for academic comparison to demonstrate how different algorithms perform on the same data.

### 7.1 Models Used

| Model | Description | Role |
|---|---|---|
| Random Forest | An ensemble of decision trees that votes on the most likely class, reducing overfitting and improving generalisation. | Primary model |
| Decision Tree | A single tree-based classifier that splits data based on symptom presence to reach a disease prediction. | Comparison |
| Naive Bayes | A probabilistic classifier based on Bayes' theorem, assuming feature independence between symptoms. | Comparison |
| K-Nearest Neighbours | Classifies a case based on the majority class among its 'k' most similar training samples. | Comparison |

### 7.2 Model Performance

All four models were evaluated on the held-out `Testing.csv` dataset using accuracy as the primary metric. Given the clean and well-separated nature of the dataset, all four models achieved high accuracy, with Random Forest selected as the primary model due to its robustness and ability to provide reliable confidence scores via `predict_proba()`.

| Model | Test Accuracy |
|---|:---:|
| Random Forest | ~95–100% |
| Decision Tree | ~95–100% |
| Naive Bayes | ~90–100% |
| K-Nearest Neighbours | ~90–100% |

*Note: Exact percentages may vary slightly depending on the train/test split and random state used during training.*

---

## 8. Development Phases

The project was developed in five structured phases:

### Phase 1 – Project Setup

Established the folder structure (`dataset`, `models`, `notebooks`, `static/css`, `templates`) and configured `requirements.txt` with all required dependencies (Flask, NumPy, pandas, scikit-learn, joblib).

### Phase 2 – Model Training

Developed `model_train.py`, which loads `Training.csv` and `Testing.csv`, separates the 132 symptom features from the prognosis label, trains all four models, evaluates their accuracy on the test set, and saves the trained models, symptom column list, and disease list to the `models/` folder using joblib.

### Phase 3 – Flask Backend

Developed `app.py`, which loads the trained models and supplementary CSV files at startup. It defines the home route (`/`) which passes the 132 symptoms to the frontend, the `/predict` route which processes the selected symptoms and returns the top 3 predictions with descriptions and precautions, and an `/api/predict` route for JSON-based testing.

### Phase 4 – Frontend Development

Built `index.html` with a searchable, checkbox-based symptom selection grid, live symptom pills, and a selection counter. Built `result.html` to display the top 3 predictions as ranked cards with animated confidence bars, expandable description and precaution sections, and a model comparison table. Designed `style.css` using a navy-and-teal medical theme with the DM Serif Display and DM Sans fonts for a professional, modern appearance.

### Phase 5 – Testing, Polish & Documentation

Performed end-to-end testing of the application by running the Flask server and verifying that symptom selection, prediction, and result display function correctly without errors. Finalised UI polish and prepared this project report for academic submission and presentation.

---

## 9. Scope and Limitations

### 9.1 Scope

- Provides an interactive web interface for selecting symptoms and viewing predicted diseases.
- Demonstrates comparison between four different machine learning classification algorithms.
- Supplies descriptive information and precautions for each predicted disease.
- Serves as a foundation that can be extended with additional features such as user accounts, prediction history, or a larger dataset.

### 9.2 Limitations

- The dataset is limited to 132 symptoms and 42 diseases, and may not reflect the full complexity of real-world medical diagnosis.
- The system does not account for symptom severity, duration, patient age, gender, or medical history.
- Predictions are based purely on statistical patterns in the training data and should not be used as a substitute for professional medical advice.
- The model's accuracy depends heavily on the quality and completeness of the underlying dataset.

---

## 10. Conclusion

This project successfully demonstrates the application of machine learning to the problem of disease prediction based on symptoms. By training and comparing four classification algorithms – Random Forest, Decision Tree, Naive Bayes, and K-Nearest Neighbours – on a structured symptom-disease dataset, the system is able to identify the most probable diseases with high accuracy on the test data.

The Flask-based web application provides an accessible interface where users can select their symptoms and instantly receive the top 3 predicted diseases, along with confidence scores, descriptions, and precautionary measures. While the system is intended strictly for educational purposes and is not a replacement for professional medical consultation, it illustrates how machine learning techniques can be integrated into a complete, functional web application – from data preprocessing and model training to backend logic and frontend design.

Future enhancements could include expanding the dataset to cover more diseases and symptoms, incorporating patient-specific factors such as age and medical history, and deploying the application to a cloud platform for wider accessibility.

---

## 11. References

1. Kaushil268. "Disease Prediction Using Machine Learning." Kaggle. Available at: [kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
2. Itachi9604. "Disease Symptom Description Dataset." Kaggle. Available at: [kaggle.com/datasets/itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
3. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 2011.
4. Flask Documentation. Available at: [flask.palletsprojects.com](https://flask.palletsprojects.com)
5. Project Repository: [github.com/SumanThokdarProg/DiseasePredictionUsingML](https://github.com/SumanThokdarProg/DiseasePredictionUsingML)
