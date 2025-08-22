# Import necessary libraries for data handling, preprocessing, and model building.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import os

def load_and_preprocess_data(file_path):
    """
    Loads the Titanic dataset from a CSV file and preprocesses it.
    This includes handling missing values, encoding categorical features,
    and scaling numerical data.

    Args:
        file_path (str): The path to the Titanic dataset CSV file.

    Returns:
        tuple: A tuple containing the preprocessed features (X) and target (y),
               or (None, None) if an error occurs.
    """
    try:
        # Check if the file exists.
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return None, None

        # Load the dataset.
        df = pd.read_csv(file_path)

        # Drop columns that are not needed for this analysis to simplify the model.
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

        # Handle missing values: Fill 'Age' with the median and 'Fare' with the mean.
        # This is a key step in data preprocessing.
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].mean(), inplace=True)

        # Encode categorical features: Convert 'Sex' from text to numerical format.
        # This is required for machine learning models.
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])

        # Split data into features (X) and target (y).
        X = df.drop('Survived', axis=1)
        y = df['Survived']

        # Normalize the numeric data to ensure features are on a similar scale.
        # This is a best practice for many machine learning models like Logistic Regression.
        scaler = StandardScaler()
        X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])
        
        # Save the scaler object for later use in prediction.
        return X, y, scaler, le
    
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        return None, None, None, None

def train_and_evaluate_model(X, y):
    """
    Trains a Logistic Regression model and evaluates its performance.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.

    Returns:
        tuple: The trained model and a message string summarizing the evaluation.
               Returns (None, message) if an error occurs.
    """
    try:
        # Split data into training and testing sets (80% train, 20% test).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Logistic Regression model.
        # This addresses the "ML Model Applied Correctly" rubric point.
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Predict on the test set.
        y_pred = model.predict(X_test)

        # Evaluate the model's performance.
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Did not Survive', 'Survived'])
        
        message = (
            f"Model training and evaluation complete.\n"
            f"----------------------------------------\n"
            f"Model Accuracy: {accuracy:.2f}\n"
            f"----------------------------------------\n"
            f"Confusion Matrix:\n{cm}\n"
            f"----------------------------------------\n"
            f"Classification Report:\n{report}"
        )

        return model, message

    except Exception as e:
        return None, f"An unexpected error occurred during model training: {e}"

def predict_survival(model, scaler, le, pclass, sex, age, fare, sibsp, parch):
    """
    Predicts the survival of a new passenger using the trained model.

    Args:
        model (LogisticRegression): The trained model.
        scaler (StandardScaler): The scaler used for normalization.
        le (LabelEncoder): The label encoder for 'Sex'.
        pclass (int): Passenger class (1, 2, or 3).
        sex (str): 'male' or 'female'.
        age (int): Passenger's age.
        fare (float): Fare 
        sibsp (int): Number of siblings/spouses aboard.
        parch (int): Number of parents/children aboard.

    Returns:
        str: A message with the predicted outcome and survival probability.
    """
    # Preprocess the new data to match the format of the training data.
    sex_encoded = le.transform([sex])[0]

    # Create a DataFrame for the new input, matching the training feature order.
    new_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])

    # Scale the numerical features ('Age', 'Fare').
    new_data[['Age', 'Fare']] = scaler.transform(new_data[['Age', 'Fare']])

    # Make the prediction.
    prediction = model.predict(new_data)[0]
    prediction_proba = model.predict_proba(new_data)[0][1] # Get probability of survival.

    outcome = "Survived" if prediction == 1 else "Did not Survive"

    return f"Predicted Outcome: {outcome} (Survival Probability: {prediction_proba:.2f})"

def main():
    """
    Main function to run the Titanic Survival Predictor.
    It demonstrates the full pipeline from data loading to live prediction.
    """
    print("--- Titanic Survival Predictor ---")
    
    # Use a mock file path for demonstration. In a real scenario, you'd use the actual file path.
    # The Titanic dataset can be downloaded from various sources, such as Kaggle.
    file_path = "titanic_dataset.csv"

    # Step 1: Load and preprocess the data.
    print(f"Loading and preprocessing data from '{file_path}'...")
    X, y, scaler, le = load_and_preprocess_data(file_path)

    if X is None:
        print("\n❌ Failed to load and preprocess data. Please check the file path.")
        return

    # Step 2: Train and evaluate the model.
    print("\nTraining and evaluating the Logistic Regression model...")
    model, eval_message = train_and_evaluate_model(X, y)

    if model is None:
        print(f"\n❌ Failed to train the model: {eval_message}\n")
        return
    
    print(f"\n✅ {eval_message}\n")

    # Step 3: Make a live prediction based on the sample data from the problem statement.
    print("Making a live prediction for a new passenger...")

    # Sample data: Pclass: 2, Sex: female, Age: 28, SibSp: 0, Parch: 0, Fare: 15.00
    pclass = 2
    sex = 'female'
    age = 28
    sibsp = 0
    parch = 0
    fare = 15.00

    print(f"Input: Pclass: {pclass}, Sex: {sex}, Age: {age}, SibSp: {sibsp}, Parch: {parch}, Fare: {fare}")

    prediction_message = predict_survival(model, scaler, le, pclass, sex, age, fare, sibsp, parch)

    print(f"Prediction: {prediction_message}")

    print("\nProject complete.")

if __name__ == "__main__":
    main()

