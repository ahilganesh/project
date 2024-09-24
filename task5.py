# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_and_clean_data(file_path):
    """
    Load and clean the heart disease dataset.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Display summary statistics
    print(df.describe())

    return df


def exploratory_data_analysis(df):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Age distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['age'], kde=True)
    plt.title('Age Distribution')
    plt.show()

    # Cholesterol levels vs target
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='chol', data=df)
    plt.title('Cholesterol Levels vs Heart Disease')
    plt.show()

    # Chest Pain Type (cp) vs Heart Disease
    plt.figure(figsize=(8, 6))
    sns.countplot(x='cp', hue='target', data=df)
    plt.title('Chest Pain Type and Heart Disease')
    plt.show()


def preprocess_data(df):
    """
    Preprocess the data for model training.
    """
    # Define features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Build a Random Forest model and evaluate its performance.
    """
    # Build the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the results
    y_pred = model.predict(X_test)

    # Confusion Matrix and Classification Report
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualize Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    # Load and clean the dataset (set the correct path to your dataset)
    df = load_and_clean_data('heart.csv')

    # Perform EDA
    exploratory_data_analysis(df)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and evaluate the model
    build_and_evaluate_model(X_train, X_test, y_train, y_test)
