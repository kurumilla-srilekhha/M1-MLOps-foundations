from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def train_model():
    logging.info("Starting model training...")	
    # Loading the dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Training the  model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluating the model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc}")
    
    logging.info("Model training completed.")
    
    return model, acc


if __name__ == "__main__":
    train_model()
