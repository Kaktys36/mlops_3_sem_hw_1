import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Function to load the dataset (you can modify this based on your structure)
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to train the logistic regression model
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Test the data loading function
def test_load_data():
    file_path = '/app/creditdefault.csv'
    data = load_data(file_path)
    assert not data.empty, "DataFrame should not be empty"
    assert 'Income' in data.columns, "'Income' column should be in the DataFrame"
    assert 'Default' in data.columns, "'Default' column should be in the DataFrame"


# Test the preprocessing steps
def test_preprocessing():
    # Load data
    file_path = '/app/creditdefault.csv'
    data = load_data(file_path)

    # Check for missing values
    assert data.isnull().sum().sum() == 0, "There should be no missing values in the dataset"

    # Check that target variable is binary
    assert set(data['Default'].unique()) == {0, 1}, "Target variable 'Default' should be binary (0 or 1)"


# Test the model training function
def test_train_model():
    # Load data
    file_path = '/app/creditdefault.csv'
    data = load_data(file_path)

    # Preprocess data
    X = data[['Income', 'Age', 'Loan']]
    y = data['Default']

    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), "Model should be an instance of LogisticRegression"


# Test the model accuracy
def test_model_accuracy():
    # Load data
    file_path ='/app/creditdefault.csv'
    data = load_data(file_path)

    # Preprocess data
    X = data[['Income', 'Age', 'Loan']]
    y = data['Default']

    # Split data into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.80, "Model accuracy should be greater than 80%"
    
