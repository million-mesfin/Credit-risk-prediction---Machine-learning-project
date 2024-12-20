# Naive Bayes Credit Risk Prediction

This project implements a Naive Bayes classifier to predict credit risk (whether a person will default on a loan) based on a cleaned dataset of credit applications.

## Project Structure

-   `naive.py`: Contains the core Naive Bayes algorithm and a basic implementation for training, testing, and making predictions.
-   `main.py`: Provides a more user-friendly interface by prompting the user to input the data to be predicted and includes the functionality of naive.py

-   `credit_risk_dataset_cleaned.csv`:  The cleaned dataset used for training and testing the model.

## Dependencies

-   **Python 3.6+**
-   **pandas**: Data manipulation library (`pip install pandas`)

## Dataset

The model uses a dataset named `credit_risk_dataset_cleaned.csv`, this dataset should be in the same directory as your Python files.

This dataset contains the following features:

- `person_age`: Age of the applicant
- `person_income`: Annual income of the applicant
- `person_home_ownership`: Housing situation of the applicant (Mortgage, Own, Rent)
- `person_emp_length`: Employment length of the applicant
- `loan_intent`: The purpose of the loan (Debt Consolidation, Education, Home Improvement, Medical, Personal, Venture)
- `loan_grade`: The grade of the loan (A, B, C, D, E, F, G)
- `loan_amnt`: The amount of the loan
- `loan_int_rate`: The interest rate of the loan
- `loan_status`: The status of the loan
- `loan_percent_income`: The percentage of the loan compared to the applicants income
- `cb_person_default_on_file`: The target variable (0 = No Default, 1 = Defaulted)

## Code Explanation

### `naive.py`

1.  **Data Loading and Preprocessing:**
    -   Loads the data from `credit_risk_dataset_cleaned.csv` using pandas.
    -   Converts categorical features (object type columns) to numerical representations using category codes.
    -   Splits data into features (X) and target variable (y).
    -   Divides the data into training (80%) and testing (20%) sets.

2.  **Naive Bayes Functions:**
    -   `mean(numbers)`: Calculates the mean of a list of numbers.
    -   `stdev(numbers)`: Calculates the standard deviation of a list of numbers.
    -   `summarize_dataset(dataset)`:  Calculates the mean, standard deviation, and count for each attribute in the given dataset.
    -   `separate_by_class(dataset, labels)`: Separates the dataset into groups based on the labels (target variable values).
    -   `calculate_probability(x, mean, stdev)`:  Calculates the probability of a given `x` value for a specific Gaussian distribution (mean and standard deviation).
    -   `predict(summaries, row)`: Predicts the class for a given data point `row` based on calculated probabilities for each class.
    -   `train_model(X_train, y_train)`: Trains the Naive Bayes model by summarizing the training data for each class.
    -   `test_model(summaries, X_test)`: Tests the model on the provided dataset and returns a list of predictions.

3.  **Model Training, Testing, and Prediction:**
    -   Trains the model using the training data with the function `train_model`.
    -   Tests the model using the test data with the function `test_model`.
    -   Calculates the accuracy of the model by comparing the predictions to the correct values.
    -   Prints the accuracy and the result of a new prediction.
   

### `main.py`

1.  **Data Loading and Preprocessing:**
    -   Loads the data from `credit_risk_dataset_cleaned.csv` using pandas.
    -   Converts categorical features (object type columns) to numerical representations using category codes.
    -   Splits data into features (X) and target variable (y).
    -   Divides the data into training (80%) and testing (20%) sets.

2.  **Naive Bayes Functions:**
    -   Implements the same Naive Bayes functions as `naive.py`.

3.  **User Input:**
    -   The code prompts the user to input the required data to be predicted.
  
4.  **Model Training, Testing, and Prediction:**
    -   Trains the model using the training data with the function `train_model`.
    -   Tests the model using the test data with the function `test_model`.
    -   Calculates the accuracy of the model by comparing the predictions to the correct values.
    -   Prints the accuracy and the result of a new prediction.

## How to Run the Code

1.  **Make sure you have the necessary dependencies installed.**
    -   Open a terminal or command prompt.
    -   Run: `pip install pandas`

2.  **Ensure that you have the file `credit_risk_dataset_cleaned.csv` in the same directory** as your Python files (`naive.py` and `main.py`).

3.  **To execute `naive.py`:**
    -   Open a terminal or command prompt.
    -   Navigate to the directory containing the Python file.
    -   Run: `python naive.py`
    -   The output will print the accuracy of the model with the test data. It will also make a prediction based on the provided new data, and display the result.

4.  **To execute `main.py`:**
    -   Open a terminal or command prompt.
    -   Navigate to the directory containing the Python file.
    -   Run: `python main.py`
    -   The program will ask for user input for the required data, and will print the accuracy of the model, followed by the prediction result.

## Important Notes

-   **Data Preprocessing:** The dataset must be cleaned and preprocessed before being used. Make sure that there are no missing values or non-numeric data in the data set. 
-   **Categorical Data Encoding:** In this project categorical data is converted to numeric, for more complex categorical data more sophisticated encoding methods can be used. 
-   **Model Evaluation:** Accuracy is used to evaluate the model in this example but other methods can be used depending on the need.
-   **Real-World Use:** This model is a simple implementation for educational purposes and can be used as a basic guide, not as a finished product in real-world scenarios.
