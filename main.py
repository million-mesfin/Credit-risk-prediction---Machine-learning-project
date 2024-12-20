
import pandas as pd
import math


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / \
        float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))
                 for column in zip(*dataset)]
    return summaries


def separate_by_class(dataset, labels):
    separated = {}
    for i in range(len(dataset)):
        label = labels[i]
        if label not in separated:
            separated[label] = []
        separated[label].append(dataset[i])
    return separated


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean)**2 / (2 * stdev**2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def predict(summaries, row):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(
                row[i], mean, stdev)
    return max(probabilities, key=probabilities.get)


def train_model(X_train, y_train):
    separated = separate_by_class(X_train, y_train)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def test_model(summaries, X_test):
    predictions = []
    for row in X_test:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions


def main():
    # Load the cleaned dataset
    file_path = 'credit_risk_dataset_cleaned.csv'  # Replace with your file path
    df = pd.read_csv(file_path)

    # Convert categorical variables to numerical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # Split data into features and target variable
    X = df.drop('cb_person_default_on_file', axis=1).values
    y = df['cb_person_default_on_file'].values

    # Split data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train the model
    summaries = train_model(X_train, y_train)

    # Test the model
    predictions = test_model(summaries, X_test)

    # Calculate accuracy
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1

    accuracy = correct / float(len(y_test)) * 100.0

    print("Enter the values for the following: \n")
    age = int(input("Age: "))
    income = int(input("Income: "))
    home_ownership = int(input("Home ownership (0 = Mortgage, 1 = Own, 2 = Rent): "))
    person_employment_lenght = int(input("Person employment lenght: "))
    loan_intent = int(input("Loan intent (DebtConsolidation = 1, Education = 2, HomeImprovement = 3, Mediacal = 4, Personal = 5, Venture = 6): "))
    loan_grade = int(input("Loan grade (0 = A, 1 = B, 2 = C, 3 = D, 4 = E, 5 = F, 6 = G): "))
    loan_amount = int(input("Loan amount: "))
    loan_interest_rate = float(input("Loan interest rate: "))
    loan_status = int(input("Loan status: "))
    loan_percent_income = float(input("Loan percent income: "))
    credit_history_length = int(input("Credit history length: "))

    # Predict new data (replace new_data with your actual data)
    new_data = [[age, income, home_ownership, person_employment_lenght, loan_intent, loan_grade, loan_amount, loan_interest_rate, loan_status, loan_percent_income, credit_history_length]]
    new_predictions = test_model(summaries, new_data)
    predictionResult = ''

    if (new_predictions[0] == 0):
        predictionResult = "No Default"
    elif (new_predictions[0] == 1):
        predictionResult = "Defaulted"


    print(f'Accuracy: {accuracy}%')
    print(f"Predicted default result: {predictionResult}")


if __name__ == '__main__':
    main()
