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
print(f'Accuracy: {accuracy}%')

# Predict new data (replace new_data with your actual data)
new_data = [[25, 50000, 1, 2.0, 3, 1, 15000, 12.0, 1, 0.3, 5]]
new_predictions = test_model(summaries, new_data)
predictionResult = ''

if (new_predictions[0] == 0):
    predictionResult = "No Default"
elif (new_predictions[0] == 1):
    predictionResult = "Defaulted"

print(f"Predicted default result: {predictionResult}")

# print(f'Prediction for new data: {new_predictions[0]}')
