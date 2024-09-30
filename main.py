import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()

train_file_path = askopenfilename(title="Select the training CSV file", filetypes=[("CSV files", "*.csv")])

if not train_file_path:
    print("No training file selected. Exiting.")
    exit()
    
train_data = pd.read_csv(train_file_path)

print(train_data.head())
print("\n")
print(train_data.describe())

# function for data preprocessing
def preprocess_data(data):
    # Drop unnecessary columns
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    # Fill missing values in 'Age', 'Fare', and 'Embarked'
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    
    return data

# calling preprocess_data function
train_data = preprocess_data(train_data)

print("\n")
print("close the visualise window after viewing")
print("\n")

#Data Visualisation

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Plot 1: Survival Count
survived_counts = train_data['Survived'].value_counts()
axes[0, 0].bar(survived_counts.index, survived_counts.values, tick_label=['Did not Survive', 'Survived'])
axes[0, 0].set_title('Survival Count')

# Plot 2: Survival by Class
class_survival = train_data.groupby(['Pclass', 'Survived']).size().unstack()
class_survival.plot(kind='bar', stacked=True, ax=axes[0, 1])
axes[0, 1].set_title('Survival by Class')
axes[0, 1].set_xlabel('Pclass')
axes[0, 1].set_xticklabels(['Class 1', 'Class 2', 'Class 3'], rotation=0)
axes[0, 1].legend(['Did not Survive', 'Survived'])

# Plot 3: Age Distribution by Class
boxplot_data = [train_data[train_data['Pclass'] == i]['Age'].dropna() for i in range(1, 4)]
axes[1, 0].boxplot(boxplot_data, labels=['Class 1', 'Class 2', 'Class 3'])
axes[1, 0].set_title('Age Distribution by Class')
axes[1, 0].set_xlabel('Pclass')
axes[1, 0].set_ylabel('Age')

# Plot 4: Survival by Embarked
embarked_survival = train_data.groupby(['Embarked', 'Survived']).size().unstack()
embarked_survival.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_title('Survival by Embarked')
axes[1, 1].set_xlabel('Embarked')
axes[1, 1].set_xticklabels(embarked_survival.index, rotation=0)
axes[1, 1].legend(['Did not Survive', 'Survived'])

plt.tight_layout()
plt.show()


# Split the data into features (X) and the target variable (y)
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

# Ask the user to select the test CSV file
test_file_path = askopenfilename(title="Select the test CSV file", filetypes=[("CSV files", "*.csv")])

if not test_file_path:
    print("No test file selected. Exiting.")
    exit()

test_data = pd.read_csv(test_file_path)

passenger_ids = test_data['PassengerId']

# Preprocess the test data
test_data = preprocess_data(test_data)

test_predictions = clf.predict(test_data)

# Save the predictions to a CSV file
result_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_predictions})
result_df.to_csv('titanic_survival_predictions.csv', index=False)
