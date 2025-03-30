import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_excel(r'C:\Users\agraw\OneDrive\Desktop\project\dss\School_Dropout_Dataset.xlsx')

# Handling missing values
data.fillna(data.median(numeric_only=True), inplace=True)  # Impute missing numeric values with median

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Define features and target
features = [
    'Age', 'Gender', 'Socioeconomic_Status', 'Parental_Education',
    'Primary_Language', 'Previous_Grades', 'Current_Grades',
    'Attendance_Rate', 'Homework_Completion_Rate', 
    'Learning_Disabilities', 'Course_Failures', 'Disciplinary_Actions',
    'Participation_in_Extracurriculars', 'Family_Support_Score',
    'Parental_Involvement', 'Home_Internet_Access', 
    'Teacher_Student_Ratio', 'Mental_Health_Support', 
    'Bullying_Incidents', 'Chronic_Absenteeism', 
    'Tutoring_Services_Used'
]
target = 'Dropout_Status'

# Define features and target variables
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")


# Adjust prediction threshold
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (at risk)
threshold = 0.3  # Adjust this threshold as needed
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)


# Students at risk with adjusted predictions
risk_students_adjusted = X_test.copy()
risk_students_adjusted['Predicted_Dropout_Status'] = y_pred_adjusted
risk_students_adjusted['Student_ID'] = data.loc[X_test.index, 'Student_ID']

# Filter students predicted to drop out with adjusted predictions
at_risk_students_adjusted = risk_students_adjusted[risk_students_adjusted['Predicted_Dropout_Status'] == 1]

if at_risk_students_adjusted.empty:
    print("No students predicted to be at risk of dropping out with adjusted threshold.")
else:
    print("\nStudents at Risk of Dropping Out:")
    print(at_risk_students_adjusted[['Student_ID', 'Predicted_Dropout_Status']])



# Visualizing certain features of at-risk students
plt.figure(figsize=(12, 6))
sns.histplot(at_risk_students_adjusted['Current_Grades'], bins=10, kde=True)
plt.title('Distribution of Current Grades for At-Risk Students')
plt.xlabel('Current Grades')
plt.ylabel('Frequency')
plt.show()




# Calculate the total number of students in the test set
total_students = len(X_test)

# Calculate the number of students at risk
number_at_risk = len(at_risk_students_adjusted)

# Calculate percentage
percentage_at_risk = (number_at_risk / total_students) * 100

print(f"Percentage of Students at Risk of Dropping Out: {percentage_at_risk:.2f}%")