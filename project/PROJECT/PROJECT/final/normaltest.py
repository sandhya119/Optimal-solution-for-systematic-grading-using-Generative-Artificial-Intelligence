
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Load the saved model
model = load_model('NORMAL.h5')
print("Model loaded successfully.")


# Function to extract data from an Excel sheet
def extract_sheet_and_save_as_csv(excel_file, sheet_name, output_csv):
    # Read the specified sheet from the Excel file
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    # Save the sheet data to a CSV file
    data.to_csv(output_csv, index=False)
    return output_csv

# Input Excel file and sheet name
excel_file = 'SEM_2_MARKS.xlsx'  # Example Excel file name
sheet_name = input("Enter the Subject name: ")  # User input for sheet name
csv_file = extract_sheet_and_save_as_csv(excel_file, sheet_name, 'student_marks.csv')

# Load the CSV file
data = pd.read_csv(csv_file)

# Preprocessing
# Extract features and target
features = data[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 'Assignment',
                 'Seminar(Discipline)', 'Seminar(Report)', 'Seminar(Presentation)']].values
feedback = data['Overall Feedback'].values

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Encode feedback labels
label_encoder = LabelEncoder()
feedback_encoded = label_encoder.fit_transform(feedback)
feedback_encoded = to_categorical(feedback_encoded)


# Function to fetch marks by USN
def fetch_marks_by_usn(usn, data):
    try:
        row = data[data['USN'] == usn]
        if row.empty:
            raise ValueError(f"No data found for USN: {usn}")
        marks = row[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 
                     'Assignment', 'Seminar(Discipline)', 'Seminar(Report)', 
                     'Seminar(Presentation)']].values.flatten()
        return marks
    except Exception as e:
        print(e)
        return None

# Predict and generate feedback for a specific USN
def generate_feedback_by_usn(usn):
    marks = fetch_marks_by_usn(usn, data)
    if marks is None:
        return "Unable to generate feedback."
    marks = scaler.transform(marks.reshape(1, -1))
    marks = marks.reshape((marks.shape[0], 1, marks.shape[1]))
    prediction = model.predict(marks)
    feedback_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return feedback_label[0]

usn = input("Enter the USN: ")  # User input for USN
feedback = generate_feedback_by_usn(usn)
print(f'Generated Feedback for USN {usn}: {feedback}')
