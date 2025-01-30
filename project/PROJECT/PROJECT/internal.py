

##import pandas as pd
##
### Specify the Excel file path
##excel_file_path = 'SEM_2_MARKS.xlsx'
##
### Read the Excel file
##data = pd.read_excel(excel_file_path)
##
### Specify the output CSV file path
##csv_file_path = 'student_marks.csv'
##
### Save the data to a CSV file
##data.to_csv(csv_file_path, index=False)
##
##print(f"Excel file '{excel_file_path}' has been converted to CSV '{csv_file_path}'.")







##import pandas as pd
##
### Specify the Excel file path
##excel_file_path = 'SEM_2_MARKS.xlsx'
##
### Load the Excel file and print sheet names
##excel_file = pd.ExcelFile(excel_file_path)
##print("Available sheet names:", excel_file.sheet_names)
##
### Specify the sheet name to read (or use a loop to extract all sheets)
##sheet_name = 'AIML'  # Replace with the desired sheet name or dynamically iterate
##
### Read the specified sheet
##data = excel_file.parse(sheet_name)
##
### Display a preview of the data
##print(f"Data from sheet '{sheet_name}':")
##print(data.head())
##
### Specify the output CSV file path
##csv_file_path = 'student_marks.csv'
##
### Save the data from the specified sheet to a CSV file
##data.to_csv(csv_file_path, index=False)
##
##print(f"Sheet '{sheet_name}' from Excel file '{excel_file_path}' has been converted to CSV '{csv_file_path}'.")










##import numpy as np
##import pandas as pd
##from sklearn.model_selection import train_test_split
##from sklearn.preprocessing import LabelEncoder, MinMaxScaler
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import LSTM, Dense
##from tensorflow.keras.utils import to_categorical
##
### Load the CSV file
##data = pd.read_csv('student_marks.csv')
##
### Preprocessing
### Extract features and target
##features = data[['Internals1','Internal 2','Internal 3','Lab Marks','Assignment',
##                 'Seminar(Discipline)','Seminar(Report)','Seminar(Presentation)']].values
##feedback = data['Overall Feedback'].values
##
### Normalize features
##scaler = MinMaxScaler()
##features = scaler.fit_transform(features)
##
### Encode feedback labels
##label_encoder = LabelEncoder()
##feedback_encoded = label_encoder.fit_transform(feedback)
##feedback_encoded = to_categorical(feedback_encoded)
##
### Split data into training and testing sets
##X_train, X_test, y_train, y_test = train_test_split(features, feedback_encoded, test_size=0.2, random_state=42)
##
### Reshape data for LSTM (samples, timesteps, features)
##X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
##X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
##
### Build LSTM model
##model = Sequential()
##model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
##model.add(Dense(20, activation='relu'))
##model.add(Dense(y_train.shape[1], activation='softmax'))
##
### Compile the model
##model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
##
### Train the model
##model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
##
### Evaluate the model
##loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
##print(f'Model Accuracy: {accuracy * 100:.2f}%')
##
### Predict and generate feedback for new data
##def generate_feedback(new_data):
##    new_data = scaler.transform(new_data.reshape(1, -1))
##    new_data = new_data.reshape((new_data.shape[0], 1, new_data.shape[1]))
##    prediction = model.predict(new_data)
##    feedback_label = label_encoder.inverse_transform([np.argmax(prediction)])
##    return feedback_label[0]
##
### Example input for prediction
##new_student_marks = np.array([19,20,20,11,10,6,5,12])  # Example marks
##
##feedback = generate_feedback(new_student_marks)
##print(f'Generated Feedback: {feedback}')










##import pandas as pd
##import numpy as np
##
##from sklearn.model_selection import train_test_split
##from sklearn.preprocessing import LabelEncoder, MinMaxScaler
##from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
##
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import LSTM, Dense
##from tensorflow.keras.utils import to_categorical
##
##
##
##
### Function to extract data from an Excel sheet
##def extract_sheet_and_save_as_csv(excel_file, sheet_name, output_csv):
##    # Read the specified sheet from the Excel file
##    data = pd.read_excel(excel_file, sheet_name=sheet_name)
##    # Save the sheet data to a CSV file
##    data.to_csv(output_csv, index=False)
##    return output_csv
##
### Input Excel file and sheet name
##excel_file = 'SEM_2_MARKS.xlsx'  # Example Excel file name
##sheet_name = input("Enter the Subject name: ")  # User input for sheet name
##csv_file = extract_sheet_and_save_as_csv(excel_file, sheet_name, 'student_marks.csv')
##
### Load the CSV file
##data = pd.read_csv(csv_file)
##
### Preprocessing
### Extract features and target
##features = data[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 'Assignment',
##                 'Seminar(Discipline)', 'Seminar(Report)', 'Seminar(Presentation)']].values
##feedback = data['Overall Feedback'].values
##
### Normalize features
##scaler = MinMaxScaler()
##features = scaler.fit_transform(features)
##
### Encode feedback labels
##label_encoder = LabelEncoder()
##feedback_encoded = label_encoder.fit_transform(feedback)
##feedback_encoded = to_categorical(feedback_encoded)
##
### Split data into training and testing sets
##X_train, X_test, y_train, y_test = train_test_split(features, feedback_encoded, test_size=0.2, random_state=42)
##
### Reshape data for LSTM (samples, timesteps, features)
##X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
##X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
##
### Build LSTM model
##model = Sequential()
##model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
##model.add(Dense(20, activation='relu'))
##model.add(Dense(y_train.shape[1], activation='softmax'))
##
### Compile the model
##model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
##
### Train the model
##model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
##
### Evaluate the model
##loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
##print(f'Model Accuracy: {accuracy * 100:.2f}%')
##
##
##
##
##
##### Predict and generate feedback for new data
####def generate_feedback(new_data):
####    new_data = scaler.transform(new_data.reshape(1, -1))
####    new_data = new_data.reshape((new_data.shape[0], 1, new_data.shape[1]))
####    prediction = model.predict(new_data)
####    feedback_label = label_encoder.inverse_transform([np.argmax(prediction)])
####    return feedback_label[0]
####    return prediction
####    return new_data 
####
##### Example input for prediction
####new_student_marks = np.array([11, 13, 10, 7, 6, 2, 4, 8])  # Example marks
####
####feedback = generate_feedback(new_student_marks)
####print(f'Generated Feedback: {feedback}')
##
##
##
##
##
##
##
### Function to fetch marks by USN
##def fetch_marks_by_usn(usn, data):
##    try:
##        row = data[data['USN'] == usn]
##        if row.empty:
##            raise ValueError(f"No data found for USN: {usn}")
##        marks = row[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 
##                     'Assignment', 'Seminar(Discipline)', 'Seminar(Report)', 
##                     'Seminar(Presentation)']].values.flatten()
##        return marks
##    except Exception as e:
##        print(e)
##        return None
##
##
##
##
##
##
##
##
### Predict and generate feedback for a specific USN
##def generate_feedback_by_usn(usn):
##    marks = fetch_marks_by_usn(usn, data)
##    if marks is None:
##        return "Unable to generate feedback."
##    marks = scaler.transform(marks.reshape(1, -1))
##    marks = marks.reshape((marks.shape[0], 1, marks.shape[1]))
##    prediction = model.predict(marks)
##    feedback_label = label_encoder.inverse_transform([np.argmax(prediction)])
##    return feedback_label[0]
##
##
##
##
##usn = input("Enter the USN: ")  # User input for USN
##feedback,marks,prediction = generate_feedback_by_usn(usn)
##print(f'Generated Feedback for USN {usn}: {feedback}')







##import pandas as pd
##import numpy as np
##
##from sklearn.model_selection import train_test_split
##from sklearn.preprocessing import LabelEncoder, MinMaxScaler
##from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
##
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import LSTM, Dense
##from tensorflow.keras.utils import to_categorical
##import matplotlib.pyplot as plt
##
### Function to extract data from an Excel sheet
##def extract_sheet_and_save_as_csv(excel_file, sheet_name, output_csv):
##    # Read the specified sheet from the Excel file
##    data = pd.read_excel(excel_file, sheet_name=sheet_name)
##    # Save the sheet data to a CSV file
##    data.to_csv(output_csv, index=False)
##    return output_csv
##
### Input Excel file and sheet name
##excel_file = 'SEM_2_MARKS.xlsx'  # Example Excel file name
##sheet_name = input("Enter the Subject name: ")  # User input for sheet name
##csv_file = extract_sheet_and_save_as_csv(excel_file, sheet_name, 'student_marks.csv')
##
### Load the CSV file
##data = pd.read_csv(csv_file)
##
### Preprocessing
### Extract features and target
##features = data[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 'Assignment',
##                 'Seminar(Discipline)', 'Seminar(Report)', 'Seminar(Presentation)']].values
##feedback = data['Overall Feedback'].values
##
### Normalize features
##scaler = MinMaxScaler()
##features = scaler.fit_transform(features)
##
### Encode feedback labels
##label_encoder = LabelEncoder()
##feedback_encoded = label_encoder.fit_transform(feedback)
##feedback_encoded = to_categorical(feedback_encoded)
##
### Split data into training and testing sets
##X_train, X_test, y_train, y_test = train_test_split(features, feedback_encoded, test_size=0.2, random_state=42)
##
### Reshape data for LSTM (samples, timesteps, features)
##X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
##X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
##
### Build LSTM model
##model = Sequential()
##model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
##model.add(Dense(20, activation='relu'))
##model.add(Dense(y_train.shape[1], activation='softmax'))
##
### Compile the model
##model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
##
### Train the model
##model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
##
### Evaluate the model
##loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
##print(f'Model Accuracy: {accuracy * 100:.2f}%')
##
### Generate predictions
##predictions = model.predict(X_test)
##predicted_classes = np.argmax(predictions, axis=1)
##true_classes = np.argmax(y_test, axis=1)
##
### Confusion matrix and classification report
##conf_matrix = confusion_matrix(true_classes, predicted_classes)
##print("Confusion Matrix:")
##print(conf_matrix)
##
##
##
##
##
### Function to fetch marks by USN
##def fetch_marks_by_usn(usn, data):
##    try:
##        row = data[data['USN'] == usn]
##        if row.empty:
##            raise ValueError(f"No data found for USN: {usn}")
##        marks = row[['Internal 1', 'Internal 2', 'Internal 3', 'Lab Marks', 
##                     'Assignment', 'Seminar(Discipline)', 'Seminar(Report)', 
##                     'Seminar(Presentation)']].values.flatten()
##        return marks
##    except Exception as e:
##        print(e)
##        return None
##
### Predict and generate feedback for a specific USN
##def generate_feedback_by_usn(usn):
##    marks = fetch_marks_by_usn(usn, data)
##    if marks is None:
##        return "Unable to generate feedback."
##    marks = scaler.transform(marks.reshape(1, -1))
##    marks = marks.reshape((marks.shape[0], 1, marks.shape[1]))
##    prediction = model.predict(marks)
##    feedback_label = label_encoder.inverse_transform([np.argmax(prediction)])
##    return feedback_label[0]
##
##usn = input("Enter the USN: ")  # User input for USN
##feedback = generate_feedback_by_usn(usn)
##print(f'Generated Feedback for USN {usn}: {feedback}')








############------------FINAL CODE

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, feedback_encoded, test_size=0.2, random_state=42)

# Reshape data for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
model.add(Dense(20, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Generate predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)



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
