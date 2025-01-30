import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Convert Excel to CSV
def convert_excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)
    print(f"Excel file '{excel_file}' converted to CSV '{csv_file}'.")

# Step 2: Read Marks and Grades from CSV
def read_marks_and_grades(csv_file):
    df = pd.read_csv(csv_file)
    # Merge all marks columns into a single array
    marks_columns = ['AIML(Marks)', 'ADVANCE JAVA(Marks)', 'DSA(Marks)', 'C++(Marks)', 'SQL(Marks)']
    marks = df[marks_columns].values.flatten()  # Flatten the marks into a single array
    
    # Merge all grade columns into a single array
    grades_columns = ['AIML(Grade)', 'ADVANCE JAVA(Grade)', 'DSA(Grade)', 'C++(Grade)', 'SQL(Grade)']
    grades = df[grades_columns].values.flatten()  # Flatten the grades into a single array
 
    print("Marks and Grades Data:")
    print(df.head())
    return marks, grades

# Step 3: Preprocess Data
def preprocess_data(marks, grades):
    # Scale marks to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    marks_scaled = scaler.fit_transform(marks.reshape(-1, 1))

    # Encode grades as integers
    label_encoder = LabelEncoder()
    grades_encoded = label_encoder.fit_transform(grades)

    # Reshape marks into sequences for LSTM
    sequence_length = 5  # You can adjust this based on your dataset
    X = []
    y = []

    for i in range(sequence_length, len(marks_scaled)):
        X.append(marks_scaled[i-sequence_length:i])
        y.append(grades_encoded[i])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to be compatible with LSTM input
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print("Data preprocessing complete.")
    return X, y, scaler, label_encoder

# Step 4: Train the LSTM Model
def train_lstm_model(X, y):
    # Build the LSTM model
    model = Sequential()
    
    # Add LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))

    # Add another LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Add output layer
    model.add(Dense(units=len(np.unique(y)), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)
    print("Model training complete.")
    return model

# Step 5: Generate Grade from Input Mark
def predict_grade(model, marks_input, scaler, label_encoder, sequence_length=5):
    # Scale the input mark
    marks_scaled_input = scaler.transform(np.array([[marks_input]]))
    
    # Create a sequence with the last 10 marks for prediction (use the same mark repeated if it's a single input)
    marks_scaled_input = np.tile(marks_scaled_input, (sequence_length, 1))
    
    # Reshape to the required input shape for LSTM
    X_input = marks_scaled_input.reshape((1, sequence_length, 1))  # Shape (1, 10, 1)
    
    # Predict the grade using the trained LSTM model
    predicted_grade_encoded = model.predict(X_input)
    
    # Get the grade corresponding to the predicted label
    predicted_grade = label_encoder.inverse_transform([np.argmax(predicted_grade_encoded)])
    
    return predicted_grade[0]



# File paths (replace with your actual file paths)
excel_file = 'SEM_2_MARKS.xlsx'  # Your input Excel file
csv_file = 'marks_data.csv'     # Output CSV file

# Step 1: Convert Excel to CSV
convert_excel_to_csv(excel_file, csv_file)

# Step 2: Read Marks and Grades
marks, grades = read_marks_and_grades(csv_file)

# Step 3: Preprocess Data
X, y, scaler, label_encoder = preprocess_data(marks, grades)

# Step 4: Train the LSTM Model
model = train_lstm_model(X, y)

# Step 5: Predict Grade based on input mark
# Example usage
input_mark = input("Enter the SEM MARK: ") 

predicted_grade = predict_grade(model, input_mark, scaler, label_encoder)
print(f"Predicted Grade for {input_mark} marks: {predicted_grade}")


