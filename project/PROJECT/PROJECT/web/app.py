
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

app = Flask(__name__)

# Step 1: Convert Excel to CSV
def convert_excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)

# Step 2: Read Marks and Grades from CSV
def read_marks_and_grades(csv_file):
    df = pd.read_csv(csv_file)
    marks_columns = ['AIML(Marks)', 'ADVANCE JAVA(Marks)', 'DSA(Marks)', 'C++(Marks)', 'SQL(Marks)']
    marks = df[marks_columns].values.flatten()
    
    grades_columns = ['AIML(Grade)', 'ADVANCE JAVA(Grade)', 'DSA(Grade)', 'C++(Grade)', 'SQL(Grade)']
    grades = df[grades_columns].values.flatten()
    return marks, grades

# Step 3: Preprocess Data
def preprocess_data(marks, grades):
    scaler = MinMaxScaler(feature_range=(0, 1))
    marks_scaled = scaler.fit_transform(marks.reshape(-1, 1))

    label_encoder = LabelEncoder()
    grades_encoded = label_encoder.fit_transform(grades)

    sequence_length = 5
    X = []
    y = []
    for i in range(sequence_length, len(marks_scaled)):
        X.append(marks_scaled[i-sequence_length:i])
        y.append(grades_encoded[i])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler, label_encoder

# Step 4: Train the LSTM Model
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=len(np.unique(y)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=32)
    return model

# Step 5: Generate Grade from Input Mark
def predict_grade(model, marks_input, scaler, label_encoder, sequence_length=5):
    marks_scaled_input = scaler.transform(np.array([[marks_input]]))
    marks_scaled_input = np.tile(marks_scaled_input, (sequence_length, 1))
    X_input = marks_scaled_input.reshape((1, sequence_length, 1))
    predicted_grade_encoded = model.predict(X_input)
    predicted_grade = label_encoder.inverse_transform([np.argmax(predicted_grade_encoded)])
    return predicted_grade[0]

# Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for uploading the file and predicting the grade
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Convert Excel to CSV
    csv_file = file_path.replace('.xlsx', '.csv')
    convert_excel_to_csv(file_path, csv_file)

    # Read Marks and Grades
    marks, grades = read_marks_and_grades(csv_file)

    # Preprocess Data
    X, y, scaler, label_encoder = preprocess_data(marks, grades)

    # Train LSTM Model
    model = train_lstm_model(X, y)

    # Render the grade prediction page
    return render_template('predict.html', scaler=scaler, label_encoder=label_encoder, model=model)

# Flask route for predicting the grade based on input mark
@app.route('/predict', methods=['POST'])
def predict():
    marks_input = float(request.form['marks_input'])
    scaler = request.form['scaler']
    label_encoder = request.form['label_encoder']
    model = request.form['model']

    predicted_grade = predict_grade(model, marks_input, scaler, label_encoder)
    return render_template('result.html', predicted_grade=predicted_grade)

if __name__ == '__main__':
    app.run(debug=True)


