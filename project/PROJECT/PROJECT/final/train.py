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
def extract_sheet_and_save_as_csv(excel_file, output_csv):
    # Read the specified sheet from the Excel file
    data = pd.read_excel(excel_file)
    # Save the sheet data to a CSV file
    data.to_csv(output_csv, index=False)
    return output_csv

# Input Excel file and sheet name
excel_file = 'SQL.xlsx'  # Example Excel file name
csv_file = extract_sheet_and_save_as_csv(excel_file, 'student_marks.csv')



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

model.save('SQL.h5')

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
