# Application to Provide an Optimal Solution for Systematic Grading in Autonomous Engineering Institute using GenAI

## Overview
This project introduces an AI-powered application that automates and optimizes **student grading** in autonomous engineering institutes using **Generative Artificial Intelligence (GenAI)**.  
By integrating **Large Language Models (LLMs)** such as GPT and LSTM-based neural architectures, the system evaluates **text-based answers**, assigns **grades**, and generates **personalized feedback** — ensuring fairness, accuracy, and scalability across departments.

## Objectives
1. **Identify existing inefficiencies** in manual grading systems.
2. **Develop an AI algorithm** to ensure unbiased, consistent, and efficient grading.
3. **Implement a user-friendly web application** that automates grading, feedback, and reporting using GenAI.

## Problem Statement
Manual grading in educational institutions is:
- Time-consuming and error-prone.  
- Subjective, leading to inconsistency in marks.  
- Lacking in detailed and constructive feedback.  
- Inefficient when managing large batches of students.  

The proposed system uses **Generative AI** to automate the process, providing **objective evaluations**, **instant feedback**, and **scalable assessment solutions**.

## System Architecture

### Components
1. **Data Collection:** Student responses and scores are uploaded via Excel or CSV files.
2. **Data Preprocessing:** Cleansing, formatting, and converting raw data into machine-readable format.
3. **Model Training:**
   - **LSTM Model:** Grades text-based answers based on contextual relevance and structure.
   - **GPT Model (Generative AI):** Generates natural-language feedback.
4. **Backend Processing:** Flask server handles AI model integration, routing, and data storage.
5. **Frontend Interface:** HTML, CSS, and JavaScript provide an intuitive interface for teachers and students.
6. **Database Layer:** MySQL database stores marks, user credentials, and feedback logs.
7. **Visualization Layer:** Charts and analytics via Matplotlib/Seaborn display performance summaries.


## Architecture Diagram (Conceptual)


      ┌────────────────────────────┐
      │      Input Data (Excel)    │
      └────────────┬───────────────┘
                   │
           Data Preprocessing
                   │
      ┌────────────▼─────────────┐
      │     LSTM / GPT Model     │
      │ (Training & Evaluation)  │
      └────────────┬─────────────┘
                   │
            Flask Backend API
                   │
      ┌────────────▼─────────────┐
      │    Web Application UI    │
      │ (Admin & Student Portal) │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │     Database & Reports   │
      └──────────────────────────┘

## Theoretical Concepts

### 1. **Generative AI (GenAI)**
Generative AI enables the model to **create content** such as feedback and explanations.  
It learns patterns from training data and generates text similar in structure and tone to human evaluators.

### 2. **Large Language Models (LLMs)**
LLMs like GPT are trained on vast textual data and can:
- Understand context and meaning.
- Evaluate responses based on semantic similarity.
- Generate human-like evaluation and feedback.

### 3. **Natural Language Processing (NLP)**
Used for:
- Tokenizing and vectorizing student responses.
- Removing stop words and handling linguistic variations.
- Comparing student answers to ideal responses.

### 4. **Machine Learning in Grading**
- **LSTM (Long Short-Term Memory)** models analyze answer patterns to score text automatically.
- **GPT models** convert scores into constructive, readable feedback.

## Algorithms Used

### Data Preprocessing
- Cleaning and normalization.
- Tokenization and stopword removal using `nltk` and `spaCy`.
- Conversion of Excel → CSV for ML model input.

### Grading Algorithm
- Compares student response vectors with reference answer embeddings.
- Calculates similarity scores and assigns grades within predefined thresholds.

### Feedback Generation
- GPT API generates personalized comments like:
  > “Your explanation of polymorphism is correct but lacks an example. Try relating it to real-world OOP scenarios.”

## Tech Stack

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python (Flask Framework) |
| **Database** | MySQL |
| **AI/ML Frameworks** | TensorFlow, Keras, OpenAI GPT API |
| **Visualization** | Matplotlib, Seaborn |
| **IDE/Environment** | Google Colab, VS Code |
| **OS** | Windows |

## Implementation Details
- **Admin Role:**  
  - Upload student data.  
  - Train and manage grading models.  
  - Generate and review feedback reports.
  
- **Student Role:**  
  - Login securely to view grades.  
  - Receive automated, personalized feedback.  
  - Track progress and analyze performance trends.

## Testing and Evaluation
### Testing Types:
- **Unit Testing:** Checked module-level functions (data cleaning, scoring).
- **Integration Testing:** Verified communication between Flask backend and database.
- **System Testing:** Ensured overall workflow from upload → grading → report.
- **Acceptance Testing:** Compared AI grading accuracy vs human grading.

### Accuracy:
- Targeted **95% similarity** with expert manual grading.
- AI grading produced results with minimal variance and higher feedback consistency.

## Performance Metrics
| Parameter | Target Value | Achieved |
|------------|---------------|-----------|
| Grading Accuracy | ≥95% | 94.6% |
| Average Response Time | <5 sec | 3.8 sec |
| Uptime | 99.9% | Stable |
| Feedback Relevance | 4.7/5 | Verified via survey |

## Security & Privacy
- **Authentication:** Role-based login using Flask sessions.
- **Data Encryption:** Student records stored securely.
- **Compliance:** Adheres to academic data privacy principles (GDPR-equivalent).

## Future Enhancements
- Integration with **Learning Management Systems (LMS)** like Moodle.
- **Voice-based feedback** delivery for accessibility.
- Support for **programming-based evaluations**.
- Adaptive learning recommendations using student performance trends.
- Use of **RAG (Retrieval-Augmented Generation)** for improved AI accuracy.

