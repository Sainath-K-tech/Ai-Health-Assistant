import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pyttsx3
import speech_recognition as sr
import warnings
warnings.filterwarnings("ignore")  # Corrected method name

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to speak text and print it
def speak(text):
    print(f"Assistant: {text}")  # Print the spoken text
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to listen and recognize speech with text fallback
def listen():
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")
                return text
        except sr.UnknownValueError:
            speak(f"Attempt {attempt + 1}: Sorry, I didn't understand. Please try again.")
        except sr.RequestError as e:
            speak(f"Attempt {attempt + 1}: Speech recognition service error: {str(e)}. Please try again.")
        except Exception as e:
            speak(f"Attempt {attempt + 1}: An error occurred: {str(e)}. Please try again.")
    
    # Fallback to text input
    speak("Speech recognition failed. Please type your input.")
    text = input("Type your input: ").lower()
    print(f"You typed: {text}")
    return text

# Load datasets (simulated for demo)
def load_diabetes_dataset():
    data = {
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
        'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
        'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
        'Outcome': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)

def load_heart_disease_dataset():
    data = {
        'Age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
        'Sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        'ChestPainType': [3, 2, 1, 1, 0, 0, 1, 2, 2, 0],
        'RestingBP': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150],
        'Cholesterol': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
        'FastingBS': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'RestingECG': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        'MaxHR': [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
        'ExerciseAngina': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'Oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0.0, 0.5, 1.6],
        'ST_Slope': [0, 0, 2, 2, 2, 1, 1, 2, 2, 2],
        'Outcome': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)

def load_ckd_dataset():
    data = {
        'Age': [48, 36, 62, 45, 50, 41, 58, 60, 55, 44],
        'BloodPressure': [80, 70, 90, 70, 80, 75, 85, 90, 80, 70],
        'SpecificGravity': [1.020, 1.025, 1.010, 1.015, 1.020, 1.015, 1.010, 1.020, 1.025, 1.015],
        'Albumin': [1, 0, 3, 2, 0, 1, 4, 2, 0, 1],
        'BloodUrea': [36, 18, 53, 25, 30, 22, 70, 40, 28, 20],
        'SerumCreatinine': [1.2, 0.8, 1.8, 1.0, 1.1, 0.9, 3.2, 1.5, 1.0, 0.7],
        'Hemoglobin': [15.4, 16.0, 11.2, 13.5, 14.8, 15.0, 10.8, 12.5, 15.5, 16.2],
        'Outcome': [1, 0, 1, 0, 0, 0, 1, 1, 0, 0]  # 1: CKD, 0: No CKD
    }
    return pd.DataFrame(data)

# Train models for each disease
def train_models():
    # Diabetes model
    diabetes_df = load_diabetes_dataset()
    X_diabetes = diabetes_df.drop('Outcome', axis=1)
    y_diabetes = diabetes_df['Outcome']
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
    scaler_diabetes = StandardScaler()
    X_train_d_scaled = scaler_diabetes.fit_transform(X_train_d)
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X_train_d_scaled, y_train_d)

    # Heart disease model
    heart_df = load_heart_disease_dataset()
    X_heart = heart_df.drop('Outcome', axis=1)
    y_heart = heart_df['Outcome']
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    scaler_heart = StandardScaler()
    X_train_h_scaled = scaler_heart.fit_transform(X_train_h)
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_model.fit(X_train_h_scaled, y_train_h)

    # Chronic kidney disease model
    ckd_df = load_ckd_dataset()
    X_ckd = ckd_df.drop('Outcome', axis=1)
    y_ckd = ckd_df['Outcome']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_ckd, y_ckd, test_size=0.2, random_state=42)
    scaler_ckd = StandardScaler()
    X_train_c_scaled = scaler_ckd.fit_transform(X_train_c)
    ckd_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ckd_model.fit(X_train_c_scaled, y_train_c)

    return {
        'diabetes': (diabetes_model, scaler_diabetes, X_diabetes.columns),
        'heart_disease': (heart_model, scaler_heart, X_heart.columns),
        'chronic_kidney_disease': (ckd_model, scaler_ckd, X_ckd.columns)
    }

# Function to convert spoken input to numeric value
def get_numeric_input(feature):
    max_attempts = 3
    for _ in range(max_attempts):
        speak(f"Please say your {feature}.")
        response = listen()
        if response:
            try:
                value = float(response)
                if value < 0:
                    speak("Value cannot be negative. Please try again.")
                else:
                    return value
            except ValueError:
                speak("Invalid input. Please say a numeric value.")
    speak(f"Could not get a valid {feature}. Using default value of 0.")
    return 0

# Function to select diseases via voice
def select_diseases():
    diseases = ['diabetes', 'heart disease', 'chronic kidney disease']
    selected = []
    speak("Which diseases would you like to assess? Say the name of each disease, one at a time. Say 'finish' when finished.")
    while True:
        response = listen()
        if response == 'finish':
            break
        if response in [d.lower() for d in diseases]:
            selected.append(response)
            speak(f"Added {response} to the assessment.")
        else:
            speak("Invalid disease name. Please say 'diabetes', 'heart disease', 'chronic kidney disease', or 'finish'.")
    return selected if selected else diseases  # Default to all if none selected

# Voice-based health assistant
def voice_health_assistant():
    models = train_models()
    
    speak("Welcome to the Multi-Disease Health Assistant!")
    speak("I can assess risks for diabetes, heart disease, and chronic kidney disease.")
    
    # Select diseases
    selected_diseases = select_diseases()
    if not selected_diseases:
        speak("No diseases selected. Exiting.")
        return

    for disease in selected_diseases:
        speak(f"Now assessing {disease} risk.")
        model, scaler, feature_names = models[disease.replace(' ', '_')]
        
        # Collect user inputs
        inputs = []
        for feature in feature_names:
            value = get_numeric_input(feature)
            inputs.append(value)
        
        # Prepare input for prediction
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        
        # Provide feedback
        if disease == 'diabetes':
            if prediction == 1:
                speak(f"Your {disease} result indicates a high risk, with a probability of {probability:.2f} percent.")
                speak("I recommend consulting a healthcare provider for further tests.")
                speak("Tips: Maintain a balanced diet, exercise regularly, and monitor blood sugar levels.")
            else:
                speak(f"Your {disease} result indicates a low risk, with a probability of {probability:.2f} percent.")
                speak("Continue your healthy lifestyle habits.")
                speak("Tips: Regular check-ups and a balanced diet can help maintain low risk.")
        elif disease == 'heart disease':
            if prediction == 1:
                speak(f"Your {disease} result indicates a high risk, with a probability of {probability:.2f} percent.")
                speak("Consult a cardiologist for further evaluation.")
                speak("Tips: Avoid smoking, manage stress, and engage in regular physical activity.")
            else:
                speak(f"Your {disease} result indicates a low risk, with a probability of {probability:.2f} percent.")
                speak("Keep up your heart-healthy habits.")
                speak("Tips: Maintain a healthy weight and monitor cholesterol levels.")
        elif disease == 'chronic kidney disease':
            if prediction == 1:
                speak(f"Your {disease} result indicates a high risk, with a probability of {probability:.2f} percent.")
                speak("Consult a nephrologist for further evaluation, such as kidney function tests.")
                speak("Tips: Manage blood pressure, stay hydrated, and avoid excessive use of painkillers.")
            else:
                speak(f"Your {disease} result indicates a low risk, with a probability of {probability:.2f} percent.")
                speak("Continue maintaining kidney-healthy habits.")
                speak("Tips: Regular check-ups and a low-sodium diet can help protect your kidneys.")

# Run the voice assistant
if __name__ == "__main__":
    try:
        voice_health_assistant()
    except KeyboardInterrupt:
        speak("Assistant stopped. Goodbye!")