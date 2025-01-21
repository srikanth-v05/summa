import cv2
import os
import pandas as pd
import speech_recognition as sr
import pyttsx3
import streamlit as st
import threading
from pandasai import Agent

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sample Student Data
data_a_sec = {
    "REGISTER NO": [
        "23UAI002", "23UAI003", "23UAI004", "23UAI006", "23UAI008", "23UAI010", "23UAI016", "23UAI018",
        "23UAI019", "23UAI025", "23UAI026", "23UAI029", "23UAI030", "23UAI033", "23UAI036", "23UAI038",
        "23UAI041", "23UAI044", "23UAI045", "23UAI046", "23UAI047", "23UAI049", "23UAI050", "23UAI051",
        "23UAI052", "23UAI054", "23UAI055", "23UAI056", "23UAI057", "23UAI059", "23UAI061", "23UAI062",
        "23UAI068", "23UAI069", "23UAI070", "23UAI073", "23UAI074", "23UAI076", "23UAI082", "23UAI083",
        "23UAI086", "23UAI087", "23UAI089", "23UAI091", "23UAI092", "23UAI094", "23UAI097", "23UAI098",
        "23UAI103", "23UAI108", "23UAI109", "23UAI113", "23UAI114", "23UAI116", "23UAI117", "23UAI118",
        "23UAI119", "23UAI120", "23AIL002", "23AIL003", "23AIL004"
    ],
    "ENROLL NO": [
        231332, 230712, 230634, 231321, 231562, 230864, 230829, 231217, 231366, 230685, 230584, 231555,
        230337, 231102, 230830, 230085, 230367, 231168, 231055, 230383, 230735, 230300, 230369, 231351,
        231135, 230890, 230172, 231305, 231358, 230677, 230192, 230264, 231091, 231036, 230841, 230483,
        230896, 231016, 231225, 230996, 231174, 230116, 230202, 231245, 230986, 230438, 230566, 230344,
        231413, 230978, 231606, 230217, 230206, 230409, 231004, 231224, 231239, 230592, 240019, 240009, 241015
    ],
    "NAME": [
        "ABBHISHEK BEHERA", "ABDULKALAM J", "ABIESWAR T", "AISVA MALAR A", "AKHILESH B", "AL RAAFHATH R K",
        "ARUN SRINIVAS A", "ASWITHA ALIAS SWETHA K", "BHARANIDHARAN T", "DEVISRI I", "DHAKSHA CHARAN R",
        "DIVYANAND M", "GANESH DEEPAK N", "GURALARASAN G", "HARINI A", "HARINI P (16.04.2005)",
        "HEMAJOTHI S", "JAIDHAR S", "JANANI N", "JASSEM", "KANIMOZHI S", "KARTHIK M", "KARTHIKEYAN V",
        "KAVITHA S", "KAVIYAA P S", "LOGAPRIYA B", "LOGESHWAR R", "LOKITHA K", "MAHESH R", "MANOJKUMAR P",
        "MITHUN K", "MOHAMED NAFEEZ S", "NARMADHA D", "NAVANEETHAKRISHNAN J", "NAVEEN KUMAR S", "NITISH M",
        "NITYASRI R S", "PRABHANJAN J", "PRAVEEN KUMARAN P", "PRIYA DARSHINI K", "RAGUL T", "RATHNAPRASAD D",
        "RAYYAN", "RINDHIYA A", "ROHAN KUMAR K V", "SANJIVKUMAR J", "SHIVAANI VAITHIYANATHAN GANESH",
        "SIDDARTH S", "SUBITSHA S", "THANUSH K", "THARANIYA S", "VIDHESH D", "VIGNESH VIJAIRAJ S", "VISHAL S",
        "VISHNU PRASATH P G", "VISHNULAKSHMI A", "VISHWA P", "YADU RAJ A", "HEMSAKTHYRAAM A", "PRADEEP R",
        "VAISHAL MALU K"
    ],
    'Section': ['A'] * 61
}

# B Section Student Data
data_b_sec = {  
      'sl.no': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    'enroll number': [230365, 230851, 230320, 230984, 231318, 231699, 230040, 230466, 231361, 230782, 231685, 230885, 231081, 231445, 231347, 230273, 231303, 231574, 230886, 231044, 230973, 230607, 230543, 230262, 231526, 230825, 230853, 230961, 231762, 230880, 230889, 230970, 230154, 230362, 231406, 231147, 231027, 230696, 231340, 230470, 231108, 231561, 230504, 230291, 231603, 231195, 231392, 231278, 230989, 230377, 231596, 231052, 230760, 231315, 231177, 230447, 230903, 230877, 230424, 231593],
    'register number': ['23uai001', '23uai005', '23uai007', '23uai009', '23uai011', '23uai012', '23uai013', '23uai014', '23uai017', '23uai015', '23uai020', '23uai021', '23uai022', '23uai023', '23uai024', '23uai027', '23uai028', '23uai031', '23uai034', '23uai035', '23uai037', '23uai039', '23uai040', '23uai042', '23uai043', '23uai048', '23uai053', '23uai058', '23uai060', '23uai063', '23uai064', '23uai065', '23uai066', '23uai067', '23uai071', '23uai072', '23uai075', '23uai077', '23uai079', '23uai080', '23uai081', '23uai084', '23uai085', '23uai088', '23uai090', '23uai093', '23uai095', '23uai096', '23uai099', '23uai100', '23uai101', '23uai102', '23uai104', '23uai105', '23uai106', '23uai107', '23uai110', '23uai111', '23uai112', '23uai115'],
    'name': ['aadhithya', 'agalya', 'ajay govind', 'akshya', 'amirthabhanu', 'andrew surjit ronald f', 'annie maerlin', 'anushree', 'aravind raj', 'arjun', 'bhavesh', 'bhuvanesh', 'chandra arul nishanthini', 'daamini', 'dakshnamorthy', 'dhavanesh', 'dhivya', 'ganesh', 'hafzafarzana', 'harija', 'harini', 'harish', 'hemachandran', 'hemeshwaran', 'indhuja', 'karmugilan', 'lalithambigha', 'manikandan', 'meenaloshani', 'mohammed aashiq', 'mohana priya', 'mordheeshvara', 'mugaesh', 'muthu krishna', 'nithya shri', 'nithya sri', 'prabhakaran', 'pradeepraj', 'pragadeeswaran', 'pramila', 'pratheeb', 'priyamadhan', 'priyanka', 'ravikanth', 'reyash', 'sangaradas', 'sathiyam', 'sarathy', 'sivaranjani', 'sreevardhini', 'sri aishwarya', 'srivathsan', 'sudharsan', 'sushma saraswathi', 'sushmidha', 'tarunraj', 'theenash', 'uthradevi', 'venkatesh', 'vishaal'],
    'Section': ['B'] * 60

}

# Combine Sections
df_a_sec = pd.DataFrame(data_a_sec)
df_b_sec = pd.DataFrame(data_b_sec)
df_combined = pd.concat([df_a_sec, df_b_sec], ignore_index=True)

# Load API Key from Streamlit Secrets
st.secrets.load_if_needed()
os.environ["PANDASAI_API_KEY"] = st.secrets["PANDASAI_API_KEY"]
agent = Agent(df_combined)

def speak(text):
    def tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts).start()

def listen_for_command():
    try:
        with sr.Microphone() as source:
            st.write("Listening for voice command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            st.write(f"Command received: {command}")
            return command
    except sr.UnknownValueError:
        st.write("Sorry, I did not understand the command.")
        return None
    except sr.RequestError:
        st.write("Could not request results from Google Speech Recognition service.")
        return None

st.title("Student Data Query System")
st.header("Data Query System")
manual_command = st.text_input("Type your query here:")
if st.button('Submit Query') and manual_command:
    response = agent.chat(manual_command)
    st.write(f"AI Response: {response}")
    speak(response)