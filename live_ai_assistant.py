import cv2
import base64
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import io
import pyttsx3
import threading
import queue
import time
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr

# Configuration 
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file.")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
WAKE_WORD = "hey ted" 

# State Management
class AppState:
    def __init__(self):
        self.status = "LISTENING" # LISTENING, WAITING_FOR_PROMPT, PROCESSING, SPEAKING
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.status

    def set(self, new_status):
        with self.lock:
            self.status = new_status

app_state = AppState()
task_queue = queue.Queue()

# Text-to-Speech Engine
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: Could not initialize TTS engine. Responses will be text-only. Error: {e}")
    tts_engine = None

def speak(text):
    """Speaks the given text using the TTS engine."""
    if tts_engine:
        try:
            app_state.set("SPEAKING")
            print(f"AI: {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error during speech: {e}")
    else:
        print(f"AI (TTS disabled): {text}")
    app_state.set("LISTENING")

# Gemini API Call
def call_gemini_api(image_bytes, prompt):
    """Sends the image and prompt to the Gemini API."""
    print("\nSending request to Gemini...")
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "contents": [{
            "parts": [
                {"text": f"You are a helpful, observant, and friendly AI assistant. Concisely answer the user's question based on the image. Question: '{prompt}'"},
                {"inlineData": {"mimeType": "image/png", "data": encoded_image}}
            ]
        }]
    }
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
        response.raise_for_status()
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text or "I'm sorry, I couldn't find an answer in the image."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Speech Recognition
def recognize_speech_from_mic(recognizer, microphone):
    """Captures speech from the microphone using Google SR."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Google SR request failed: {e}")
        return None

# Wake Word Detection using Porcupine 
def listen_for_commands():
    """Main loop using Porcupine for wake word detection and speech recognition."""
    porcupine = pvporcupine.create(
    access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
    keywords=[WAKE_WORD], 
    keyword_paths=["hey-ted_en_windows_v3_0_0.ppn"]

    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        while True:
            current_status = app_state.get()
            if current_status not in ["LISTENING", "WAITING_FOR_PROMPT"]:
                time.sleep(0.1)
                continue

            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if current_status == "LISTENING" and keyword_index >= 0:
                print(f"Wake word '{WAKE_WORD}' detected.")
                app_state.set("WAITING_FOR_PROMPT")

            elif current_status == "WAITING_FOR_PROMPT":
                print("Listening for your question...")
                text = recognize_speech_from_mic(recognizer, microphone)
                if text:
                    print(f"You asked: '{text}'")
                    app_state.set("PROCESSING")
                    task_queue.put(text)
                else:
                    speak("I didn't catch that. Please try again.")
                    app_state.set("LISTENING")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()

# Main Application
def main():
    """Main function to run webcam feed and orchestrate threads."""
    listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listener_thread.start()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_colors = {
        "LISTENING": (0, 255, 0),
        "WAITING_FOR_PROMPT": (0, 255, 255),
        "PROCESSING": (255, 0, 0),
        "SPEAKING": (0, 0, 255)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        status = app_state.get()
        color = status_colors.get(status, (255, 255, 255))

        cv2.putText(display_frame, f"STATUS: {status}", (20, 40), font, 1, (0, 0, 0), 3)
        cv2.putText(display_frame, f"STATUS: {status}", (20, 40), font, 1, color, 2)
        cv2.putText(display_frame, "Press 'q' to quit", (20, 80), font, 0.7, (255, 255, 255), 2)

        cv2.imshow('ted camera', display_frame)

        try:
            prompt = task_queue.get_nowait()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            with io.BytesIO() as output:
                pil_image.save(output, format="PNG")
                image_bytes = output.getvalue()

            response_text = call_gemini_api(image_bytes, prompt)
            speak(response_text)
            task_queue.task_done()
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()