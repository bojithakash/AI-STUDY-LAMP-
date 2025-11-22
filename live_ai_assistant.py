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

load_dotenv(override=True) 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
if not PICOVOICE_ACCESS_KEY:
    raise ValueError("PICOVOICE_ACCESS_KEY not found in .env file.")

# Print masked key to verify it loaded the NEW one
print(f"Loaded Picovoice Key: {PICOVOICE_ACCESS_KEY[:10]}... (check if this matches your new key)")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
WAKE_WORD = "hey ted" 

class AppState:
    def __init__(self):
        self.status = "LISTENING" 
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.status

    def set(self, new_status):
        with self.lock:
            self.status = new_status

app_state = AppState()
task_queue = queue.Queue()

# --- Threaded TTS Engine ---
class TextToSpeech(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.queue = queue.Queue()
        self.started = False

    def run(self):
        self.started = True
        while True:
            try:
                text = self.queue.get()
                if text is None: break 
                
                app_state.set("SPEAKING")
                print(f"AI: {text}")
                
                engine = pyttsx3.init()
                engine.setProperty('rate', 175) 
                voices = engine.getProperty('voices')
                if len(voices) > 1:
                    engine.setProperty('voice', voices[1].id)

                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine

                app_state.set("LISTENING")
                self.queue.task_done()
                
            except Exception as e:
                print(f"TTS Error: {e}")
                app_state.set("LISTENING")
                try:
                    self.queue.task_done()
                except:
                    pass

    def speak(self, text):
        if not self.started:
            self.start()
            time.sleep(0.5)
        self.queue.put(text)

tts_handler = TextToSpeech()
tts_handler.start()

def speak(text):
    tts_handler.speak(text)

# --- Gemini API ---
def call_gemini_api(image_bytes, prompt):
    print("\nSending request to Gemini...")
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "contents": [{
            "parts": [
                {"text": f"You are a helpful AI. Concisely answer based on the image. Question: '{prompt}'"},
                {"inlineData": {"mimeType": "image/png", "data": encoded_image}}
            ]
        }]
    }
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
        response.raise_for_status()
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text or "I didn't catch that."
    except Exception as e:
        return f"Error: {e}"

# --- Speech Recognition ---
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # Short timeout to keep UI responsive
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio).lower()
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
            return None

# --- Wake Word Listener ---
def listen_for_commands():
    try:
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY, # Uses the variable loaded at the top
            keywords=[WAKE_WORD], 
            keyword_paths=["hey-ted_en_windows_v3_0_0.ppn"] 
        )
    except Exception as e:
        print(f"Porcupine Error: {e}")
        return

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

    print("Audio Listener Started...")

    try:
        while True:
            current_status = app_state.get()
            
            if current_status in ["PROCESSING", "SPEAKING"]:
                time.sleep(0.1)
                continue

            if current_status == "LISTENING":
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print(f"Wake word '{WAKE_WORD}' detected.")
                    app_state.set("WAITING_FOR_PROMPT")

            elif current_status == "WAITING_FOR_PROMPT":
                print("Listening for prompt...")
                text = recognize_speech_from_mic(recognizer, microphone)
                
                if text:
                    print(f"User: '{text}'")
                    app_state.set("PROCESSING")
                    task_queue.put(text)
                else:
                    print("No speech detected.")
                    app_state.set("LISTENING")
    finally:
        if stream: stream.stop_stream(); stream.close()
        if pa: pa.terminate()
        porcupine.delete()

# --- Main Loop ---
def main():
    listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listener_thread.start()

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

    print("System Ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        status = app_state.get()
        color = status_colors.get(status, (255, 255, 255))

        cv2.putText(display_frame, f"STATUS: {status}", (20, 40), font, 1, (0, 0, 0), 6)
        cv2.putText(display_frame, f"STATUS: {status}", (20, 40), font, 1, color, 2)

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