🎙️👁️ Live AI Voice Assistant for PC
![AI LAMP](https://github.com/user-attachments/assets/8610322b-73bc-4aec-8cec-0589c0aeb73e)
“An Alexa with Eyes” – A Multimodal Desktop AI Assistant
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Overview

Live AI Voice Assistant for PC is a hands-free, real-time multimodal AI assistant that can:

    👂 Listen to your voice

    👁️ See through your webcam

    🧠 Understand context using Google Gemini

    🗣️ Respond with natural speech

This project turns your computer into an “Alexa with Eyes”, capable of answering questions about what it sees around you.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✨ Key Features

✅ Wake Word Activation
Say “hey ted” to activate the assistant (no keyboard needed)

✅ Real-Time Vision
Live webcam feed with visual context sent to the AI

✅ Voice Interaction
Ask questions naturally using your microphone

✅ AI-Powered Responses
Uses Google Gemini (Multimodal) for text + image understanding

✅ Spoken Replies
AI responses are spoken aloud using text-to-speech

✅ Status Display
On-screen status:

LISTENING

WAITING_FOR_PROMPT

PROCESSING

SPEAKING
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 How It Works

The system runs multiple components in parallel using multithreading:

🔄 System Flow

Video Thread

Continuously captures webcam frames

Audio Thread

Always listens for the wake word: “hey ted”

After Wake Word

Records your spoken question

Captures the latest camera frame

AI Processing

Sends text + image to Gemini API

Receives AI response

Speech Output

Converts text response into voice

Speaks it out loud

Returns to Listening Mode
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🛠️ Tech Stack

Python 3.7+

OpenCV – Webcam video

SpeechRecognition – Voice input

PyAudio – Microphone streaming

Picovoice Porcupine – Wake word detection

pyttsx3 – Text-to-Speech

Google Gemini API – Multimodal AI

Multithreading – Real-time performance
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📦 Requirements
Hardware

Webcam 📷

Microphone 🎤

Internet connection 🌐

Software

Python 3.7 or higher

Google Gemini API Key

Picovoice Access Key
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔧 Installation & Setup
1️⃣ Clone the Repository

     git clone https://github.com/bojithakash/AI-STUDY-LAMP-.git
     cd AI-STUDY-LAMP-

2️⃣ Create a Virtual Environment

Windows

    python -m venv venv
    venv\Scripts\activate

macOS / Linux

     python3 -m venv venv
     source venv/bin/activate

3️⃣ Install PyAudio (Important!)
🪟 Windows

Go to Unofficial Windows Binaries

Download the correct .whl file for your Python version
Example:

     PyAudio-0.2.14-cp311-cp311-win_amd64.whl


Install:

     pip install PyAudio-0.2.14-cp311-cp311-win_amd64.whl

🍎 macOS

     brew install portaudio
     pip install pyaudio

🐧 Linux

     sudo apt-get install portaudio19-dev python3-pyaudio
     pip install pyaudio

4️⃣ Install Remaining Dependencies
 
      pip install -r requirements.txt

requirements.txt
opencv-python
requests
python-dotenv
Pillow
SpeechRecognition
pyttsx3

5️⃣ Create .env File

Create a file named .env in the project root:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
PICOVOICE_ACCESS_KEY=YOUR_PICOVOICE_ACCESS_KEY_HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
▶️ Running the Assistant

With the virtual environment activated:

     python live_ai_assistant.py


✔️ A webcam window will appear
✔️ Status will show LISTENING
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🗣️ How to Use

Activate
Say 👉 “hey ted”

Ask a Question
Example:

“What is this object?”

“What color is this?”

Processing
Status changes to PROCESSING

Listen
AI replies with a spoken response

Quit
Press q on the video window
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧪 Troubleshooting
🔊 No Voice Output

Run the TTS test:

     python test_tts.py


Ensure:

System speakers are working

TTS drivers are available

🎙️ Microphone Not Working

Check system default microphone

Ensure it is not muted

Run:

     python -m speech_recognition
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚠️ PyAudio Installation Errors

Make sure PortAudio is installed

Windows users must use the correct .whl file
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🌟 Future Improvements

🧠 Memory & conversation history

🖥️ GUI dashboard

🤖 ROS2 integration

🔌 Offline speech models

📱 Mobile / IoT companion mode

🙌 Credits

Google Gemini API

Picovoice Porcupine

OpenCV

Python Open-Source Community

📜 License

This project is open-source and intended for learning, research, and innovation.
