Live AI Voice Assistant for PC
!
This repository contains the code for a hands-free, conversational AI assistant that runs on your PC. It uses your webcam to "see" and your microphone to "hear," allowing you to ask questions about your environment in real-time. The assistant is powered by Google's Gemini API and responds to you with a spoken voice.
This project turns your computer into an "Alexa with eyes," capable of understanding both verbal commands and visual context.

Features

Real-time Video: Displays a live feed from your webcam.

Hands-Free Activation: Continuously listens for a "wake word" ("Okay Gemini").

Voice Commands: Uses SpeechRecognition to understand your spoken questions.

Spoken Responses: Uses pyttsx3 for text-to-speech, enabling a full conversational loop.

Visual Context: Captures the current video frame when you ask a question and sends it to the AI along with your query.

Powered by Gemini: Leverages the Google Gemini API for powerful multimodal (text + image) understanding.

Status Display: The video window shows the assistant's current state: LISTENING, ASK YOUR QUESTION, or PROCESSING.

How It Works

The application runs multiple processes in parallel using Python's multithreading:

Video Thread: Continuously captures and displays the webcam feed.

Audio Thread: Continuously listens to the microphone.

Main Loop:

When the wake word ("Okay Gemini") is detected, the status changes.

It then records your question and grabs the latest video frame.

Both the audio (as text) and the image are sent to the Gemini API.

The AI's text response is received.

The response is converted to speech and spoken out loud.

The assistant returns to the LISTENING state.

Setup and Installation

Prerequisites

Python 3.7+

A connected webcam and microphone

A Google Gemini API Key

Installation Steps

Clone the Repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name


Get Your Gemini API Key:

Visit the Google AI Studio website.

Create an API key and copy it.

Create a Virtual Environment:

Windows:

python -m venv venv
.\venv\Scripts\activate


macOS/Linux:

python3 -m venv venv
source venv/bin/activate


Install Dependencies:
This project has dependencies that require careful installation, especially PyAudio.

Step 4a: Install PyAudio

Windows:

Go to the Unofficial Windows Binaries page.

Download the .whl file that matches your Python version (e.g., PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl for Python 3.11 64-bit).

Install it directly: pip install PyAudio-0.2.14-cp311-cp311-win_amd64.whl

macOS:

brew install portaudio
pip install pyaudio


Linux:

sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio


Step 4b: Install Other Libraries
Install the rest of the required packages from the requirements.txt file:

pip install -r requirements.txt


Create .env File:
Create a file named .env in the root of the project folder and add your API key:

GEMINI_API_KEY=YOUR_API_KEY_HERE


How to Run

With your virtual environment active, simply run the main script:

python live_ai_assistant.py


A window will pop up showing your webcam feed and the "LISTENING" status.

How to Use

Activate: Say the wake word, "Okay Gemini". The status text will change to "ASK YOUR QUESTION".

Ask: You have a few seconds to ask your question (e.g., "What is this object I'm holding?").

Process: The status will change to "PROCESSING" as it sends the data to the API.

Listen: The assistant will speak its response. The answer will also be printed in your terminal.

Quit: To stop the application, click on the video window and press the 'q' key.

Troubleshooting

No Voice Output: If the assistant doesn't speak, your pyttsx3 engine may have issues. Run the test_tts.py script to debug:

python test_tts.py


This will help diagnose any problems with your system's text-to-speech drivers.

PyAudio Errors: Most installation failures are related to PyAudio. Ensure you have installed portaudio (on macOS/Linux) or used the correct .whl file (on Windows).

Microphone Not Working: Ensure your microphone is not muted and that the correct one is set as your system's default. The SpeechRecognition library will print an error if it can't find a microphone.

License

This project is licensed under the MIT License. See the LICENSE file for details.
