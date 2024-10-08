import os
import wave
import time
import torch
import struct
import pyaudio
import pyttsx3
import webrtcvad
import pvporcupine
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from transformers import pipeline
from elevenlabs import ElevenLabs, play

# Load environment vars
load_dotenv()
print("loading env vars...")
PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")
KEYWORD_FILE_PATH = os.getenv("KEYWORD_FILE_PATH")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Configure audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 480
SILENCE_THRESHOLD = 10
GRACE_PERIOD = 2.0
GRACE_CHUNKS = int(GRACE_PERIOD * SAMPLE_RATE / CHUNK_SIZE)

# Initiate pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Initiate webrtcvad for silence detection
vad = webrtcvad.Vad(3)

# Initiate pvporcupine for hotword detection
porcupine = pvporcupine.create(
    access_key=PV_ACCESS_KEY, keyword_paths=[KEYWORD_FILE_PATH])

# Initiate anthropic
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

# Initiate elevenlabs
elevenlabs = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

# Configure pyaudio for audio stream
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=SAMPLE_RATE,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length,
)

# Initiate transformers pipeline for speech recognition
print("loading pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3-turbo",
    torch_dtype = torch.float16,
    device = "cuda:0"
)

# Start hotword detection
def detect_hotword():
    print("listening for hotword...")
    try:
        while True:
            audio_frame = audio_stream.read(porcupine.frame_length)
            audio_frame = struct.unpack_from("h" * porcupine.frame_length, audio_frame)
            keyword_index = porcupine.process(audio_frame)
            if keyword_index >= 0:
                print("hotword detected")
                transcribe()
                print("Resuming hotword detection...")
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in hotword detection: {e}")

# Helper function to play audio notification sound
def play_audio(file_path):
    try:
        wf = wave.open(file_path, 'rb')
        stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        
        stream.stop_stream()
        stream.close()
        wf.close()
    except Exception as e:
        print(f"Error playing notification audio: {e}")
    
# Helper function to transcribe audio
def transcribe():
    buffer = []
    silent_chunks = 0
    grace_chunks_remaining = GRACE_CHUNKS
    print("Transcribing...")
    try:
        # Start buffering immediately
        for _ in range(int(0.5 * SAMPLE_RATE / CHUNK_SIZE)):  # Buffer 0.5 seconds initially
            chunk = audio_stream.read(CHUNK_SIZE)
        play_audio("noti.wav")
        while True:
            chunk = audio_stream.read(CHUNK_SIZE)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            
            buffer.append(chunk)
            
            if is_speech:
                silent_chunks = 0
                grace_chunks_remaining = GRACE_CHUNKS
            else:
                silent_chunks += 1
                if grace_chunks_remaining > 0:
                    grace_chunks_remaining -= 1
                elif silent_chunks > SILENCE_THRESHOLD:
                    break
            
            if len(buffer) * CHUNK_SIZE >= SAMPLE_RATE * 10:  # Max 10 seconds of audio
                break

        if buffer:
            audio_data = b''.join(buffer)
            numpy_data = np.frombuffer(audio_data, dtype=np.int16)
            result = pipe(numpy_data, return_timestamps=True)
            print(result["text"])
            query_llm(result["text"])
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in transcription: {e}")

# Query Claude
def query_llm(prompt):
    print("Querying Claude...")
    try:
        message = client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"You are a helpful assistant. Answer the following question with a consice answer (preferably 1 sentence): {prompt}"
                }
            ],
            model="claude-3-opus-20240229",
        )
        print(message.content)
        tts(message.content[0].text)
    except Exception as e:
        print(f"Error querying Claude: {e}")
   
# Text to speech
def tts(response):
    try:
        print("Speaking response...")
        engine.say(response)
        engine.runAndWait()
        # audio = elevenlabs.generate(
        #     text=response,
        #     voice="Rachel",
        #     model="eleven_multilingual_v2"
        # )
        # play(audio)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Start the program      
detect_hotword()
