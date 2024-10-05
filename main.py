import os
import torch
import struct
import pyaudio
import pvporcupine
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline
from matplotlib import pyplot as plt

# Load environment vars
print("loading env vars...")
load_dotenv()
PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")
KEYWORD_FILE_PATH = os.getenv("KEYWORD_FILE_PATH")
SAMPLE_RATE = 16000
CHUNK_SIZE = 256

# Initiate pvporcupine for hotword detection
porcupine = pvporcupine.create(
    access_key=PV_ACCESS_KEY, keyword_paths=[KEYWORD_FILE_PATH])

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

def detect_hotword():
    print("listening for hotword...")
    try:
        while True:
            audio_frame = audio_stream.read(porcupine.frame_length)
            audio_frame = struct.unpack_from("h" * porcupine.frame_length, audio_frame)
            keyword_index = porcupine.process(audio_frame)
            if keyword_index>=0:
                print("hotword detected")
                transcribe()
                print("Resuming hotword detection...")
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in hotword detection: {e}")

def buffer_and_stream():
    buffer = []
    print("Listening and buffering...")
    try:
        while True:
            chunk = audio_stream.read(CHUNK_SIZE)
            buffer.append(chunk)

            if len(buffer) * CHUNK_SIZE >= SAMPLE_RATE:
                audio_data = b''.join(buffer)
                numpy_data = np.frombuffer(audio_data, dtype=np.int16)
                result = pipe(numpy_data, return_timestamps=True)
                print(result["text"])
                buffer = []
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in buffering and streaming: {e}")

def transcribe():
    try:
        buffer_and_stream()
    except Exception as e:
        print(f"Error in transcription: {e}")
        
detect_hotword()
