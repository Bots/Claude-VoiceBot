import os
import time
import torch
import struct
import pyaudio
import webrtcvad
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
CHUNK_SIZE = 480
SILENCE_THRESHOLD = 10
GRACE_PERIOD = 0.5

# Initiate webrtcvad for silence detection
vad = webrtcvad.Vad(3)

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

def transcribe():
    buffer = []
    silent_chunks = 0
    is_speech = False
    print("Listening and buffering...")
    try:
        # Add grace period
        time.sleep(GRACE_PERIOD)
        
        while True:
            chunk = audio_stream.read(CHUNK_SIZE)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            
            if is_speech:
                buffer.append(chunk)
                silent_chunks = 0
            else:
                silent_chunks += 1
            
            if silent_chunks > SILENCE_THRESHOLD:
                break
            
            if len(buffer) * CHUNK_SIZE >= SAMPLE_RATE * 5:  # 5 seconds of audio
                break

        if buffer:
            audio_data = b''.join(buffer)
            numpy_data = np.frombuffer(audio_data, dtype=np.int16)
            result = pipe(numpy_data, return_timestamps=True)
            print(result["text"])
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in buffering and streaming: {e}")
        
detect_hotword()
