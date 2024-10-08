import os
import wave
import time
import torch
import struct
import logging
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

# Other settings
global tts_method

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Ask user which TTS method they want to use
def get_tts_preference():
    while True:
        choice = input("Which TTS method do you want to use? (1) ElevenLabs(cloud) (2) pyttsx3(on-device) default is 1: ")
        if choice == "" or choice == "1":
            return 'elevenlabs' 
        elif choice == '2':
            return 'pyttsx3'
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return 'elevenlabs'

# Start hotword detection
def detect_hotword():
    logger.info("Starting hotword detection...")
    start_time = time.time()
    try:
        while True:
            audio_frame = audio_stream.read(porcupine.frame_length)
            audio_frame = struct.unpack_from("h" * porcupine.frame_length, audio_frame)
            keyword_index = porcupine.process(audio_frame)
            if keyword_index >= 0:
                logger.info(f"Hotword detected in {time.time() - start_time:.2f} seconds")
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
    logger.info("Starting transcription...")
    play_audio("noti.wav")
    buffer = []
    silent_chunks = 0
    start_time = time.time()
    
    try:
        for chunk in iter(lambda: audio_stream.read(CHUNK_SIZE), b''):
            buffer.append(chunk)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            
            if is_speech:
                silent_chunks = 0
            else:
                silent_chunks += 1
                
            if silent_chunks > SILENCE_THRESHOLD:
                break
            
            if len(buffer) * CHUNK_SIZE >= SAMPLE_RATE * 20:  # Max 20 seconds of audio
                break
        
        if buffer:
            audio_data = b''.join(buffer)
            numpy_data = np.frombuffer(audio_data, dtype=np.int16)
            transcription_start = time.time()
            result = pipe(numpy_data, return_timestamps=True)
            print("Query understood: ", result["text"])
            logger.info(f"Transcription completed in {time.time() - transcription_start:.2f} seconds")
            logger.info(f"Total transcription process took {time.time() - start_time:.2f} seconds")
            query_llm(result["text"])
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in transcription: {e}")

# Query Claude
def query_llm(prompt):
    logger.info("Starting LLM query...")
    start_time = time.time()
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
        text = message.content[0].text
        logger.info(f"LLM query completed in {time.time() - start_time:.2f} seconds")
        tts(text)
    except Exception as e:
        print(f"Error querying Claude: {e}")
   
# Text to speech
def tts(response):
    logger.info("Starting text-to-speech...")
    start_time = time.time()
    try:
        print("Speaking response...")
        if tts_method == 'elevenlabs':
            audio = elevenlabs.generate(
                text=response,
                voice="Rachel",
                model="eleven_multilingual_v2"
            )
            logger.info(f"Text-to-speech completed in {time.time() - start_time:.2f} seconds")
            play(audio)
        elif tts_method == 'pyttsx3':
            engine.say(response)
            engine.runAndWait()
            logger.info(f"Text-to-speech completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Start the program      
if __name__ == "__main__":
    tts_method = get_tts_preference()
    print(f"Using {tts_method} for text-to-speech")
    detect_hotword()
