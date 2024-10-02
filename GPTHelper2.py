import os
import struct
import pyaudio
import pvporcupine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
access_key = os.getenv("PICOVOICE_ACCESS_KEY")
keyword_path = os.getenv("KEYWORD_FILE_PATH")

porcupine = None
paud = None
audio_stream = None


def initialize_audio():
    """Initialize PVPorcupine and audio stream."""
    porcupine = pvporcupine.create(
        access_key=access_key, keyword_paths=[keyword_path])
    paud = pyaudio.PyAudio()
    audio_stream = paud.open(rate=porcupine.sample_rate, channels=1,
                             format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
    return porcupine, paud, audio_stream


try:
    # Initialize PVPorcupine and audio stream
    porcupine, paud, audio_stream = initialize_audio()

    # Listen for hotword
    print("Listening for hotword...")
    while True:
        keyword = audio_stream.read(porcupine.frame_length)
        keyword = struct.unpack_from("h" * porcupine.frame_length, keyword)
        keyword_index = porcupine.process(keyword)
        if keyword_index >= 0:
            print("Hotword detected.")

finally:
    print("Shutting down...")
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if paud is not None:
        paud.terminate()
