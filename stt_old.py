import os
import pyaudio
import wave
import numpy as np

from io import BytesIO
import speech_recognition as sr
from openai import OpenAI

# def speech_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         print("Could not understand audio")
#         return None
#     except sr.RequestError as e:
#         print(f"API request error: {e}")
#         return None

def speech_to_text(audio_file):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )
    return transcription.text

def record_on_voice_stop(silence_threshold=500, silence_duration=2, chunk=1024, rate=44100):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print("Listening... Start talking.")

    frames = []
    silent_chunks = 0
    max_silent_chunks = int(rate / chunk * silence_duration)

    while True:
        data = stream.read(chunk)
        frames.append(data)
        

        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > max_silent_chunks:
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    
    buffer.seek(0)
    buffer.name = "audio_file.wav"
    print(speech_to_text(buffer))

record_on_voice_stop()