import os
import pyaudio
import wave
import numpy as np

from io import BytesIO
import openai

from openai import OpenAI

import requests
import json

from colorama import Fore, Back, Style, init
from datetime import datetime

import pyttsx3
import speech_recognition as sr

def tt():
    return datetime.now().strftime("[%H:%M:%S]")

# messages:list = [
#    {"role":"user/assistant", "content":"some text"}
# ]

init(autoreset=True)
engine = pyttsx3.init()
messages:list = [{"role":"user", "content":"My name is Ivan and I've created you. Your name is Mario and you are helpful assistant. You give me info and solve problems I tell you. Also you are my friend and you can be rude sometimes. Ivan is inteligent human being, since he created you LOL. You can be styled as 4chan and anything someone asked you "}]
openai.api_key = os.environ["OPENAI_API_KEY"]

def ai_res(messages:list):
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=1000,
    )
    text = response.choices[0].message.content
    messages.append({"role":"assistant", "content":text})
    print(Fore.CYAN + f"{tt()} <@assistant> {text}")
    engine.say(text)
    engine.runAndWait()

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
    transcription = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )
    return transcription.text

def record_on_voice_stop(silence_threshold=1500, silence_duration=2, chunk=1024, rate=44100, min_voice_duration=0.25):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    print("Started...")

    try:
        while True:
            frames = []
            silent_chunks = 0
            voiced_chunks = 0
            max_silent_chunks = int(rate / chunk * silence_duration)
            min_voiced_chunks = int(rate / chunk * min_voice_duration)
            is_recording = False

            while True:
                data = stream.read(chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if volume > silence_threshold:
                    frames.append(data)
                    voiced_chunks += 1
                    silent_chunks = 0
                    is_recording = True
                elif is_recording:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        break

                elif not is_recording:
                    continue
            
            if voiced_chunks >= min_voiced_chunks:
                buffer = BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(rate)
                    wf.writeframes(b''.join(frames))
                buffer.seek(0)
                buffer.name = "audio_file.wav"
                text = speech_to_text(buffer)
                print(Fore.MAGENTA + f"{tt()} <@me> {text}")
                messages.append({"role":"user", "content":text})
                ai_res(messages)
            else:
                print(Style.DIM + "audio short or empty, skipping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

record_on_voice_stop()