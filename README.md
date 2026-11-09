# stt

A talking assistant in the terminal: speak into the mic, it transcribes your speech, asks an OpenAI model and reads the answer back out loud (pyttsx3).

- `stt.py`: the voice assistant ("Mario")
- `stt_old.py`: earlier version using OpenAI Whisper transcription
- `example.py`: local RAG experiment with Ollama + LangChain + Chroma
- `testing.ipynb`: scratch notebook

## Run

See `requirements.md` for dependencies, then:

```sh
export OPENAI_API_KEY=...
python3 stt.py
```
