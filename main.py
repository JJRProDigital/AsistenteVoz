from dotenv import load_dotenv
import os
import streamlit as st
import openai
from openai import OpenAI
from config import *
from audio_recorder_streamlit import audio_recorder


load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai.api_key)

assistant_id = os.getenv("ASSISTANT_KEY")



def ensure_single_thread_id():
    if "thread_id" not in st.session_state:
        thread= client.beta.threads.create()
        st.session_state.thread_id=thread.id
    return st.session_state.thread_id

def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text  # Accedemos al texto directamente

def stream_generator(prompt,thread_id):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    with st.spinner("Wait... Generating response..."):
        stream= client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True
        )
        for event in stream:
            if event.data.object == "thread.message.delta":
                for content in event.data.delta.content:
                    if content.type == 'text':
                        yield content.text.value
            else:
                pass

def text_to_audio(client, text, audio_path):
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    response.stream_to_file(audio_path)

def main():
    st.title("Asistente de voz")
    st.write("Asistente de voz con OpenAI")

    # Grabamos el audio
    recorded_audio = audio_recorder(key="record")

    if recorded_audio:
        thread_id = ensure_single_thread_id()
        audio_file = "user_question.mp3"
        with open(audio_file, "wb") as f:
            f.write(recorded_audio)

        # Transcribimos el audio a texto
        transcribed_text = transcribe_audio(client, audio_file)
        st.write("You said:", transcribed_text)

        # Generamos la respuesta
        response_text=st.write_stream(stream_generator(transcribed_text,thread_id))

        # Convertimos la respuesta en audio
        response_audio_file = "response.mp3"
        text_to_audio(client, response_text, response_audio_file)

        # Reproducimos la respuesta
        st.audio(response_audio_file)

if __name__ == "__main__":
    main()