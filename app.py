import gradio as gr
import torch
from transformers import pipeline
from openai import OpenAI
import os
import time
    
#ASR
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
#For text-to-text response
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def transcribe_speech(filepath):
    print("starting transcribe_speech...", time.ctime())
    output = pipe(
        filepath,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=8,
    )
    print("ending transcribe_speech...", time.ctime())
    return output["text"]

def chat_response(prompt):
    print("starting chat.completions...", time.ctime())
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=256,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print("ending chat.completions...", time.ctime())
    return response.choices[0].message.content

def tts(text):
    print("starting synthesise...", time.ctime())
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    response.stream_to_file("speech.mp3")
    print("ending synthesise...", time.ctime())
    return "speech.mp3"

def process(filepath):
    prompt = transcribe_speech(filepath)
    text = chat_response(prompt)
    return tts(text)
    
demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=process,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Audio(label="Speech Output")
    # outputs = "text"
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe],
        ["Process Speech"],
    )

demo.launch(debug=True, share=True)
