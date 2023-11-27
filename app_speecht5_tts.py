import gradio as gr
from transformers import pipeline
from openai import OpenAI
import scipy.io.wavfile as wavfile
import numpy as np
import torch
import os
import time
    
#ASR - HuggingFace pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
#For text-to-text response
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#start TTS
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()
#end TTS
    
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
    audio_array = synthesise(text)
    print("ending synthesise...", time.ctime())
    wavfile.write('speech.wav', 16000, np.array(audio_array))
    return "speech.wav"

def process(filepath):
    prompt = transcribe_speech(filepath)
    text = chat_response(prompt)
    return tts(text)
    
demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=process,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Audio(label="Speech Output")
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe],
        ["Process speech"],
    )

demo.launch(debug=True, share=True, auth=("admin", "godisgreat123!"))

