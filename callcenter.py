import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import pygame
import os
import whisper
import edge_tts
import playsound
import speech_recognition as sr
import pyttsx3
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# API Key management
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Load hotel information from file
def load_hotel_info(filename="hotel_infos.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        hotel_data = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                hotel_data[key.strip()] = value.strip()
        
        return hotel_data
    except FileNotFoundError:
        print("Error: hotel_infos.txt not found.")
        return {}

# Save updated hotel info back to file
def save_hotel_info(hotel_data, filename="hotel_infos.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for key, value in hotel_data.items():
            f.write(f"{key}: {value}\n")

# Initialize hotel info
data = load_hotel_info()

# Prompt template
template = """
You are Sam, a kind and professional AI hotel receptionist for Ramada.
You only answer questions related to the hotel and its services.
If the user asks about room reservations, check availability and confirm or deny the request.
If the question is not about the hotel, politely refuse to answer.
If the user asks about hotel facilities, provide relevant information.

Conversation History:
{context}

Retrieved Information:
{retrieved_info}

Question: {question}

Answer:
"""

# Initialize model
model = Ollama(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load and vectorize documents
try:
    loader = TextLoader("hotel_infos.txt", encoding="utf-8")
    documents = loader.load()
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(documents, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )
except FileNotFoundError:
    print("Error: hotel_infos.txt not found.")
    qa = None

def check_and_book_room(room_type):
    key = f"{room_type}"
    if key in data and int(data[key].split()[0]) > 0:
        data[key] = f"{int(data[key].split()[0]) - 1} rooms available"
        save_hotel_info(data)
        return f"Reservation confirmed for a {room_type} room."
    return f"Sorry, no {room_type} rooms are available."

whisper_model = whisper.load_model("medium") 

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  
            
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())

            result = whisper_model.transcribe("temp.wav", language="en")
            text = result["text"]
            print(f"You said: {text}")

            os.remove("temp.wav")

            return text
        except Exception as e:
            print(f"Error: {e}")
            return None

async def speak_edge(text):
    """Text-to-speech using Edge TTS"""
    try:
        communication = edge_tts.Communicate(text, "en-US-GuyNeural")
        
        output_file = "output1.mp3"
        await communication.save(output_file)
        
        # Ensure that the file is properly closed before playing it
        if os.path.exists(output_file):
            play_audio(output_file)
        else:
            print("Error: Audio file not found.")
        
        os.remove(output_file)  # Remove after playing
    except Exception as e:
        print(f"TTS error: {e}")

def play_audio(file_path):
    """Play audio using pygame"""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Stop audio playback and release the file
    pygame.mixer.music.stop()
    pygame.mixer.quit()  # Ensure mixer is properly quit after playing
    
    # Delete the file after playing
    try:
        os.remove(file_path)
    except PermissionError as e:
        print(f"Error: Could not delete {file_path}. Reason: {e}")

async def handle_conversation():
    context = ""
    print("Welcome to the AI Call Center! Speak or type 'exit' to quit.")
    
    while True:
        user_input = recognize_speech()
        if not user_input:
            continue
        
        if user_input.lower() == "exit." or user_input == "Exit.":
            print("Conversation ended.")
            break
        
        if "book" in user_input.lower() or "reserve" in user_input.lower():
            for room_type in ["Eco", "Lux", "Super Lux"]:
                if room_type.lower() in user_input.lower():
                    response = check_and_book_room(room_type)
                    print("Bot:", response)
                    await speak_edge(response)
                    continue
        
        retrieved_info = qa.invoke(user_input)["result"] if qa else "Information unavailable."
        result = chain.invoke({
            "context": context,
            "retrieved_info": retrieved_info,
            "question": user_input
        })
        
        bot_response = result
        print("Bot:", bot_response)
        await speak_edge(bot_response)
        context += f"\nUser: {user_input}\nAI: {bot_response}"
        
async def main():
    await handle_conversation()

if __name__ == "__main__":
    asyncio.run(main())
