# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\callcenter\Scripts\Activate

import speech_recognition as sr
import pyttsx3
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

client = ElevenLabs(api_key="api_key") 

template = """
Answer the question below based on the context.

Here is the conversation history:
{context}

Here is the retrieved information:
{retrieved_info}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
loader = TextLoader("hotel_infos.txt", encoding="utf-8")
documents = loader.load()
vectorstore = FAISS.from_documents(documents, OllamaEmbeddings(model="llama3"))
qa = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())

def speak_pyttsx3(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Google API error.")
            return None

def speak_elevenlabs(text):
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)

def handle_conversation():
    context = ""
    print("Welcome to the AI Call Center! Type 'exit' to quit.")   
    while True:
        user_input = recognize_speech()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Conversation is ended.")
            break
        retrieved_info = qa.invoke(user_input)["result"]
        result = chain.invoke({
            "context": context,
            "retrieved_info": retrieved_info,
            "question": user_input
        })

        bot_response = result
        print("Bot:", bot_response)
        speak_elevenlabs(bot_response)
        context += f"\nUser: {user_input}\nAI: {bot_response}"

handle_conversation()