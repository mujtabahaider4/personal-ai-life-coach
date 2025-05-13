import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime
import os
import pandas as pd
import speech_recognition as sr
import pyttsx3
from textblob import TextBlob
import plotly.express as px
import sqlite3
import smtplib
import time

st.set_page_config(page_title="🧘 Personal AI Life Coach", layout="wide", page_icon="🧠")
st.title("🧘 Personal AI Life Coach")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme = st.radio("🌙 Select Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state.theme))

if theme != st.session_state.theme:
    st.session_state.theme = theme

if st.session_state.theme == "Dark":
    st.markdown("""
    <style>
        .reportview-container {
            background: #1E1E1E;
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            font-weight: bold;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.session_state.theme = "Light"
    st.markdown("""
    <style>
        .reportview-container {
            background: #f5f5f5;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            font-weight: bold;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

engine = pyttsx3.init()

conn = sqlite3.connect("user_data.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users
             (
                 id
                 INTEGER
                 PRIMARY
                 KEY
                 AUTOINCREMENT,
                 name
                 TEXT,
                 join_date
                 TEXT
             )""")
c.execute("""CREATE TABLE IF NOT EXISTS moods
(
    id
    INTEGER
    PRIMARY
    KEY
    AUTOINCREMENT,
    user_id
    INTEGER,
    timestamp
    TEXT,
    mood
    TEXT,
    FOREIGN
    KEY
             (
    user_id
             ) REFERENCES users
             (
                 id
             )
    )""")
c.execute("""CREATE TABLE IF NOT EXISTS goals
(
    id
    INTEGER
    PRIMARY
    KEY
    AUTOINCREMENT,
    user_id
    INTEGER,
    goal
    TEXT,
    progress
    INTEGER,
    FOREIGN
    KEY
             (
    user_id
             ) REFERENCES users
             (
                 id
             )
    )""")
c.execute("""CREATE TABLE IF NOT EXISTS habits
(
    id
    INTEGER
    PRIMARY
    KEY
    AUTOINCREMENT,
    user_id
    INTEGER,
    habit
    TEXT,
    streak
    INTEGER,
    FOREIGN
    KEY
             (
    user_id
             ) REFERENCES users
             (
                 id
             )
    )""")
conn.commit()


def send_daily_motivation(user_email):
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("youremail@example.com", "yourpassword")

        message = "Subject: Daily Motivation\n\nRemember, today is a new day to achieve your goals! Stay positive!"
        server.sendmail("youremail@example.com", user_email, message)
        server.quit()
        st.success("Daily motivation sent to your email!")
    except Exception as e:
        st.error(f"❌ Error sending email: {str(e)}")


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("#### This app helps you manage your mood, set goals, track habits, and get personalized advice.")
    user_name = st.text_input("👤 Enter your name to begin:")
    if user_name:
        c.execute("SELECT * FROM users WHERE name = ?", (user_name,))
        user = c.fetchone()
        if user is None:
            c.execute("INSERT INTO users (name, join_date) VALUES (?, ?)",
                      (user_name, datetime.now().strftime("%Y-%m-%d")))
            conn.commit()
            user = c.execute("SELECT * FROM users WHERE name = ?", (user_name,)).fetchone()
        st.session_state.user = user
        st.success(f"Welcome, {user_name}!")

        user_email = st.text_input("📧 Enter your email to receive daily motivation:")
        if user_email:
            send_daily_motivation(user_email)

user_id = st.session_state.user[0] if st.session_state.user else None

pdf_file = st.file_uploader("📄 Upload your Journal (PDF)", type=["pdf"])
txt_file = st.file_uploader("📝 Upload your Goals (TXT, optional)", type=["txt"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

habit_name = st.text_input("🏋️‍♂️ Add a new habit to track:")
if habit_name:
    c.execute("INSERT INTO habits (user_id, habit, streak) VALUES (?, ?, ?)", (user_id, habit_name, 0))
    conn.commit()
    st.success(f"✅ Habit '{habit_name}' has been added!")


def analyze_mood(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    mood = "Neutral"
    if polarity > 0:
        mood = "Positive"
    elif polarity < 0:
        mood = "Negative"
    c.execute("INSERT INTO moods (user_id, timestamp, mood) VALUES (?, ?, ?)",
              (user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mood))
    conn.commit()
    return mood


def generate_goal_suggestions():
    return "Based on your journal, you may want to try setting a goal to improve your physical health through daily exercise!"


if pdf_file:
    if pdf_file.size > 5 * 1024 * 1024:
        st.error("❌ PDF file is too large. Please upload a file smaller than 5MB.")
        st.stop()

    with open("temp_journal.pdf", "wb") as f:
        f.write(pdf_file.read())
    if txt_file:
        with open("temp_goals.txt", "wb") as f:
            f.write(txt_file.read())

    try:
        pdf_loader = PyPDFLoader("temp_journal.pdf")
        documents = pdf_loader.load()

        if txt_file:
            txt_loader = TextLoader("temp_goals.txt")
            documents += txt_loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = splitter.split_documents(documents)

        if not docs:
            st.error("❌ No content found in your uploaded files.")
            st.stop()

        embeddings = OllamaEmbeddings()
        vectorstore_path = "vectorstore"

        if not os.path.exists(vectorstore_path):
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(vectorstore_path)
        else:
            db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        llm = Ollama(model="mistral", base_url="http://localhost:11434")

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a kind, thoughtful, and motivational personal life coach.
Use the context to provide insightful, positive guidance.

Context:
{context}

Question:
{question}

Answer:"""
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False,
        )

        query = st.text_input("💬 Ask your life coach:")

        if query:
            context = "\n".join([entry for entry in st.session_state.chat_history[-3:]])
            with st.spinner("🤖 Generating response..."):
                response = qa.run(query)

            mood = analyze_mood(response)
            st.markdown(f"**🧠 Coach says ({mood}):** {response}")
            st.session_state.chat_history.append(f"Q: {query}\nA: {response}\n")

            goal_suggestion = generate_goal_suggestions()
            st.subheader("✨ AI-based Goal Suggestions")
            st.write(goal_suggestion)

            habit_df = pd.read_sql_query("SELECT habit, streak FROM habits WHERE user_id = ?", conn, params=(user_id,))
            if not habit_df.empty:
                fig = px.line(habit_df, x="habit", y="streak", title="Habit Streaks")
                st.plotly_chart(fig)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            st.download_button(
                label="💾 Download Coaching Session",
                data="\n".join(st.session_state.chat_history),
                file_name=f"life_coach_log_{timestamp}.txt",
                mime="text/plain"
            )

            st.success("✅ Session saved. Keep going on your journey!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("📌 Please upload your journal (PDF) to get started.")

# Voice Recognition for Mood Input
if st.button("🎤 Use Voice Input for Mood"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🔴 Speak now...")
        audio = recognizer.listen(source)
        try:
            voice_text = recognizer.recognize_google(audio)
            st.write(f"🎤 You said: {voice_text}")
            mood = analyze_mood(voice_text)
            st.success(f"🧠 Your mood is: {mood}")
        except Exception as e:
            st.error("❌ Sorry, could not recognize your speech. Try again.")

# Time Display (Updated every minute)
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"🕒 Current Time: {current_time}")
