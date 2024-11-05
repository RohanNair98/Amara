import os
import sqlite3
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from functools import wraps
import hashlib
from textblob import TextBlob
from groq import Groq
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Secret key for sessions
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))

# Initialize the Groq client with your API key
client = Groq(api_key='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')

# Database initialization
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  user_message TEXT NOT NULL,
                  bot_response TEXT NOT NULL,
                  sentiment_score REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def store_chat(user_id, user_message, bot_response, sentiment_score):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO chats (user_id, user_message, bot_response, sentiment_score)
                 VALUES (?, ?, ?, ?)''', (user_id, user_message, bot_response, sentiment_score))
    conn.commit()
    conn.close()

# New generate_response function using Groq API
def generate_response(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Amara, a chatbot created to help. Be helpful, truthful, and attentive to emotions. Consider the previous conversation context for personalized responses. Keep it short and concise."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Define request models
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: Request, message: UserMessage):
    user_message = message.message
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User not logged in")

    # Get user's recent chat history for context
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT user_message, bot_response 
                 FROM chats 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 5''', (user_id,))
    recent_chats = c.fetchall()
    conn.close()
    
    context = "Previous conversation:\n"
    for user_msg, bot_msg in reversed(recent_chats):
        context += f"User: {user_msg}\nAmara: {bot_msg}\n"
    
    full_prompt = f"{context}\nUser: {user_message}\nAmara:"

    try:
        bot_response = generate_response(full_prompt)
        sentiment_score = get_sentiment(user_message)
        store_chat(user_id, user_message, bot_response, sentiment_score)

        return JSONResponse({
            'response': bot_response,
            'sentiment': sentiment_score
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"I apologize, but I encountered an error: {str(e)}")

