from flask import Flask, jsonify, request, session
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob
import sqlite3
from datetime import datetime
from functools import wraps
import hashlib
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Database initialization
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  user_id INTEGER NOT NULL,
                  auth_token TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# Model setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def generate_response(system_prompt, user_message, chat_history=[]):
    messages = [{"role": "system", "content": system_prompt}]
    for hist_msg in chat_history:
        messages.append({"role": "user", "content": hist_msg[0]})
        messages.append({"role": "assistant", "content": hist_msg[1]})
    messages.append({"role": "user", "content": user_message})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# Trulience REST Endpoints
@app.route('/api/', methods=['POST'])
def handle_request():
    data = request.get_json()
    action = data.get("action")
    session_id = data.get("sessionId")
    user_id = data.get("userId")
    auth_token = data.get("authToken")
    
    if action == "LOGIN":
        return handle_login(session_id, user_id, auth_token, data)
    elif action == "CHAT":
        return handle_chat(session_id, data.get("message"))
    elif action == "LOGOUT":
        return handle_logout(session_id)
    else:
        return jsonify({"status": "FAIL", "statusMessage": "Invalid action"}), 400

def handle_login(session_id, user_id, auth_token, data):
    # Store session and user information
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO sessions (session_id, user_id, auth_token)
                 VALUES (?, ?, ?)''', (session_id, user_id, auth_token))
    conn.commit()
    conn.close()
    return jsonify({
        "sessionId": session_id,
        "status": "OK",
        "statusMessage": "Session Created"
    }), 201

def handle_chat(session_id, user_message):
    # Retrieve chat history for the session
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT user_message, bot_response 
                 FROM chats 
                 WHERE session_id = ? 
                 ORDER BY timestamp DESC LIMIT 5''', (session_id,))
    recent_chats = c.fetchall()
    conn.close()
    
    # System prompt for the chatbot
    system_prompt = """You are Amara, a chatbot created to help. Be helpful, truthful, and attentive to emotions. 
    Detect sentiments in the conversation and provide solutions if necessary. Keep it short and concise."""

    # Generate bot response
    try:
        bot_response = generate_response(
            system_prompt=system_prompt,
            user_message=user_message,
            chat_history=list(reversed(recent_chats))
        )
        sentiment_score = get_sentiment(user_message)

        # Store chat
        store_chat(session_id, user_message, bot_response, sentiment_score)

        return jsonify({
            "sessionId": session_id,
            "reply": bot_response,
            "status": "OK",
            "statusMessage": "Reply Sent"
        }), 200
    except Exception as e:
        return jsonify({"status": "FAIL", "statusMessage": f"Error: {str(e)}"}), 500

def handle_logout(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
    return jsonify({
        "sessionId": session_id,
        "status": "OK",
        "statusMessage": "Session Ended"
    }), 200

def store_chat(session_id, user_message, bot_response, sentiment_score):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO chats (session_id, user_message, bot_response, sentiment_score)
                 VALUES (?, ?, ?, ?)''', (session_id, user_message, bot_response, sentiment_score))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
