from flask import Flask, jsonify, request, render_template, session, redirect, url_for
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
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Chats table with user_id
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

# Initialize database
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Model initialization
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Initialize model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully!")

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

def generate_response(system_prompt, user_message, chat_history=[]):
    # Prepare messages including chat history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for hist_msg in chat_history:
        messages.append({"role": "user", "content": hist_msg[0]})
        messages.append({"role": "assistant", "content": hist_msg[1]})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    # Post-process the generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat_interface'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                     (username, hash_password(password)))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists'
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and user[1] == hash_password(password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('chat_interface'))
        
        return 'Invalid username or password'
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat-interface')
@login_required
def chat_interface():
    return render_template('chat.html', username=session.get('username'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message', '')
    user_id = session['user_id']
    
    # Get user's recent chat history for context
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT user_message, bot_response 
                 FROM chats 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 5''', (user_id,))
    recent_chats = c.fetchall()
    conn.close()

    system_prompt = """You are Amara, a chatbot created to help. Be helpful, truthful, and attentive to emotions. 
    Detect sentiments in the conversation and provide solutions if necessary. Keep it short and concise."""

    try:
        bot_response = generate_response(
            system_prompt=system_prompt,
            user_message=user_message,
            chat_history=list(reversed(recent_chats))
        )
        
        sentiment_score = get_sentiment(user_message)
        store_chat(user_id, user_message, bot_response, sentiment_score)

        return jsonify({
            'response': bot_response,
            'sentiment': sentiment_score
        })
    except Exception as e:
        return jsonify({
            'response': f"I apologize, but I encountered an error: {str(e)}",
            'sentiment': 0
        }), 500

@app.route('/chat_history')
@login_required
def chat_history():
    user_id = session['user_id']
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM chats 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 50''', (user_id,))
    history = c.fetchall()
    conn.close()
    
    return jsonify([{
        'id': row[0],
        'user_message': row[2],
        'bot_response': row[3],
        'sentiment': row[4],
        'timestamp': row[5]
    } for row in history])

if __name__ == '__main__':
    app.run(debug=True)