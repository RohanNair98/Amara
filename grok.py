import os
import sqlite3
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
import hashlib
from textblob import TextBlob
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(api_key='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')

app = Flask(__name__)
app.secret_key = os.urandom(24)

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

# Decorator to require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
    
    return render_template('registration.html')

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
    
    context = "Previous conversation:\n"
    for user_msg, bot_msg in reversed(recent_chats):
        context += f"User: {user_msg}\nAmara: {bot_msg}\n"
    
    full_prompt = f"{context}\nUser: {user_message}\nAmara:"

    try:
        bot_response = generate_response(full_prompt)
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
