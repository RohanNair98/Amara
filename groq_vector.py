import os
import sqlite3
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
import hashlib
from textblob import TextBlob
from groq import Groq
import chromadb
from chromadb.config import Settings

# Initialize the Groq client with your API key
client = Groq(api_key='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')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
chat_collection = chroma_client.get_or_create_collection(name="chat_history")

# Database initialization (only for user management; ChromaDB is used for chat history)
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
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

# Function to store chat data in ChromaDB
def store_chat(user_id, user_message, bot_response, sentiment_score):
    # Insert data into ChromaDB collection
    chat_collection.add(
        documents=[user_message],
        metadatas=[{
            "user_id": user_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "sentiment_score": sentiment_score
        }]
    )

def retrieve_recent_chats(user_id, limit=5):
    # Fetch all messages for the user from the collection
    results = chat_collection.query(
        where={"user_id": user_id}
    )
    
    # Extract and sort results by timestamp in descending order, then limit to `limit` entries
    recent_chats = sorted(
        [(doc['user_message'], doc['bot_response'], doc['timestamp']) for doc in results["metadatas"]],
        key=lambda x: x[2],  # Sort by timestamp (assuming it’s stored in doc['timestamp'])
        reverse=True
    )[:limit]

    # Return only user_message and bot_response fields
    return [(chat[0], chat[1]) for chat in recent_chats]


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
    
    # Get user's recent chat history for context from ChromaDB
    recent_chats = retrieve_recent_chats(user_id)
    
    context = "Previous conversation:\n"
    for user_msg, bot_msg in reversed(recent_chats):
        context += f"User: {user_msg}\nAmara: {bot_msg}\n"
    
    full_prompt = f"{context}\nUser: {user_message}\nAmara:"

    try:
        bot_response = generate_response(full_prompt)
        sentiment_score = get_sentiment(user_message)
        
        # Store chat in ChromaDB
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
    
    # Retrieve full chat history for the user from ChromaDB
    history = retrieve_recent_chats(user_id, limit=50)
    
    return jsonify([{
        'user_message': row[0],
        'bot_response': row[1],
    } for row in history])

if __name__ == '__main__':
    app.run(debug=True)
