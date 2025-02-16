import sqlite3
import os
from datetime import datetime

DB_PATH = "chroma_db/chat_history.db"

def init_db():
    """Initialize the database and create necessary tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         model TEXT NOT NULL,
         system_prompt TEXT,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    
    # Create messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         conversation_id INTEGER,
         role TEXT NOT NULL,
         content TEXT NOT NULL,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
         FOREIGN KEY (conversation_id) REFERENCES conversations (id))
    ''')
    
    conn.commit()
    conn.close()

def create_conversation(model, system_prompt):
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO conversations (model, system_prompt)
        VALUES (?, ?)
    ''', (model, system_prompt))
    
    conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return conversation_id

def add_message(conversation_id, role, content):
    """Add a message to a conversation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
    ''', (conversation_id, role, content))
    
    conn.commit()
    conn.close()

def get_conversation_history(conversation_id):
    """Get all messages from a conversation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    ''', (conversation_id,))
    
    messages = [{"role": role, "content": content} for role, content in c.fetchall()]
    conn.close()
    
    return messages

def get_recent_conversations(limit=5):
    """Get recent conversations with their first message."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT c.id, c.model, c.system_prompt, c.created_at,
               m.content as first_message
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE m.id IN (
            SELECT MIN(id)
            FROM messages
            GROUP BY conversation_id
        )
        ORDER BY c.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    conversations = [
        {
            "id": row[0],
            "model": row[1],
            "system_prompt": row[2],
            "created_at": row[3],
            "first_message": row[4]
        }
        for row in c.fetchall()
    ]
    
    conn.close()
    return conversations

def delete_conversation(conversation_id):
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
    
    conn.commit()
    conn.close()

# Initialize the database when the module is imported
init_db() 