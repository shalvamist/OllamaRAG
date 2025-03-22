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
    
    # Check if chat_name column exists, if not add it
    c.execute("PRAGMA table_info(conversations)")
    columns = [column[1] for column in c.fetchall()]
    if 'chat_name' not in columns:
        c.execute('ALTER TABLE conversations ADD COLUMN chat_name TEXT')
    
    conn.commit()
    conn.close()

def create_conversation(model: str, system_prompt: str, chat_name: str = None):
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO conversations (model, system_prompt, chat_name, created_at)
        VALUES (?, ?, ?, datetime('now', 'localtime'))
    ''', (model, system_prompt, chat_name))
    
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

def get_recent_conversations():
    """Get list of recent conversations."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('''
            SELECT 
                c.id, 
                c.model, 
                c.created_at, 
                m.content as first_message,
                c.chat_name
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE m.id = (
                SELECT MIN(id)
                FROM messages
                WHERE conversation_id = c.id
            )
            ORDER BY c.created_at DESC
        ''')
        
        conversations = []
        for row in c.fetchall():
            # Format the date for display
            created_at = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
            formatted_date = created_at.strftime('%Y-%m-%d %H:%M')
            
            # Use chat_name if available, otherwise use first message or default text
            display_name = row[4] if row[4] else (row[3] if row[3] else "Untitled Chat")
            
            conversations.append({
                'id': row[0],
                'model': row[1],
                'created_at': row[2],
                'first_message': f"{display_name} ({formatted_date})"
            })
        
        return conversations
    except Exception as e:
        print(f"Error fetching conversations: {str(e)}")
        return []
    finally:
        conn.close()

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