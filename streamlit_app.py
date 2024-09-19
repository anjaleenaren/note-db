import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import sqlite3
import datetime

load_dotenv()

# Database setup
@st.cache_resource
def init_db():
    conn = sqlite3.connect('class_notes.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS classes
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS notes
                 (id INTEGER PRIMARY KEY, class_id INTEGER, content TEXT, timestamp DATETIME,
                 FOREIGN KEY (class_id) REFERENCES classes(id))''')
    conn.commit()
    return conn

if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()

# Helper functions
def add_class(class_name):
    c = st.session_state.db_conn.cursor()
    c.execute("INSERT OR IGNORE INTO classes (name) VALUES (?)", (class_name,))
    st.session_state.db_conn.commit()

def get_classes():
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT name FROM classes")
    return [row[0] for row in c.fetchall()]

def add_note(class_name, note_content):
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT id FROM classes WHERE name = ?", (class_name,))
    class_id = c.fetchone()[0]
    c.execute("INSERT INTO notes (class_id, content, timestamp) VALUES (?, ?, ?)",
              (class_id, note_content, datetime.datetime.now()))
    st.session_state.db_conn.commit()

def get_notes(class_name):
    c = st.session_state.db_conn.cursor()
    c.execute("""SELECT notes.id, notes.content, notes.timestamp 
                 FROM notes 
                 JOIN classes ON notes.class_id = classes.id 
                 WHERE classes.name = ?
                 ORDER BY notes.timestamp DESC""", (class_name,))
    return c.fetchall()

def get_tables():
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    return [table[0] for table in tables]

# Add these new functions
def get_classes_data():
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT * FROM classes")
    return c.fetchall()

def get_notes_data():
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT * FROM notes")
    return c.fetchall()

# Add this new function
def clear_database():
    c = st.session_state.db_conn.cursor()
    c.execute("DELETE FROM notes")
    c.execute("DELETE FROM classes")
    # c.execute("DELETE FROM sqlite_sequence WHERE name IN ('notes', 'classes')")  # Reset auto-increment
    st.session_state.db_conn.commit()

# Add these new functions
def update_note(note_id, content):
    c = st.session_state.db_conn.cursor()
    c.execute("UPDATE notes SET content = ?, timestamp = ? WHERE id = ?",
              (content, datetime.datetime.now(), note_id))
    st.session_state.db_conn.commit()

def get_note_by_id(note_id):
    c = st.session_state.db_conn.cursor()
    c.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
    return c.fetchone()[0]

# Add this new function to delete a note
def delete_note(note_id):
    c = st.session_state.db_conn.cursor()
    c.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    st.session_state.db_conn.commit()

# Streamlit UI
st.title("🎓 Class Notes Manager")

# Add this new button for clearing the database
# if st.button("Clear Database"):
#     clear_database()
#     st.success("Database cleared successfully!")
#     st.experimental_rerun()

# Add this new button
if st.button("Show Database Tables"):
    tables = get_tables()
    st.write("Tables in the database:", tables)

# Add these new buttons
if st.button("Show Classes Data"):
    classes_data = get_classes_data()
    st.write("Classes table contents:")
    st.table(classes_data)

if st.button("Show Notes Data"):
    notes_data = get_notes_data()
    st.write("Notes table contents:")
    st.table(notes_data)

# Sidebar for class management
st.sidebar.title("Class Management")

# Add new class
new_class = st.sidebar.text_input("Add a new class:")
if st.sidebar.button("Add Class"):
    if new_class:
        add_class(new_class)
        st.sidebar.success(f"Added {new_class}")
        st.experimental_rerun()

# Display existing classes
st.sidebar.subheader("Existing Classes")
classes = get_classes()
selected_class = st.sidebar.selectbox("Select a class:", [""] + classes)

# OpenAI API Key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

# Main content area
if selected_class:
    st.subheader(f"Notes for {selected_class}")
    
    # Initialize session state for current note
    if 'current_note_id' not in st.session_state:
        st.session_state.current_note_id = None
    if 'current_note_content' not in st.session_state:
        st.session_state.current_note_content = ""

    # Text input for notes
    note_content = st.text_area("Edit note:", value=st.session_state.current_note_content)
    
    # Save button for the current note
    if st.button("Save Note"):
        if st.session_state.current_note_id:
            update_note(st.session_state.current_note_id, note_content)
            st.success("Note updated!")
        else:
            add_note(selected_class, note_content)
            st.success("New note added!")
        st.session_state.current_note_id = None
        st.session_state.current_note_content = ""
        st.experimental_rerun()
    
    # Display existing notes
    st.subheader("Existing Notes")
    for note_id, note_content, timestamp in get_notes(selected_class):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            date_only = timestamp.split()[0]  # Extract only the date part
            if st.button(f"{date_only}: {note_content[:50]}...", key=f"note_{note_id}"):
                if st.session_state.current_note_id:
                    update_note(st.session_state.current_note_id, note_content)
                st.session_state.current_note_id = note_id
                st.session_state.current_note_content = get_note_by_id(note_id)
                st.experimental_rerun()
        with col2:
            if st.button("Delete", key=f"delete_{note_id}"):
                delete_note(note_id)
                if st.session_state.current_note_id == note_id:
                    st.session_state.current_note_id = None
                    st.session_state.current_note_content = ""
                st.experimental_rerun()
        with col3:
            st.write("")  # Empty column for spacing

    # AI-TA chat
    st.subheader("AI TA Chat :)")
    if openai_api_key.startswith("sk-"):
        chat_input = st.text_input("Ask AI-TA a question about this class:")
        if st.button("Ask"):
            model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
            response = model.invoke(f"Question about {selected_class}: {chat_input}")
            st.info(response)
    else:
        st.warning("Please enter your OpenAI API key to use the AI-TA feature!", icon="⚠")
else:
    st.info("Please select a class from the sidebar or add a new one.")

# Close the database connection when the app is done
# st.on_script_run.add(st.session_state.db_conn.close)
