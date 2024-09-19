import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🎓 Class Notes Manager")

# Sidebar for class management
st.sidebar.title("Class Management")

# Initialize session state for classes if not exists
if 'classes' not in st.session_state:
    st.session_state.classes = []

# Add new class
new_class = st.sidebar.text_input("Add a new class:")
if st.sidebar.button("Add Class"):
    if new_class and new_class not in st.session_state.classes:
        st.session_state.classes.append(new_class)
        st.sidebar.success(f"Added {new_class}")

# Display existing classes
st.sidebar.subheader("Existing Classes")
selected_class = st.sidebar.selectbox("Select a class:", [""] + st.session_state.classes)

# OpenAI API Key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

# Main content area
if selected_class:
    st.subheader(f"Notes for {selected_class}")
    
    # Here you can add functionality for recording, transcribing, and managing notes
    
    # Example: Text input for notes
    note = st.text_area("Add a note:")
    if st.button("Save Note"):
        # Here you would save the note to the selected class
        st.success("Note saved!")
    
    # Example: AI-TA chat
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
