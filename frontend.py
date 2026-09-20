import streamlit as st
from rag_pipeline import get_answer_text

# Page Configuration
st.set_page_config(
    page_title="AI Legal Assistant - Pakistani Criminal Law",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .stChatInput textarea {
        font-size: 1.1rem;
    }
    
    .stButton button {
        font-size: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chats" not in st.session_state:
    st.session_state.chats = []

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# Sidebar (Chat History)
with st.sidebar:
    st.markdown('<p class="sidebar-title">Chat History</p>', unsafe_allow_html=True)
    
    if st.button("New Conversation", use_container_width=True, type="primary"):
        st.session_state.chats.append([])
        st.session_state.active_chat = len(st.session_state.chats) - 1
        st.rerun()

    st.divider()
    
    if st.session_state.chats:
        for i, chat in enumerate(st.session_state.chats):
            if chat:
                title = chat[0]["content"][:35] + "..."
            else:
                title = f"Conversation {i + 1}"

            if st.button(title, key=f"chat_{i}", use_container_width=True):
                st.session_state.active_chat = i
                st.rerun()
    
    st.divider()
    
    with st.expander("About", expanded=False):
        st.markdown("""
        **AI Legal Assistant**
        
        This chatbot provides information on:
        - Pakistan Penal Code (PPC) 1860
        - Code of Criminal Procedure (CrPC) 1898
        
        **Note:** This is for informational purposes only and does not constitute legal advice.
        """)

# If no chat exists, create one
if st.session_state.active_chat is None:
    st.session_state.chats.append([])
    st.session_state.active_chat = 0

current_chat = st.session_state.chats[st.session_state.active_chat]

# Main Chat UI
st.markdown('<p class="main-title">AI Legal Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Expert guidance on Pakistani Criminal Law</p>', unsafe_allow_html=True)

# Display chat messages
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask your legal question here...")

if query:
    current_chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            answer = get_answer_text(query)
            st.markdown(answer)

    current_chat.append({"role": "assistant", "content": answer})
