import streamlit as st
import os
import uuid
import requests
import logging
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from auth import is_user_authenticated
from pathlib import Path

# Redirect to login if not authenticated
if not is_user_authenticated():
    st.switch_page("app.py")

# ===================== Helper Functions =====================
def call_chat_api(messages):
    try:
        payload = {
            "messages": messages,
            "config": {
                "configurable": {
                    "thread_id": str(uuid.uuid4())
                }
            }
        }
        response = requests.post(f"{BACKEND_API_URL}/chat", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        return None

# ===================== Constants =====================
ASSISTANT_LOGO = str(Path(__file__).parent.parent / "assets" / "assistant_logo.png")
USERS_FILE = str(Path(__file__).parent.parent / "users.json")

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="G4 Investor ChatBot",
    page_icon=ASSISTANT_LOGO,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

# ===================== Load Environment Variables =====================
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BACKEND_API_URL = "https://g4investor.tech/api"

# ===================== Custom Styles =====================
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .main > div {
        padding-top: 1rem;
        max-width: 800px !important;
        margin: 0 auto;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stImage"] > img {
        border-radius: 50%;
    }
    .user-profile {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .user-info {
        flex: 1;
        min-width: 0;
    }
    .user-name {
        font-weight: bold;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .user-email {
        color: #CCCCCC;
        font-size: 0.9rem;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .logout-button {
        width: 100%;
        padding: 0.5rem;
        background-color: rgba(255,255,255,0.05);
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .logout-button:hover {
        background-color: rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.2);
    }
    .disclaimer {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.05);
        font-size: 0.9rem;
        color: #CCCCCC;
    }
    .divider {
        margin: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .past-users-title {
        color: #CCCCCC;
        font-size: 1.1rem;
        margin: 1rem 0;
        text-align: center;
    }
    @media (max-width: 768px) {
        .user-profile {
            padding: 0.75rem;
        }
        .user-info {
            font-size: 0.9rem;
        }
        .user-email {
            font-size: 0.8rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ===================== Sidebar =====================
with st.sidebar:
    user_info = st.session_state.get('user_info', {})
    
    # User Profile Section
    if user_info and 'picture' in user_info:
        st.markdown("""
            <div class="user-profile">
                <img src="{}" width="40" style="border-radius: 50%;">
                <div class="user-info">
                    <p class="user-name">{}</p>
                    <p class="user-email">{}</p>
                </div>
            </div>
        """.format(
            user_info.get('picture', ''),
            user_info.get('name', ''),
            user_info.get('email', '')
        ), unsafe_allow_html=True)
    
    # Logout button
    if st.button("Logout", use_container_width=True, type="secondary"):
        st.session_state['google_auth'] = None
        st.session_state['user_info'] = None
        st.switch_page("app.py")
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Past Users Section
    st.markdown('<p class="past-users-title">Past Users</p>', unsafe_allow_html=True)
    
    # Load and display past users
    try:
        with open(USERS_FILE, 'r') as f:
            past_users = json.load(f)
            
        current_email = user_info.get('email', '')
        
        # Display all users except the current one
        for email, user_data in past_users.items():
            if email != current_email:
                st.markdown("""
                    <div class="user-profile">
                        <img src="{}" width="40" style="border-radius: 50%;">
                        <div class="user-info">
                            <p class="user-name">{}</p>
                            <p class="user-email">{}</p>
                        </div>
                    </div>
                """.format(
                    user_data.get('picture', ''),
                    user_data.get('name', ''),
                    user_data.get('email', '')
                ), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading past users: {str(e)}")
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong><br>
            This is an educational project. We do not provide real financial information or advice. 
            All data and responses are for demonstration purposes only.
        </div>
    """, unsafe_allow_html=True)

# ===================== Main Content =====================
st.title("G4 Investor ChatBot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "charts" not in st.session_state:
    st.session_state["charts"] = []

# Display messages and charts
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        with st.chat_message("user", avatar=user_info.get('picture')):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_LOGO):
            st.write(msg["content"])
    
    if msg["role"] == "assistant" and i < len(st.session_state.get("charts", [])):
        if st.session_state["charts"][i] is not None:
            chart_data = st.session_state["charts"][i]
            fig = go.Figure(chart_data)
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"historic_chart_{i}")

# Chat input
prompt = st.chat_input("Type your question here...")
if prompt:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_info.get('picture')):
        st.write(prompt)
    st.session_state["charts"].append(None)
    
    # Get bot response
    with st.spinner("G4 Investor is thinking..."):
        try:
            response = call_chat_api(st.session_state["messages"])
            
            if response and "response" in response:
                formatted_response = response["response"]
                st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
                
                with st.chat_message("assistant", avatar=ASSISTANT_LOGO):
                    st.write(formatted_response)
                
                try:
                    if response.get("has_plot"):
                        figure_dict = json.loads(response["plot_data"])
                        st.session_state["charts"].append(figure_dict)
                        
                        fig = go.Figure(figure_dict)
                        fig.update_layout(
                            height=600,
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            yaxis_title="Price (USD)",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"new_chart_{len(st.session_state['charts'])-1}")
                    else:
                        st.session_state["charts"].append(None)
                except Exception as e:
                    st.error(f"Error displaying chart: {str(e)}")
                    st.session_state["charts"].append(None)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}") 