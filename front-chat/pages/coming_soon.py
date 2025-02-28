import streamlit as st

# Hide sidebar in this page
st.set_page_config(
    page_title="G4 Investor | Coming Soon",
    page_icon="📈",
    layout="centered",
    menu_items=None
)

# Hide all menu items and sidebar
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none;
    }
    .main > div {
        max-width: 768px !important;
        padding: 4rem 0;
    }
    .policy-container {
        color: #FFFFFF;
        padding: 0 2rem;
    }
    .policy-header {
        margin-bottom: 3rem;
        text-align: center;
    }
    .policy-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .policy-header p {
        color: #CCCCCC;
        font-size: 1.1rem;
    }
    .policy-section {
        margin-bottom: 2.5rem;
        background-color: #1E1E1E;
        padding: 2rem;
        border-radius: 8px;
    }
    .policy-section p {
        color: #CCCCCC;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    .policy-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #333;
        color: #999;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Contenido principal usando los componentes nativos de Streamlit
st.title("Coming Soon")

with st.container():
    st.write("We are working on it:")
    st.markdown("""
        <div class='policy-section'>
            <p>Our team is working hard to bring you new features and improvements.</p>
            <p>We're focusing on creating the best possible experience for our users.</p>
            <p>Stay tuned for updates!</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class='policy-footer'>
        <p>Want to be notified when we launch? Contact our support team.</p>
    </div>
""", unsafe_allow_html=True) 