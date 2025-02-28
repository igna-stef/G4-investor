import streamlit as st

# Hide sidebar in this page
st.set_page_config(
    page_title="G4 Investor | Terms of Service",
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
    .policy-section {
        background-color: #1E1E1E;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2.5rem;
    }
    .policy-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #333;
        color: #999;
        font-size: 0.9rem;
        text-align: center;
    }
    .subheader {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Terms of Service")
st.markdown("<p style='text-align: center; color: #CCCCCC; margin-bottom: 3rem;'>Last updated: February 2024</p>", unsafe_allow_html=True)

# Section 1: Introduction
with st.container():
    st.header("1. Introduction")
    st.write("Welcome to our G4 Investor ChatBot. By using our service, you agree to these terms. Please read them carefully.")
    
    st.markdown('<p class="subheader">1.1 Acceptance of Terms</p>', unsafe_allow_html=True)
    st.write("By accessing or using our service, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service.")
    
    st.markdown('<p class="subheader">1.2 Changes to Terms</p>', unsafe_allow_html=True)
    st.write("We reserve the right to modify these terms at any time. We'll notify you of significant changes by posting an update here.")

# Section 2: Service Description
with st.container():
    st.header("2. Service Description")
    st.write("G4 Investor ChatBot provides automated financial information and guidance. However, please note:")
    st.markdown("""
    - This is an educational tool only
    - We do not provide professional financial advice
    - You should consult with qualified professionals for specific advice
    - Past performance is not indicative of future results
    """)

# Section 3: User Responsibilities
with st.container():
    st.header("3. User Responsibilities")
    st.write("As a user of our service, you agree to:")
    st.markdown("""
    - Provide accurate information
    - Use the service responsibly
    - Not attempt to manipulate or abuse the system
    - Maintain the confidentiality of your account
    """)

# Footer
st.markdown("""
    <div class="policy-footer">
        <p>For questions about these terms, please contact our support team.</p>
    </div>
""", unsafe_allow_html=True) 