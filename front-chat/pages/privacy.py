import streamlit as st

# Hide sidebar in this page
st.set_page_config(
    page_title="G4 Investor | Privacy Policy",
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
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Privacy Policy")
st.markdown("<p style='text-align: center; color: #CCCCCC; margin-bottom: 3rem;'>Last updated: February 2024</p>", unsafe_allow_html=True)

# Section 1: Information We Collect
with st.container():
    st.header("1. Information We Collect")
    st.write("We collect information to provide better services to our users. This includes:")
    st.markdown("""
    - Account information (email, name) when you sign up
    - Usage data and interaction with our chatbot
    - Technical information about your device and connection
    - Financial information you choose to share during conversations
    """)

# Section 2: How We Use Your Information
with st.container():
    st.header("2. How We Use Your Information")
    st.write("We use the collected information for:")
    st.markdown("""
    - Providing and improving our services
    - Personalizing your experience
    - Analyzing usage patterns to enhance our chatbot
    - Ensuring security and preventing fraud
    """)

# Section 3: Data Protection
with st.container():
    st.header("3. Data Protection")
    st.write("We implement strong security measures to protect your data:")
    st.markdown("""
    - Encryption of sensitive information
    - Regular security audits and updates
    - Strict access controls for our team
    - Compliance with data protection regulations
    """)

# Section 4: Your Rights
with st.container():
    st.header("4. Your Rights")
    st.write("You have the right to:")
    st.markdown("""
    - Access your personal data
    - Request corrections or deletion
    - Opt out of certain data collection
    - Export your data
    """)

# Footer
st.markdown("""
    <div class="policy-footer">
        <p>For questions about our privacy practices, please contact our support team.</p>
    </div>
""", unsafe_allow_html=True) 