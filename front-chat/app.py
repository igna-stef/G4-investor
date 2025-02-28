import os
import logging
import streamlit as st
from dotenv import load_dotenv
from auth import init_google_auth, get_google_flow, is_user_authenticated, authenticate_user, save_user_info
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# ===================== Logging Configuration =====================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/g4investor.log')
    ]
)
logger = logging.getLogger(__name__)

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="G4 Investor | Login",
    page_icon="📈",
    layout="centered"
)

# ===================== Load Environment Variables =====================
load_dotenv()

# ===================== Constants =====================
BASE_DIR = Path(__file__).parent.absolute()
ASSISTANT_LOGO = str(BASE_DIR / "assets" / "assistant_logo.png")
USERS_FILE = str(BASE_DIR / "users.json")  # Usar ruta relativa

# Logging inicial de rutas y ambiente
logger.info("=== Application Startup ===")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"__file__ value: {__file__}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Users file path: {USERS_FILE}")
logger.info(f"Environment variables loaded: {list(os.environ.keys())}")

# ===================== Initialize Files =====================
def initialize_users_file():
    try:
        logger.info("=== Initializing Users File ===")
        # Asegurarnos de que el directorio base existe
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        
        if not os.path.exists(USERS_FILE):
            logger.info(f"Creating new users file at {USERS_FILE}")
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f, indent=2)
            # Asegurar permisos correctos
            os.chmod(USERS_FILE, 0o666)
            logger.info("Users file created successfully")
        else:
            logger.info("Users file already exists")
            # Verificar que el archivo sea válido
            try:
                with open(USERS_FILE, 'r') as f:
                    current_users = json.load(f)
                    logger.info(f"Users file is valid JSON with {len(current_users)} users")
                    logger.debug(f"Current users: {list(current_users.keys())}")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in users file - resetting")
                with open(USERS_FILE, 'w') as f:
                    json.dump({}, f, indent=2)
                logger.info("Users file reset successfully")
            
            # Asegurar permisos correctos
            os.chmod(USERS_FILE, 0o666)
    except Exception as e:
        logger.error(f"Error initializing users file: {str(e)}", exc_info=True)

# Inicializar archivos al arranque
initialize_users_file()

# ===================== Initialize Authentication =====================
init_google_auth()

# ===================== Email Functions =====================
def send_welcome_email(user_email, user_name):
    try:
        # Verificar configuración de correo
        email_sender = os.getenv("EMAIL_SENDER")
        email_password = os.getenv("EMAIL_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")

        # Logging detallado de configuración
        logger.info("=== Email Configuration ===")
        logger.info(f"SMTP Server: {smtp_server}")
        logger.info(f"SMTP Port: {smtp_port}")
        logger.info(f"Sender: {email_sender}")
        logger.info(f"Recipient: {user_email}")
        logger.info(f"All required vars present: {all([email_sender, email_password, smtp_server, smtp_port])}")

        if not all([email_sender, email_password, smtp_server, smtp_port]):
            missing_vars = [var for var, val in {
                "EMAIL_SENDER": email_sender,
                "EMAIL_PASSWORD": "***" if email_password else None,
                "SMTP_SERVER": smtp_server,
                "SMTP_PORT": smtp_port
            }.items() if not val]
            raise ValueError(f"Missing email configuration: {', '.join(missing_vars)}")

        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = email_sender
        msg['To'] = user_email
        msg['Subject'] = "¡Welcome to G4 Investor!"

        body = f'''
        <html>
        <body>
            <p><strong>Welcome {user_name} to <span style="color:#007bff;">G4 Investor</span>!</strong></p>
            <p>We are thrilled to have you onboard. With <strong>G4 Investor</strong>, you can leverage the power of AI to make informed financial decisions.</p>
            
            <h3>🔍 Explore Our Features:</h3>
            <ul>
                <li><strong>Try some silly questions:</strong> "Who is Isaac Newton? Was he good at investing?", or “whats the weather in buenos aires?”</li>
                <li><strong>Investment Guidance:</strong> "What investing advice would you give me during my 30's?"</li>
                <li><strong>Stock Market Data:</strong> "Give me AAPL stock price." or “Give me the stock price of google”</li>
                <li><strong>Real-time Charts:</strong> "Show me a chart for NVDA."</li>
                <li><strong>Macroeconomic Impact:</strong> "What is the effect of interest rates on the NASDAQ100 index after Trump's election?"</li>
                <li><strong>Deep Financial Insights using +1000 PDF database:</strong> "What challenges did Ark Restaurants face in fiscal 2022 due to global events?" or “What is the effect of the interest rate on the NASDAQ100 index price after Trump's election? only reply using the documents in our database or "From the NASDAQ100 companies, which would be the top five companies that you recommend me to invest in? "</li>
                <li><strong>Technical Analysis:</strong> "Can you show me a technical analysis of NVDA?"</li>
            </ul>
            
            <h3>📈 Start Exploring Today!</h3>
            <p>Simply log in and ask your first question. Whether you're looking for market trends, financial insights, or investment advice, G4 Investor is here to help.</p>
            
            <p><strong><a href="https://g4investor.tech" style="color:#007bff; text-decoration:none;">➡️ Get Started Now</a></strong></p>
            
            <p>Happy investing,</p>
            <p><strong>The G4 Investor Team</strong></p>
        </body>
        </html>
        '''
        
        msg.attach(MIMEText(body, 'html'))

        # Enviar correo con manejo de errores detallado
        try:
            logger.info("Initiating SMTP connection...")
            with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
                logger.debug("Starting TLS...")
                server.starttls()
                
                logger.debug("Attempting login...")
                server.login(email_sender, email_password)
                
                logger.debug("Sending email...")
                server.send_message(msg)
                
                logger.info(f"Welcome email sent successfully to {user_email}")
                return True
                
        except smtplib.SMTPAuthenticationError as e:
            logger.error("SMTP Authentication failed. Check credentials.", exc_info=True)
            st.error("Error de autenticación al enviar el correo. Por favor contacta al administrador.")
            return False
            
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error occurred: {str(e)}", exc_info=True)
            st.error("Error al enviar el correo de bienvenida. Por favor intenta más tarde.")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in send_welcome_email: {str(e)}", exc_info=True)
        st.error("Error inesperado al enviar el correo de bienvenida.")
        return False

def is_new_user(email):
    try:
        if not email:
            logger.warning("No email provided to is_new_user")
            return False

        logger.info(f"=== Checking New User Status ===")
        logger.info(f"Checking user status for email: {email}")
        logger.info(f"Looking for users file at: {USERS_FILE}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Verificar si el archivo existe y sus permisos
        file_exists = os.path.exists(USERS_FILE)
        logger.info(f"Users file exists: {file_exists}")
        
        if file_exists:
            # Verificar permisos
            permissions = oct(os.stat(USERS_FILE).st_mode)[-3:]
            logger.info(f"File permissions: {permissions}")
            
            # Verificar si es legible
            if not os.access(USERS_FILE, os.R_OK):
                logger.error("File exists but is not readable")
                return True

        if not file_exists:
            logger.info("Users file not found - treating as new user")
            return True

        # Leer el archivo de usuarios
        try:
            with open(USERS_FILE, 'r') as f:
                content = f.read()
                logger.debug(f"Raw file content: {content}")
                if not content.strip():
                    logger.warning("Empty users file - treating as new user")
                    return True
                users = json.loads(content)
                
                # Verificar si el email está en los usuarios originales
                is_new = email not in users
                logger.info(f"Users in file: {list(users.keys())}")
                logger.info(f"Is new user result: {is_new}")
                return is_new

        except (IOError, OSError) as e:
            logger.error(f"Error reading file: {str(e)}")
            return True
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding users.json: {str(e)}")
        logger.info("JSON decode error - treating as new user")
        return True
    except Exception as e:
        logger.error(f"Unexpected error in is_new_user: {str(e)}", exc_info=True)
        logger.info("Error checking user status - treating as new user")
        return True

def update_users_file(user_info):
    try:
        email = user_info.get('email')
        if not email:
            logger.warning("No email provided for user update")
            return False

        logger.info(f"=== Updating Users File ===")
        logger.info(f"Updating file for email: {email}")
        
        # Asegurarnos de que el archivo existe
        if not os.path.exists(USERS_FILE):
            logger.info("Creating new users file")
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f)

        # Leer usuarios existentes
        with open(USERS_FILE, 'r') as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in users file - starting fresh")
                users = {}

        # Actualizar información del usuario
        users[email] = {
            "name": user_info.get('name', ''),
            "picture": user_info.get('picture', ''),
            "email": email
        }

        # Guardar archivo actualizado
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
            logger.info(f"Users file updated successfully for {email}")
            logger.debug(f"Current users in file: {list(users.keys())}")
        return True

    except Exception as e:
        logger.error(f"Error updating users file: {str(e)}", exc_info=True)
        return False

# ===================== Login UI =====================
def main():
    try:
        # Get Google auth URL
        flow = get_google_flow()
        auth_url, _ = flow.authorization_url(prompt='consent')

        # Container principal
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 1rem;'>G4 Investor</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #CCCCCC; font-size: 1.1rem; margin-bottom: 2rem;'>Log in with your Google account to continue</p>", unsafe_allow_html=True)
                
                # Botón de Google
                st.markdown(f"""
                    <a href="{auth_url}" target="_self" style="text-decoration: none;">
                        <button style="
                            width: 100%;
                            background-color: #4285F4;
                            color: white;
                            border: none;
                            border-radius: 6px;
                            padding: 0.75rem 1.5rem;
                            font-size: 1rem;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 0.75rem;
                            margin: 1rem 0;
                        ">
                            <img src="https://www.google.com/favicon.ico" style="width: 18px; height: 18px;">
                            Continue with Google
                        </button>
                    </a>
                """, unsafe_allow_html=True)
                
                # Enlaces
                st.markdown("""
                    <div style="
                        display: flex;
                        justify-content: center;
                        gap: 2rem;
                        margin-top: 1rem;
                    ">
                        <a href="/terms" target="_self" style="color: #666; text-decoration: none; font-size: 0.9rem;">Terms of Service</a>
                        <a href="/privacy" target="_self" style="color: #666; text-decoration: none; font-size: 0.9rem;">Privacy Policy</a>
                    </div>
                """, unsafe_allow_html=True)

                 # Handle Google OAuth
                if authenticate_user():
                    logger.info("=== Authentication Process ===")
                    logger.info("User authenticated successfully")
                    
                    # Obtener información del usuario
                    user_info = st.session_state.get('user_info', {})
                    user_email = user_info.get('email')
                    user_name = user_info.get('name')

                    logger.info(f"User info retrieved: {user_info}")
                    logger.info(f"Email: {user_email}, Name: {user_name}")

                    if user_email and user_name:
                        try:
                            # Verificar si es un usuario nuevo ANTES de guardarlo
                            logger.info("Checking if user is new...")
                            is_new = is_new_user(user_email)
                            logger.info(f"Is new user check result: {is_new}")

                            if is_new:
                                logger.info(f"=== New User Process ===")
                                logger.info(f"Starting welcome process for: {user_email}")
                                
                                # Verificar configuración de correo antes de intentar enviar
                                email_config = {
                                    "smtp_server": os.getenv("SMTP_SERVER"),
                                    "smtp_port": os.getenv("SMTP_PORT"),
                                    "email_sender": os.getenv("EMAIL_SENDER"),
                                    "has_password": bool(os.getenv("EMAIL_PASSWORD"))
                                }
                                logger.info(f"Email configuration: {email_config}")
                                
                                # Intentar enviar el correo primero
                                email_sent = send_welcome_email(user_email, user_name)
                                logger.info(f"Welcome email sent: {email_sent}")

                                if email_sent:
                                    # Solo si el correo se envió, actualizar el archivo de usuarios
                                    if save_user_info(user_info):
                                        logger.info("Users file updated successfully")
                                        st.success(f"¡Bienvenido {user_name}! Te hemos enviado un correo de bienvenida.")
                                    else:
                                        logger.error("Failed to update users file")
                                        st.warning("Tu cuenta se creó pero hubo un problema actualizando la base de usuarios.")
                                else:
                                    logger.warning("Failed to send welcome email")
                                    st.warning("No pudimos enviarte el correo de bienvenida, pero puedes continuar usando la aplicación.")
                            else:
                                # Si no es nuevo, actualizar la información del usuario
                                save_user_info(user_info)
                                logger.info(f"Existing user logged in: {user_email}")
                                st.success(f"¡Bienvenido de vuelta, {user_name}!")

                        except Exception as e:
                            logger.error(f"Error in new user process: {str(e)}", exc_info=True)
                            st.error("Hubo un problema procesando tu registro. Por favor intenta más tarde.")
                    else:
                        logger.warning("User info incomplete", extra={"user_info": user_info})
                        st.error("No pudimos obtener tu información completa. Por favor intenta más tarde.")

                    # Pequeña pausa para mostrar los mensajes antes de redirigir
                    time.sleep(2)
                    st.switch_page("pages/chat.py")
       
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        st.error("Ha ocurrido un error inesperado. Por favor intenta más tarde.")

if __name__ == "__main__":
    main()

