import os
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import json
from pathlib import Path
import logging
import time

# Configurar logging
logger = logging.getLogger(__name__)

# Configuración de las credenciales de Google
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

def init_google_auth():
    """Inicializa la autenticación de Google."""
    if 'google_auth' not in st.session_state:
        st.session_state['google_auth'] = None
    
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = None


def get_google_flow():
    """Crea y retorna un objeto Flow para la autenticación de Google."""
    client_config = {
        "web": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")],
        }
    }
    
    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI")
    )
    return flow


def get_user_info(credentials):
    """Obtiene la información del usuario usando las credenciales de Google."""
    try:
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        logger.debug(f"Usuario obtenido: {user_info}")
        return user_info
    except Exception as e:
        logger.error(f"Error al obtener información del usuario: {str(e)}")
        st.error(f"Error al obtener información del usuario: {str(e)}")
        return None


def save_user_info(user_info):
    """
    Guarda la información del usuario en 'users.json'.
    Retorna True si se guardó correctamente, False si hubo error.
    """
    try:
        # Usamos una ruta relativa al directorio actual
        users_file = Path("users.json").resolve()
        
        logger.info(f"Intentando guardar usuario en: {users_file}")
        
        # Si el archivo no existe, lo creamos con un diccionario vacío
        if not users_file.exists():
            with open(users_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
            logger.info("Archivo users.json creado")
            
        # Leemos el contenido actual
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Contenido actual del archivo: {data}")
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Archivo corrupto o no encontrado, iniciando con diccionario vacío")
            data = {}
        
        # Verificamos que el email exista en el user_info
        if "email" not in user_info:
            logger.error("Email no encontrado en user_info")
            return False
            
        email = user_info["email"]
        
        # Actualizamos o creamos la entrada de este email
        data[email] = {
            "name": user_info.get("name", ""),
            "picture": user_info.get("picture", ""),
            "email": email
        }
        
        # Intentamos escribir el archivo
        try:
            with open(users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Datos escritos en el archivo: {data}")
        except Exception as e:
            logger.error(f"Error escribiendo el archivo: {str(e)}")
            return False
            
        # Verificación post-guardado
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if email in saved_data:
                    logger.info("Verificación exitosa: usuario encontrado en archivo")
                else:
                    logger.error("Verificación fallida: usuario no encontrado en archivo")
                    return False
        except Exception as e:
            logger.error(f"Error en verificación post-guardado: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar información del usuario: {str(e)}")
        logger.error(f"Ruta del archivo: {users_file}")
        st.error(f"Error al guardar información del usuario: {str(e)}")
        return False


def is_user_authenticated():
    """Verifica si el usuario está autenticado."""
    return (
        st.session_state.get('google_auth') is not None
        and st.session_state.get('user_info') is not None
    )


def authenticate_user():
    """
    Maneja el proceso de autenticación de Google.
    Ahora permite que cualquier usuario con cuenta de Google se loguee,
    sin filtrar por ALLOWED_USERS.
    """
    query_params = st.query_params
    if "code" in query_params:
        try:
            flow = get_google_flow()
            flow.fetch_token(code=query_params["code"])
            credentials = flow.credentials
            
            user_info = get_user_info(credentials)
            logger.info(f"User info received: {user_info}")
            
            if user_info and "email" in user_info:
                # Solo guardamos las credenciales y la información en la sesión
                st.session_state['google_auth'] = credentials
                st.session_state['user_info'] = user_info
                logger.info(f"User info saved in session: {st.session_state['user_info']}")
                time.sleep(0.5)
                return True
            else:
                st.error("No se pudo obtener la información del usuario")
        except Exception as e:
            st.error(f"Error de autenticación: {str(e)}")
            logger.error(f"Error de autenticación: {str(e)}")
    return False
