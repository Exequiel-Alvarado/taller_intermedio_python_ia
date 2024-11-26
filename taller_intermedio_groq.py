import streamlit as st
import sounddevice as sd

import pandas as pd
import psycopg2
import requests

from scipy.io.wavfile import write

# -----------------------
# Función para inicializar la API Key
# -----------------------
def inicializar_api_key():
    st.sidebar.title("Configuración API")
    api_key = st.sidebar.text_input("Ingresa tu API Key de Groq:", type="password")
    if not api_key:
        st.sidebar.warning("Por favor, ingresa una API Key válida para continuar.")
    return api_key

# -----------------------
# Función para configurar los encabezados de las solicitudes
# -----------------------
def configurar_headers(api_key):
    return {
        'Authorization': f'Bearer {api_key}'
    }

# -----------------------
# Función para grabar audio desde el micrófono
# -----------------------
def grabar_audio(duracion=10, frecuencia_muestreo=16000):
    try:
        st.info("Grabando audio...")
        audio = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='int16')
        sd.wait()
        st.success("Grabación completada")

        # Guardar el audio en formato wav
        archivo_audio = "audio.wav"
        write(archivo_audio, frecuencia_muestreo, audio)
        return archivo_audio
    except Exception as e:
        st.error(f"Error al grabar audio: {e}")
        return None

# -----------------------
# Función para transcribir audio con Whisper (Groq)
# -----------------------
def transcribir_audio(archivo_audio, headers):
    url = 'https://api.groq.com/openai/v1/audio/transcriptions'
    files = {
        'file': (archivo_audio, open(archivo_audio, 'rb'), 'audio/wav')
    }
    data = {
        'model': 'whisper-large-v3-turbo',
        'language': 'es'
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        result = response.json()
        return result['text']
    else:
        raise Exception(f"Error en la transcripción: {response.status_code} - {response.text}")

# -----------------------
# Función para generar consulta SQL con Llama (Groq)
# -----------------------
def generar_consulta_sql(prompt, headers):
    url = 'https://api.groq.com/openai/v1/chat/completions'
    data = {
        'model': 'llama3-8b-8192',
        'messages': [
            {'role': 'system', 'content': 'Eres un modelo que genera exclusivamente consultas SQL PostgreSQL válidas. Responde solo con el código SQL.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 150,
        'temperature': 0.2,
        'stop': [';']
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Error al generar la consulta SQL: {response.status_code} - {response.text}")

# -----------------------
# Función para ejecutar la consulta SQL
# -----------------------
def ejecutar_consulta_sql(sql_query, db_params):
    try:
        conn = psycopg2.connect(**db_params)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al ejecutar la consulta SQL: {e}")
        return None

# -----------------------
# Función principal con Streamlit
# -----------------------
def main():
    st.title("Automatización de Consultas SQL con Voz, Whisper y Llama")

    # Configurar API Key
    api_key = inicializar_api_key()
    if not api_key:
        return

    headers = configurar_headers(api_key)

    # Parámetros de la base de datos
    with st.sidebar.expander("Configuración de la Base de Datos"):
        db_host = st.text_input("Host", value="localhost")
        db_name = st.text_input("Nombre de la Base de Datos", value="postgres")
        db_user = st.text_input("Usuario", value="postgres")
        db_password = st.text_input("Contraseña", type="password")
        db_params = {
            "host": db_host,
            "database": db_name,
            "user": db_user,
            "password": db_password
        }

    # Grabar audio
    if st.button("Grabar Audio"):
        archivo_audio = grabar_audio(duracion=10)
        if archivo_audio:
            try:
                # Transcribir el audio
                transcripcion = transcribir_audio(archivo_audio, headers)
                st.write("**Transcripción:**")
                st.write(transcripcion)

                # Generar la consulta SQL
                table_info = """
                La tabla 'ventas' tiene la siguiente estructura:
                id SERIAL PRIMARY KEY,
                producto VARCHAR(100),
                cantidad INT,
                precio DECIMAL(10, 2).
                """
                prompt = f"{table_info}\nGenera una consulta SQL PostgreSQL válida para responder a la solicitud: {transcripcion}"
                sql_query = generar_consulta_sql(prompt, headers)

                # Mostrar y permitir la edición de la consulta generada
                st.write("**Consulta SQL Generada:**")
                sql_query = st.text_area("Revisa o edita la consulta SQL antes de ejecutarla:", value=sql_query)

                # Validar la consulta antes de ejecutarla
                if sql_query.strip().lower().startswith("select"):
                    # Ejecutar la consulta SQL
                    df_resultado = ejecutar_consulta_sql(sql_query, db_params)
                    if df_resultado is not None:
                        st.write("**Resultado de la Consulta:**")
                        st.dataframe(df_resultado)

                        # Opción para descargar el resultado como CSV
                        csv = df_resultado.to_csv(index=False)
                        st.download_button(
                            label="Descargar resultado como CSV",
                            data=csv,
                            file_name='resultado_consulta.csv',
                            mime='text/csv',
                        )
                else:
                    st.error("La consulta generada no es válida. Por favor, edítala antes de ejecutarla.")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
