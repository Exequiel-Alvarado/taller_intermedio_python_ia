import streamlit as st
import sounddevice as sd
import numpy as np
import pandas as pd
import psycopg2
import time
import torch
import re
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)



# Configuración de dispositivos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

if DEVICE == "cpu":
    st.warning("No se detectó una GPU. El rendimiento puede ser más lento en CPU.")

def inicializar_variables_sesion():
    for var in ['audio_data', 'transcription', 'sql_query', 'db_connection', 'df_result']:
        if var not in st.session_state:
            st.session_state[var] = None
inicializar_variables_sesion()

# Inicialización de variables de sesión
for var in ['audio_data', 'transcription', 'sql_query', 'db_connection', 'df_result']:
    if var not in st.session_state:
        st.session_state[var] = None

# -----------------------
# Funciones para cargar modelos
# -----------------------
@st.cache_resource
def cargar_modelo_whisper():
    try:
        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
            st.error(f"Error al cargar el modelo Whisper: {e}")
            return None, None

@st.cache_resource
def cargar_modelo_llama():
    try:
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id
        ).to(DEVICE)
        return tokenizer, model
    except Exception as e:
            st.error(f"Error al cargar el modelo llama: {e}")
            return None, None

# -----------------------
# Función para transcribir audio con Whisper
# -----------------------
def transcribir_audio(audio_data, model, processor):
    start_time = time.time()
    try:
        with st.spinner('Transcribiendo el audio...'):
            inputs = processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                language="es"
            )
            attention_mask = (inputs["input_features"] != processor.feature_extractor.padding_value).long()
            inputs = inputs.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            outputs = model.generate(
                inputs["input_features"],
                attention_mask=attention_mask
            )
            transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        st.info(f"Transcripción completada en {time.time() - start_time:.2f} segundos")
        return transcription
    except Exception as e:
        st.error(f"Error en la transcripción: {e}")
        return None

# -----------------------
# Función para generar consulta SQL con Llama
# -----------------------
def generar_consulta_sql(prompt, tokenizer, model):
    start_time = time.time()
    try:
        with st.spinner('Generando la consulta SQL...'):
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                do_sample=True,

            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql_query = extraer_consulta_sql(generated_text)
        st.success(f"Consulta SQL generada en {time.time() - start_time:.2f} segundos")
        return sql_query
    except Exception as e:
        st.error(f"Error al generar la consulta SQL: {e}")
        return None

def extraer_consulta_sql(generated_text):
    code_block_match = re.search(r"```sql\n(.*?)\n```", generated_text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    match = re.search(r"(SELECT.*?);?\s*$", generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# -----------------------
# Función para grabar audio
# -----------------------
def grabar_audio(samplerate=16000, duration=10, device_index=None):
    try:
        device_index = device_index or sd.default.device[0]
        device_info = sd.query_devices(device_index, 'input')
        channels = min(device_info['max_input_channels'], 1)

        st.info("Grabando audio...")
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32')
        sd.wait()
        st.success("Grabación completa")
        return np.squeeze(audio_data)
    except Exception as e:
        st.error(f"Error al grabar audio: {e}")
        return None

# -----------------------
# Función principal
# -----------------------
def main():
    st.title("Automatización de Consultas SQL con Voz, Whisper y Llama")

    # Opciones en la barra lateral
    st.sidebar.title("Opciones")

    # Input de dispositivos de audio
    dispositivos = sd.query_devices()
    dispositivos_input = [
        f"{i}: {d['name']}" for i, d in enumerate(dispositivos) if d['max_input_channels'] > 0
    ]
    dispositivo_seleccionado = st.sidebar.selectbox("Selecciona tu micrófono:", dispositivos_input)
    indice_dispositivo = int(dispositivo_seleccionado.split(":")[0])

    # Input de parámetros de conexión a la base de datos
    with st.sidebar.expander("Configuración de la Base de Datos"):
        db_host = st.text_input("Host", value="localhost")
        db_name = st.text_input("Nombre de la Base de Datos", value="postgres")
        db_user = st.text_input("Usuario", value="postgres")
        db_password = st.text_input("Contraseña", type="password")

    # Mostrar estados en Opciones
    with st.sidebar.expander("Estados"):
        st.write("**Transcripción:**", st.session_state.transcription or "No disponible")
        st.write("**Consulta SQL:**")
        st.code(st.session_state.sql_query or "No disponible")

    # Cargar modelos
    st.sidebar.info("Cargando modelos...")
    whisper_model, whisper_processor = cargar_modelo_whisper()
    llama_tokenizer, llama_model = cargar_modelo_llama()
    st.sidebar.success("Modelos cargados")

    # Grabación de audio
    if st.button("Grabar Audio"):
        st.session_state.audio_data = grabar_audio(device_index=indice_dispositivo, duration=10)
        st.session_state.transcription = None
        st.session_state.sql_query = None
        st.session_state.df_result = None

    # Transcripción y generación de consulta SQL
    if st.session_state.audio_data is not None and st.session_state.transcription is None:
        transcription = transcribir_audio(st.session_state.audio_data, whisper_model, whisper_processor)
        if transcription:
            st.session_state.transcription = transcription
            st.write("**Texto Transcrito:**", transcription)

            # Generación de consulta SQL
            table_info = """
                  La tabla 'ventas' tiene la estructura:
                  id SERIAL PRIMARY KEY,
                  producto VARCHAR(100),
                  cantidad INT,
                  precio DECIMAL(10, 2).
                  Genera una consulta SQL postgres valida en base a esto: 
            """
            prompt = f"Eres un asistente que genera consultas SQL. \n {table_info}\n{st.session_state.transcription}"
            sql_query = generar_consulta_sql(prompt, llama_tokenizer, llama_model)
            if sql_query:
                st.session_state.sql_query = sql_query

    # Mostrar y editar la consulta SQL
    if st.session_state.sql_query:
        st.write("**Consulta SQL Generada:**")
        sql_query_input = st.text_area("Puedes editar la consulta SQL aquí:", value=st.session_state.sql_query)
        st.session_state.sql_query = sql_query_input

    # Ejecución de la consulta SQL
    if st.session_state.sql_query and st.button("Ejecutar Consulta SQL"):
        if not st.session_state.db_connection:
            try:
                conn = psycopg2.connect(
                    host=db_host,
                    database=db_name,
                    user=db_user,
                    password=db_password
                )
                st.session_state.db_connection = conn
                st.success("Conexión exitosa a la base de datos")
            except Exception as e:
                st.error(f"Error al conectar a la base de datos: {e}")

        if st.session_state.db_connection:
            try:
                with st.spinner('Ejecutando la consulta SQL...'):
                    df_result = pd.read_sql_query(st.session_state.sql_query, st.session_state.db_connection)
                    st.session_state.df_result = df_result
                st.success("Consulta ejecutada exitosamente")
                st.write("**Resultado de la Consulta:**")
                st.dataframe(df_result)

                # Descarga de resultados en CSV
                csv_data = df_result.to_csv(index=False)
                st.download_button(
                    label="Descargar CSV",
                    data=csv_data,
                    file_name="resultado_consulta.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error al ejecutar la consulta SQL: {e}")
                if st.session_state.df_result is not None:
                    st.write("**Último resultado válido:**")
                    st.dataframe(st.session_state.df_result)
                else:
                    st.error("No hay resultados anteriores para mostrar.")
        else:
            st.error("No hay conexión a la base de datos")

if __name__ == "__main__":
    main()
