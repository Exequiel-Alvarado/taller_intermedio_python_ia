# README - Automatización de Consultas SQL con Voz, Whisper y Llama

Este proyecto utiliza **Streamlit**, **Groq API**, y **PostgreSQL** para automatizar la generación y ejecución de consultas SQL basadas en transcripciones de voz. El flujo incluye grabar audio, transcribirlo, generar consultas SQL con un modelo de lenguaje grande (LLM) y ejecutarlas en una base de datos.

## Instalación de dependencias

Asegúrate de tener **Python 3.8 o superior** instalado.

### 1. Clonar el repositorio
```bash
$ git clone <url-del-repositorio>
$ cd <nombre-del-directorio>
```

### 2. Crear un entorno virtual
```bash
$ python -m venv .venv
$ source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar las dependencias necesarias
```bash
$ pip install streamlit sounddevice scipy psycopg2-binary pandas requests
```

## Configuración

### 1. API Key de Groq
Este proyecto requiere una API Key válida de Groq para transcribir audio y generar consultas SQL.

- Ingresa tu API Key al ejecutar la aplicación en el campo de configuración de la barra lateral.

### 2. Configuración de la base de datos
Proporciona los siguientes detalles para conectarte a tu base de datos PostgreSQL en la barra lateral:

- **Host**: Dirección de tu servidor de base de datos (por defecto: `localhost`)
- **Nombre de la base de datos**: Nombre de tu base de datos (por defecto: `postgres`)
- **Usuario**: Usuario de la base de datos (por defecto: `postgres`)
- **Contraseña**: Contraseña de la base de datos

## Ejecución

### 1. Inicia la aplicación
Ejecuta el siguiente comando para iniciar la aplicación:
```bash
$ streamlit run app.py
```

### 2. Pasos en la interfaz
1. **Configura la API Key**: Ingresa tu API Key de Groq en el panel lateral izquierdo.
2. **Configura la base de datos**: Completa los detalles de tu conexión a PostgreSQL.
3. **Graba audio**:
    - Haz clic en el botón "Grabar Audio" para capturar tu voz.
    - El audio se guardará en formato WAV y se transcribirá usando la API de Groq.
4. **Genera la consulta SQL**:
    - Basado en la transcripción, se generará automáticamente una consulta SQL.
    - Revisa y edita la consulta generada si es necesario.
5. **Ejecuta la consulta**:
    - Si la consulta generada es válida, se ejecutará en la base de datos configurada.
    - Los resultados se mostrarán en una tabla interactiva.
6. **Exporta los resultados**:
    - Descarga los resultados en formato CSV con el botón correspondiente.

## Instalación de PyCharm Community y creación de entornos virtuales

### 1. Descarga e instalación de PyCharm Community
- Descarga PyCharm Community desde su página oficial: [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)
- Sigue las instrucciones del instalador para tu sistema operativo.

### 2. Configura el proyecto en PyCharm
1. Abre PyCharm y selecciona **"Open"** para importar el repositorio clonado.
2. Configura el intérprete del proyecto:
   - Ve a **File > Settings > Project > Python Interpreter**.
   - Haz clic en **"Add Interpreter"** y selecciona **Virtualenv Environment**.
   - Escoge la ubicación de tu entorno virtual `.venv` creado anteriormente.
3. PyCharm detectará automáticamente las dependencias del proyecto.

### 3. Instala dependencias adicionales
Abre el terminal integrado en PyCharm y ejecuta:
```bash
$ pip install -r requirements.txt  # Si el archivo requirements.txt está disponible
```

## Estructura del proyecto

```
├── app.py            # Código principal de la aplicación
├── audio.wav         # Archivo de audio grabado (temporal)
├── README.md         # Documentación del proyecto
└── requirements.txt  # Lista de dependencias (opcional)
```

## Notas importantes

1. **Formatos de audio compatibles**:
   - El archivo grabado se guarda en formato WAV, compatible con la API de Groq.

2. **Errores comunes**:
   - Si recibes un error relacionado con el formato `Content-Type`, asegúrate de que el archivo de audio se envía como `multipart/form-data`.
   - Si la consulta SQL generada no es válida, edítala antes de ejecutarla.

3. **Limitaciones**:
   - La API de Groq puede tener restricciones de tokens o latencia en solicitudes concurrentes.

## Recursos

- **Documentación de Groq**: [https://console.groq.com/docs/](https://console.groq.com/docs/)
- **Documentación de Streamlit**: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **Documentación de PostgreSQL**: [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/)
- **Guía de PyCharm**: [https://www.jetbrains.com/help/pycharm/](https://www.jetbrains.com/help/pycharm/)

## Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
