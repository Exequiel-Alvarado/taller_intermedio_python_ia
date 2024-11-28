FROM postgres:latest

# Variables de entorno para la configuración inicial de PostgreSQL
ENV POSTGRES_USER=postgress
ENV POSTGRES_PASSWORD=postgres1
ENV POSTGRES_DB=postgress

# Exponer el puerto predeterminado de PostgreSQL
EXPOSE 5432

# Copiar un script SQL inicial para configurar tablas y datos iniciales
COPY init.sql /docker-entrypoint-initdb.d/