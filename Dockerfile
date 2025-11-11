# ========================
# Base image
# ========================
FROM python:3.9-slim

# ========================
# Set working directory
# ========================
WORKDIR /app

# ========================
# Copy project files
# ========================
COPY . /app

# ========================
# Install dependencies
# ========================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ========================
# Streamlit environment variables
# ========================
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PYTHONUNBUFFERED=1

# ========================
# Expose port for Streamlit
# ========================
EXPOSE 8501

# ========================
# Run the Streamlit app
# ========================
CMD ["streamlit", "run", "app.py"]
