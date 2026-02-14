FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/dashboard/ ./src/dashboard/
COPY data/ ./data/
COPY docs/qualitative/ ./docs/qualitative/
COPY script/seed_config_db.py ./script/seed_config_db.py
COPY .streamlit/ ./.streamlit/

# Seed the configuration database
RUN python script/seed_config_db.py

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/dashboard/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
