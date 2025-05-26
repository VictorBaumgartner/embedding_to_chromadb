FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install all dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose Streamlit port (optional)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "explore_chroma.py", "--server.port=8501", "--server.address=0.0.0.0"]
