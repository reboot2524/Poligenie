FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variable to allow Streamlit to run in Docker
ENV PYTHONUNBUFFERED=1

# Expose the port Cloud Run will listen to
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "poliapp.py", "--server.port=8080", "--server.address=0.0.0.0"]
