# 1. Use an official lightweight Python base image
FROM python:3.11-slim

# 2. Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system-level dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy over the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the remaining source code and configurations into the container
COPY src/ ./src/
COPY app.py .

# 8. Expose the port that Streamlit uses by default
EXPOSE 8501

# 9. Define health check to ensure container is running successfully
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 10. Command to run the Streamlit application on container startup
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
