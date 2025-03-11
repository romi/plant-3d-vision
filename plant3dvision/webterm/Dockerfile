FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Docker CLI
RUN apt-get update && \
    apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli

# Install Python dependencies
COPY requirements.webapp.txt .
RUN pip install --no-cache-dir -r requirements.webapp.txt

# Copy application files
COPY app.py .
COPY static ./static
COPY templates ./templates

# Create uploads directory
RUN mkdir uploads

# Run the application
CMD ["python", "app.py"]