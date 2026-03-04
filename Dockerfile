# Use a slim Python base
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    PORT=7860 \
    # Keep headless = false as you requested; server will start a virtual display if needed
    ZD_HEADLESS=false \
    NO_INITIAL_FETCH=0 \
    # Default chromium path (used by some tools). Update if needed.
    CHROME_PATH=/usr/bin/chromium

# Install runtime deps including Xvfb and Chromium (and fonts / libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        gnupg \
        xvfb \
        chromium \
        chromium-driver \
        libxrender1 libxext6 libxi6 libxtst6 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
        libgtk-3-0 libnss3 libasound2 libatk1.0-0 libatk-bridge2.0-0 \
        fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy requirements and server file
COPY requirements.txt ./requirements.txt
COPY server.py ./server.py

# Install Python deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Expose port used by uvicorn
EXPOSE ${PORT}

# Default command to run the FastAPI app via uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--loop", "asyncio", "--workers", "1"]
