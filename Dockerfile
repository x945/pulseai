# Step 1: Use a lightweight base image
FROM python:3.12-slim

# Step 2: Install system dependencies in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    gettext \
    pkg-config \
    libxml2-dev \
    libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Step 3: Set a working directory inside the container
WORKDIR /app

# Step 4: Copy application files first (excluding large unnecessary files)
COPY . /app/

# Step 5: Create and activate a virtual environment
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt \
    && rm -rf ~/.cache/pip  # Clean up pip cache to reduce image size

# Step 6: Expose a port if the app is a web server
EXPOSE 8000

# Step 7: Define the entry point
CMD ["/opt/venv/bin/python", "app.py"]
