# Step 1: Use a base image
# The first step is to define the base image. In this case, we are using a Python image.
FROM python:3.12-slim

# Install gettext
RUN apt-get update && apt-get install -y gettext  

# Step 2: Set a working directory inside the container
# This is where your app will live inside the container.
WORKDIR /app

# Step 3: Copy your application files into the Docker image
# We will copy all files from the local project to the Docker container.
COPY . /app/

# Step 4: Install any dependencies specified in requirements.txt
# We use pip to install the requirements inside the virtual environment.
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install -r requirements.txt

# Step 5: Expose a port (optional)
# If your app is a web server or listens on a port, expose it.
EXPOSE 8000

# Step 6: Define the entry point for your application
# The command that will run when the container starts.
CMD ["/opt/venv/bin/python", "app.py"]
