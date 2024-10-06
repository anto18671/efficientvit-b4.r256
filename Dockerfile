# Use a slim Python image for runtime
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set the working directory for the application
WORKDIR /app

# Copy the requirements.txt file from your host to the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Hugging Face Hub
RUN pip install --no-cache-dir huggingface-hub

# Copy the pre.py file from your host to the container
COPY pre.py .

# Set the Hugging Face token as an environment variable (this will be passed during docker run)
ENV HUGGINGFACE_TOKEN=""

# Set entry point for the container to execute the pre.py script
ENTRYPOINT ["sh", "-c", "huggingface-cli login --token $HUGGINGFACE_TOKEN && python pre.py"]
