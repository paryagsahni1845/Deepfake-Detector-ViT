# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Run app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
