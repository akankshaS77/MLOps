# Use a lightweight Python 3.9 image
FROM python:3.9.21

# Set the working directory in the container
WORKDIR /App

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application into the container
COPY . .

# Ensure that the entry point is the correct application file
CMD ["python", "App.py"]
