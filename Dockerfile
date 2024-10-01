# Use an Python runtime as a parent image
FROM python:3.11.1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first for dependency installation
COPY requirements2.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements2.txt

# Copy only the necessary files for the Flask app
COPY app.py .
COPY stock_price_model.h5 .

# Expose port 5000 (default port for Flask)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
