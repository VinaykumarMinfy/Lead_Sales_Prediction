# Dockerfile

# Use an official Airflow image
FROM apache/airflow:2.9.3

# Switch back to the airflow user
USER airflow

# Copy the requirements file and install packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Create directories inside the container for organization
# Airflow will read dags from /opt/airflow/dags
# We will place other files in their own folders for clarity
RUN mkdir -p /opt/airflow/scripts /opt/airflow/data

# Copy your DAG file
COPY dags/ /opt/airflow/dags/

# Copy your main script to the scripts folder inside the container
COPY main.py /opt/airflow/scripts/

# Copy the data file to the data folder inside the container
COPY lead_scoring.csv /opt/airflow/data/