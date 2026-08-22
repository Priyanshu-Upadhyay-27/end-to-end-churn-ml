# we define the OS or the python version in which we work and slim is the
# documentation free, less space taking python x.xx image.
FROM python:3.10-slim

# We define the work directory in which we work
WORKDIR /app

# Some files are copied into the container.
# Commands written after RUN are the commands which are run when the image creation is in progress,
# when the image build is completed, the things below are present as a sub layer in the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Coped significant files
COPY app.py .
COPY production_pipeline.pkl .

COPY data/raw/stream_data.csv ./data/raw/stream_data.csv
COPY data/raw/train_data.csv ./data/raw/train_data.csv

# We need to expose a backend port, at which virtual ethernet cable is connected,
# and form a veth pair with another container
EXPOSE 8501

# This also executes a command, but when the images spins up and make a live container, not during image creation.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]