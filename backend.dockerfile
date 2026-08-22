# we define the OS or the python version in which we work and slim is the
# documentation free, less space taking python x.xx image.
FROM python:3.10-slim

# We define the work directory in which we work
WORKDIR /app

# These are some command which are run when the image creation is in progress,
# when the image build is completed, the things below are present as a sub layer in the image.

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# These things are copied in the docker file and below run command install those dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only required file in the backend container image.
COPY api.py .
COPY database.py .
COPY production_pipeline.pkl .

# We need to expose a backend port, at which virtual ethernet cable is connected,
# and form a veth pair with another container
EXPOSE 8000
# This also executes a command, but when the images spins up and make a live container.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]