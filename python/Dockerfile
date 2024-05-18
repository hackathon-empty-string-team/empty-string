FROM python:3.11.5

# Update apt-get
RUN apt-get update

# Upgrade pip
RUN pip install --upgrade pip 

# Make a new directory to put our code in
WORKDIR /python

COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt


CMD ["jupyter" , "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
