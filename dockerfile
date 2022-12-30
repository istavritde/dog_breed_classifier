FROM python:3.6-slim


# Add a /app volume
VOLUME ["/app"]

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
#ADD . /app


# Docker update package manager
RUN apt-get update
RUN apt-get -y update && apt-get install -y \
  curl \
  less \
  sudo \
  unzip \
  python3-dev \
  python-dev \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  build-essential #&& rm -rf /var/lib/apt/lists/*
# installing curl, screen and mssql odbc 
RUN apt-get install screen -y
RUN apt-get update

# installing necessary python packages declared in requirements.txt
RUN pip install --upgrade pip
#RUN pip install -r /.requirements/requirements.txt
RUN pip install opencv-python==3.1.0.5
RUN pip install h5py==2.6.0
RUN pip install matplotlib==2.0.0
RUN pip install numpy==1.12.0
RUN pip install scipy==0.18.1
RUN pip install tqdm==4.11.2
RUN pip install keras==2.0.2
RUN pip install scikit-learn==0.18.1
RUN pip install pillow==4.0.0
RUN pip install ipykernel==4.6.1
RUN pip install tensorflow==1.0.0
RUN pip install flask==2.0.3
RUN pip install itsdangerous==2.0
RUN pip install Jinja2==3.0
RUN pip install MarkupSafe==2.0
RUN pip install six==1.15.0
RUN pip install urllib3==1.25.10
RUN pip install Werkzeug==2.0
RUN pip install pandas==0.23.0
RUN pip install click==8.0
RUN pip install h5py==2.8.0

# unblock port 3000 for the Flask app to run on
EXPOSE 3000
# run tests

# execute the Flask app

CMD ["python", "run.py"]
