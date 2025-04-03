FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install opencv and other dependencies for opencv
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install dependencies
RUN pip install -r requirements.txt