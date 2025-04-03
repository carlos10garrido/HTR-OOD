FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install opencv and other dependencies for opencv
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optionally if you get an error with opencv
# 1. download the autofix tool
# pip install opencv-fixer==0.2.5
# 2. execute
# python -c "from opencv_fixer import AutoFix; AutoFix()"

RUN pip install opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

COPY . .