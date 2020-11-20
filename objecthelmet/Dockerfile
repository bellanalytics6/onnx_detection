#FROM mcr.microsoft.com/azureml/onnxruntime:latest-openvino-myriad
FROM mcr.microsoft.com/azureml/onnxruntime:latest
#RUN apt-get clean
RUN pip install -U pip
RUN apt-get update
#RUN apt-get install -y gcc
#RUN pip install --no-cache-dir numpy~=1.17.5 tensorflow~=2.0.2 flask~=1.1.2 pillow~=7.2.0
# RUN pip install --no-cache-dir numpy~=1.17.5 flask~=1.1.2 pillow~=7.2.0
#RUN pip install uwsgi

RUN apt-get install -y cmake protobuf-compiler libprotoc-dev

RUN pip install pillow onnx flask~=1.1.2

COPY app /app

# By default, we run manual image resizing to maintain parity with CVS webservice prediction results.
# If parity is not required, you can enable faster image resizing by uncommenting the following lines.
# RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-bin
# RUN pip install opencv-python-headless

# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD python -u app.py
#CMD uwsgi --http 0.0.0.0:80 --wsgi-file app.py --callable app --processes 4 --threads 2 --stats 0.0.0.0:9191