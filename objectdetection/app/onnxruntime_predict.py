# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import os
import sys
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
from datetime import datetime
import tempfile

gloves_od_model = None
helmet_od_model = None

HELMET_MODEL_FILENAME = 'helmet_model.onnx'
HELMET_LABELS_FILENAME = 'helmet_labels.txt'
GLOVES_MODEL_FILENAME = 'gloves_model.onnx'
GLOVES_LABELS_FILENAME = 'gloves_labels.txt'


class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            #temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            temp = os.path.join(dirpath, os.path.basename(HELMET_MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def initialize():
    # Load labels
    with open(HELMET_LABELS_FILENAME, 'r') as f:
        helmet_labels = [l.strip() for l in f.readlines()]

    global helmet_od_model
    helmet_od_model = ONNXRuntimeObjectDetection(HELMET_MODEL_FILENAME, helmet_labels)

    with open(GLOVES_LABELS_FILENAME, 'r') as f:
        gloves_labels = [l.strip() for l in f.readlines()]

    global gloves_od_model
    gloves_od_model = ONNXRuntimeObjectDetection(GLOVES_MODEL_FILENAME, gloves_labels)

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def gloves_predict_image(image):
    log_msg('Gloves Predicting image')

    w, h = image.size
    log_msg("Gloves Image size: {}x{}".format(w, h))

    predictions = gloves_od_model.predict_image(image)

    response = {
        'id': '',
        'project': '',
        'iteration': '',
        'created': datetime.utcnow().isoformat(),
        'predictions': predictions }
        
    log_msg('Gloves Results: ' + str(response))
    return response

def helmet_predict_image(image):
    log_msg('Helmet Predicting image')

    w, h = image.size
    log_msg("Helmet Image size: {}x{}".format(w, h))

    predictions = helmet_od_model.predict_image(image)

    response = {
        'id': '',
        'project': '',
        'iteration': '',
        'created': datetime.utcnow().isoformat(),
        'predictions': predictions }
        
    log_msg('Helmet Results: ' + str(response))
    return response    