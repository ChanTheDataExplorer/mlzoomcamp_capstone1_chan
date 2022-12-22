import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask import redirect
from flask import url_for

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))

classes = [
    'cup',
    'fork',
    'glass',
    'knife',
    'plate',
    'spoon',
]

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'kitchenware-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_35'].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_26'].float_val
    return dict(zip(classes, preds))

def predict_endpoint(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)

    # For getting the label only
    pred = list(pb_response.outputs['dense_26'].float_val)

    label_map = {'cup': 0, 'fork': 1, 'glass': 2, 'knife': 3, 'plate': 4, 'spoon': 5}
    label_lookup = {y: x for x,y in label_map.items()}

    max_label = max(pred)
    loc = pred.index(max_label)

    pred_label = label_lookup[loc]

    return response, pred_label.title()

app = Flask('gateway_webapp')

@app.get("/") #, methods=['POST'])
def main():
    return render_template("base.html", url = request.args.get('url'), pred_label= request.args.get('pred_label'), pred_list = request.args.get('pred_list'))

@app.post("/predict")
def predict():
    url = request.form.get("url")
    try:
        pred, pred_label = predict_endpoint(url)
        pred = str(pred)
    except:
        url = None
        pred_label = None
        pred = None
        
    return redirect(url_for("main", url = url, pred_label = pred_label, pred_list = pred))


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=6969)

'''
pred = list(pb_response.outputs['dense_26'].float_val)

label_map = {'cup': 0, 'fork': 1, 'glass': 2, 'knife': 3, 'plate': 4, 'spoon': 5}
label_lookup = {y: x for x,y in label_map.items()}

max_label = max(pred)
loc = pred.index(max_label)

pred_label = label_lookup[loc]
'''