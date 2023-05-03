
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import threading
from queue import Queue
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
import random
app = Flask(__name__)
@app.route('/')
def index():
    filename = 'test_model.html'
    cache_timeout = 0 if app.debug else 3600  # set cache timeout to 0 when in debug mode
    return send_file(filename, cache_timeout=cache_timeout, last_modified=True, add_etags=True, mimetype='text/html')
@app.route('/load-model', methods=['POST'])
def load_model():
    # check if the post request has the file part
    if 'model' not in request.files:
        return 'No model file uploaded'

    file = request.files['model']

    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return 'No selected model file'

    # save the model file to the uploads directory
    filename = secure_filename(file.filename)
    file.save('uploads/' + filename)

    # load the model file and perform testing in a separate thread
    model_file = 'uploads/' + filename
    t = threading.Thread(target=load, args=(model_file,))
    t.start()
    # wait for the thread to finish and get the result from the queue
    return "fileuploaded"

def load (model_file):
    #  Load the MNIST dataset
    global  x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    #  Swap axes to PyTorch's NCHW format
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    # Create and load the model
    state_dict = torch.load(model_file)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
            x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return nn.functional.log_softmax(x, dim=1)
    global classifier
    model = Net()
    model.load_state_dict(state_dict)
    #  Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )
   

@app.route('/test-origin', methods=['POST'])
def test_model():
    q = Queue()
    t = threading.Thread(target=test_orig, args=(classifier,x_test, y_test, q))
    t.start()
    # wait for the thread to finish and get the result from the queue
    accuracy = q.get()
    return {'accuracy': accuracy*100}

def  test_orig(classifier,x_test, y_test, q):
     # Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    q.put(accuracy)

    
if __name__ == '__main__':
    app.run(debug=True)
