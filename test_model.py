
from flask import Flask, request, send_file,jsonify, Response
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
import shap
import matplotlib.pyplot as plt
import io
import base64

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
    # class Net(nn.Module):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #         self.fc1 = nn.Linear(320, 50)
    #         self.fc2 = nn.Linear(50, 10)

    #     def forward(self, x):
    #         x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
    #         x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
    #         x = x.view(-1, 320)
    #         x = nn.functional.relu(self.fc1(x))
    #         x = self.fc2(x)
    #         return nn.functional.log_softmax(x, dim=1)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(50, 10),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x
    global classifier
    global model
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

@app.route('/test-adversarial-fgm', methods=['POST'])
def test_adv_fgm_main():
    q = Queue()
    t = threading.Thread(target=test_adv_fgm, args=(classifier,x_test, y_test, q))
    t.start()
    # wait for the thread to finish and get the result from the queue
    accuracy = q.get()
    return {'accuracy': accuracy*100}

def  test_adv_fgm(classifier,x_test, y_test, q):
     # Evaluate the ART classifier on benign test examples
    attack_FGD = FastGradientMethod(estimator=classifier, eps=0.2)
    global x_test_adv_FGD
    x_test_adv_FGD = attack_FGD.generate(x=x_test)
    predictions = classifier.predict(x_test_adv_FGD)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples FGD: {}%".format(accuracy * 100))
    q.put(accuracy)


@app.route('/explain-shap', methods=['GET'])
def get_image():
    q = Queue()
    t = threading.Thread(target=get_shap_explanations, args=(x_test,model, q))
    t.start()

    # wait for the thread to finish and get the result from the queue
    shap_numpy, test_numpy = q.get()

    # Use shap to plot the explanation and save the figure to a BytesIO object
    fig = shap.image_plot(shap_numpy, test_numpy, show=False)
    output = io.BytesIO()
    plt.savefig(output, format='PNG')
    plt.close(fig)  # Close the figure to free up memory

    # Convert image data to base64
    image_base64 = base64.b64encode(output.getvalue()).decode()

    # Return as JSON
    return jsonify({'image': image_base64})

def get_shap_explanations(x_test, model, q):
    x_testtensor = torch.from_numpy(x_test)
    background = x_testtensor[:100]
    test_images = x_testtensor[:5]
    pred_list=[]
    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted)
    print(pred_list)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
    q.put((shap_numpy, test_numpy))

@app.route('/explain-shap-adv', methods=['GET'])
def get_image2():
    q = Queue()
    t = threading.Thread(target=get_shap_explanations_fgd, args=(x_test_adv_FGD,model, q))
    t.start()

    # wait for the thread to finish and get the result from the queue
    shap_numpy, test_numpy = q.get()

    # Use shap to plot the explanation and save the figure to a BytesIO object
    fig = shap.image_plot(shap_numpy, test_numpy, show=False)
    output = io.BytesIO()
    plt.savefig(output, format='PNG')
    plt.close(fig)  # Close the figure to free up memory

    # Convert image data to base64
    image_base64 = base64.b64encode(output.getvalue()).decode()

    # Return as JSON
    return jsonify({'image': image_base64})

def get_shap_explanations_fgd(x_test_adv_FGD, model, q):
    x_testtensor = torch.from_numpy(x_test_adv_FGD)
    background = x_testtensor[:100]
    test_images = x_testtensor[:5]
    pred_list=[]
    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted)
    print(pred_list)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
    q.put((shap_numpy, test_numpy))


   


   
if __name__ == '__main__':
    app.run(debug=True)
