
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import torch
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import threading
from queue import Queue

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('test_model.html')

@app.route('/test-model', methods=['POST'])
def test_model():
    q = Queue()
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
    t = threading.Thread(target=test, args=(model_file,q))
    t.start()
    # wait for the thread to finish and get the result from the queue
    accuracy = q.get()
    return {'accuracy': accuracy*100}

def test(model_file,q):
    # load the test set
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = datasets.MNIST('data', train=False, transform=transform, download=True)

    # create a data loader for the test set
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # load the model
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

    model= Net()
    model.load_state_dict(torch.load(model_file))

    # set the model to evaluation mode
    model.eval()

    # define a variable to keep track of the number of correct predictions
    correct = 0

    # define a variable to keep track of the total number of predictions
    total = 0

    # loop over the test set
    for images, labels in testloader:
        # make predictions on the batch of images
        outputs = model(images)
        # get the predicted class for each image
        _, predicted = torch.max(outputs.data, 1)
        # update the correct predictions count
        correct += (predicted == labels).sum().item()
        # update the total predictions count
        total += labels.size(0)

    # calculate the accuracy
    accuracy = correct / total

    # print the accuracy
    print(f'Accuracy: {accuracy}')
    q.put(accuracy)

if __name__ == '__main__':
    app.run(debug=True)
