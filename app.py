from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super(MultiScaleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)  # Single neuron for binary classification
        self.dropout = nn.Dropout(0.5)
        self.fc3=nn.Linear(512,256)
        self.dropout = nn.Dropout(0.2)
        self.fc4=nn.Linear(256,1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))

        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)  # No softmax or sigmoid here
        x=self.fc3(x)
        x=self.fc4(x)

        return x


# Initialize the model
model = MultiScaleCNN()

model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Ensure RGB format

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(image).item()  # Get the single sigmoid output value

    # Determine class based on threshold
    predicted = 1 if output > 0.5 else 0  # 1 = defect, 0 = non-defective
    class_names = ["No Defect", "Defect"]
    return jsonify({"class_id": predicted, "class_name": class_names[predicted]})

if __name__ == "__main__":
    app.run(debug=True)
