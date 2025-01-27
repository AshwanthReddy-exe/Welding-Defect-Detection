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

        # First shared convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Common initial layer
        self.bn1 = nn.BatchNorm2d(16)

        # Multi-scale feature extraction
        self.conv2_small = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Small receptive field
        self.bn2_small = nn.BatchNorm2d(32)
        
        self.conv2_medium = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # Medium receptive field
        self.bn2_medium = nn.BatchNorm2d(32)
        
        self.conv2_large = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)  # Large receptive field
        self.bn2_large = nn.BatchNorm2d(32)

        # Combine multi-scale features and process further
        self.conv3 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)  # Combine features from all scales
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Adjusted based on tensor size after pooling
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 1)  # Single output for binary classification

    def forward(self, x):
        # Shared initial feature extraction with BatchNorm and LeakyReLU
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Multi-scale feature extraction with BatchNorm and LeakyReLU
        small_features = F.leaky_relu(self.bn2_small(self.conv2_small(x)))  # Small receptive field
        medium_features = F.leaky_relu(self.bn2_medium(self.conv2_medium(x)))  # Medium receptive field
        large_features = F.leaky_relu(self.bn2_large(self.conv2_large(x)))  # Large receptive field

        # Concatenate features from all scales
        x = torch.cat((small_features, medium_features, large_features), dim=1)

        # Further processing with convolutional layer and BatchNorm
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)  # No activation yet (logits)

        return x


# Initialize the model
model = MultiScaleCNN()

model.load_state_dict(torch.load("best_model_new.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
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
