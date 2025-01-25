from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__,template_folder="templates")

# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

import torch
import torch.nn as nn
import torch.optim as optim

class SurfaceDefectCNN(nn.Module):
    def __init__(self):
        super(SurfaceDefectCNN, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Classification output
        x = self.classifier(features)
        return x

# Load the trained model
model = SurfaceDefectCNN()
# model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("model85-acc.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET"])
def index():
    # Render the HTML page (index.html should be in the templates directory)
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
    image = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(image).item()  # Get the single sigmoid output value

    # Determine class based on threshold
    predicted = 1 if output > 0.5 else 0  # 1 = defect, 0 = non-defective
    class_names = ["Defect", "No defect"]
    return jsonify({"class_id": predicted, "class_name": class_names[predicted]})

if __name__ == "__main__":
#     # port = int(os.getenv("PORT", 8000))  # Default port is 8000
#     # app.run(host="0.0.0.0", port=port)
    app.run(debug=True)