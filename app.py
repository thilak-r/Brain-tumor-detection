from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Define the Brain Tumor Model class
class BrainTumorModel(nn.Module):
    def __init__(self):
        super(BrainTumorModel, self).__init__()
        # Load the pretrained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer to match the number of classes in your problem (e.g., 4 classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 4)

    def forward(self, x):
        return self.model(x)

# Load the pre-trained model and map it to CPU if necessary
model = BrainTumorModel()
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the function for image prediction
def test_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
    input_tensor = preprocess(image)  # Preprocess the image
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Ensure the model is on the correct device (CPU in this case)
    device = torch.device("cpu")
    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():  # Disable gradient calculation
        output = model(input_batch)
    
    _, predicted = torch.max(output, 1)  # Get the predicted class
    return predicted.item()  # Return the predicted class as an integer

# Define the routes for the Flask app
@app.route('/')
def index():
    return render_template('index.html')  # Render the homepage

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})  # Return error if no file is uploaded

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})  # Return error if no file is selected
    
    # Save the uploaded file to the uploads directory
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    # Make prediction using the test_image function
    predicted_class = test_image(filepath)
    return jsonify({'predicted_class': predicted_class, 'image_path': filepath})  # Return prediction

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
