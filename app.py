from flask import Flask, request, render_template
import torch
import torch.nn as nn
import numpy as np

# Define same model class
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Load model
model = NeuralNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    inputs = torch.FloatTensor([data])
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    return render_template("index.html", prediction_text=f"Predicted Class: {predicted.item()}")

if __name__ == "__main__":
    app.run(debug=True)
