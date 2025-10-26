<!DOCTYPE html>
<html lang="en">
<body>

<h1 align="center">🔢 MNIST Digit Classifier – Custom CNN Implementation</h1>

<p align="center">
  <b>Deep Learning | Computer Vision | PyTorch</b><br>
  Custom CNN architecture built <b>from scratch</b> using <b>PyTorch</b> for handwritten digit recognition with <b>98.32% accuracy</b> 🎯
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Accuracy-98.32%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<hr>

<h2>🚀 Overview</h2>

<p>
A custom Convolutional Neural Network (CNN) built from scratch using <b>PyTorch</b> to classify handwritten digits from the famous <b>MNIST dataset</b>. This project demonstrates fundamental deep learning concepts including custom architecture design, data augmentation, training optimization, and model evaluation.
</p>

<hr>

<h2>🎯 Key Features</h2>

<ul>
  <li>✨ <b>Custom CNN Architecture</b> - Designed from scratch without pre-trained models</li>
  <li>🎨 <b>Advanced Data Augmentation</b> - Random rotation, affine transformations, and elastic transforms</li>
  <li>📊 <b>Early Stopping Implementation</b> - Prevents overfitting with patience-based monitoring</li>
  <li>🎯 <b>High Accuracy</b> - Achieved 98.32% accuracy on test set</li>
  <li>💾 <b>Model Checkpointing</b> - Automatic saving of best performing models</li>
  <li>📈 <b>Training Visualization</b> - Loss curves and performance metrics tracking</li>
  <li>🔧 <b>GPU Acceleration</b> - CUDA support for faster training</li>
</ul>

<hr>

<h2>📊 Dataset</h2>

<ul>
  <li><b>Dataset:</b> MNIST Handwritten Digits</li>
  <li><b>Classes:</b> 10 (Digits 0-9)</li>
  <li><b>Training Samples:</b> 60,000 images</li>
  <li><b>Test Samples:</b> 10,000 images</li>
  <li><b>Image Size:</b> 28x28 pixels (Grayscale)</li>
  <li><b>Data Split:</b>
    <ul>
      <li>Training: ~55,000 images (91.67%)</li>
      <li>Validation: ~5,000 images (8.33%)</li>
      <li>Test: 10,000 images</li>
    </ul>
  </li>
</ul>

<hr>

<h2>🏗️ Model Architecture</h2>

<p>Custom CNN architecture designed specifically for MNIST digit classification:</p>

<pre><code>class customMNISTcnn(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

        # Regularization
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x
</code></pre>

<h3>Architecture Flow:</h3>
<ul>
  <li><b>Input:</b> 1 × 28 × 28 (Grayscale image)</li>
  <li><b>Conv1:</b> 1 → 32 channels, 3×3 kernel → 26 × 26 × 32</li>
  <li><b>ReLU + MaxPool:</b> 2×2 → 13 × 13 × 32</li>
  <li><b>Conv2:</b> 32 → 64 channels, 3×3 kernel → 11 × 11 × 64</li>
  <li><b>ReLU + MaxPool:</b> 2×2 → 5 × 5 × 64</li>
  <li><b>Flatten:</b> 5 × 5 × 64 = 1600 features</li>
  <li><b>FC1:</b> 1600 → 128 neurons</li>
  <li><b>ReLU + Dropout (0.2)</b></li>
  <li><b>FC2 (Output):</b> 128 → 10 classes</li>
</ul>

<hr>

<h2>🎨 Data Augmentation</h2>

<p>Robust augmentation pipeline to improve model generalization:</p>

<pre><code>transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ElasticTransform(alpha=50.0, sigma=5.0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
</code></pre>

<h3>Augmentation Techniques:</h3>
<ul>
  <li>📐 <b>RandomRotation(15):</b> Rotates images randomly by ±15 degrees</li>
  <li>🔄 <b>RandomAffine:</b> Translates images by up to 10% in x and y directions</li>
  <li>🌊 <b>ElasticTransform:</b> Applies elastic deformations (α=50.0, σ=5.0) to simulate natural handwriting variations</li>
  <li>📊 <b>Normalize:</b> Standardizes pixel values using MNIST dataset statistics (mean=0.1307, std=0.3081)</li>
</ul>

<hr>

<h2>⚙️ Training Configuration</h2>

<pre><code>model = customMNISTcnn()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
</code></pre>

<h3>Hyperparameters:</h3>
<ul>
  <li>🎯 <b>Loss Function:</b> CrossEntropyLoss()</li>
  <li>⚡ <b>Optimizer:</b> Adam with learning rate 0.001</li>
  <li>📦 <b>Batch Size:</b> 128</li>
  <li>🔄 <b>Max Epochs:</b> 100</li>
  <li>⏸️ <b>Early Stopping:</b> Patience = 5 epochs</li>
  <li>💾 <b>Checkpointing:</b> Saves best model automatically based on validation loss</li>
  <li>💻 <b>Device:</b> CUDA (GPU) if available, else CPU</li>
</ul>

<hr>

<h2>📈 Results</h2>

<table align="center">
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td><b>Test Accuracy</b></td>
    <td>98.32%</td>
  </tr>
  <tr>
    <td><b>Average Test Loss</b></td>
    <td>0.05182</td>
  </tr>
  <tr>
    <td><b>Training Stopped at</b></td>
    <td>Epoch 7 (Early Stopping)</td>
  </tr>
  <tr>
    <td><b>Total Parameters</b></td>
    <td>~133K</td>
  </tr>
  <tr>
    <td><b>Model Size</b></td>
    <td>Lightweight and efficient</td>
  </tr>
</table>

<hr>

<h2>🛠️ Technologies & Dependencies</h2>

<h3>Core Libraries:</h3>
<pre><code>import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

# Google Colab specific
from google.colab import drive
from google.colab import files
</code></pre>

<h3>Required Packages:</h3>
<p align="center">
  🔥 PyTorch • 🖼️ Torchvision • 📊 Matplotlib • 🧮 NumPy • 🧱 Pandas • 🖼️ Pillow • 📈 Scikit-learn
</p>

<hr>

<h2>📦 Installation & Setup</h2>

<h3>1. Install Dependencies</h3>
<pre><code>pip install torch torchvision
pip install numpy pandas matplotlib pillow scikit-learn
</code></pre>

<h3>2. For Google Colab</h3>
<pre><code># Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)
</code></pre>

<h3>3. Clone Repository</h3>
<pre><code>git clone https://github.com/prayashmohanty/mnist-digit-classifier.git
cd mnist-digit-classifier
</code></pre>

<hr>

<h2>🚀 Usage</h2>

<h3>1. Training the Model</h3>
<pre><code># Initialize model
model = customMNISTcnn()
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Train (with early stopping)
# See notebook for complete training loop
</code></pre>

<h3>2. Load Pre-trained Model</h3>
<pre><code># Load best saved model
model = customMNISTcnn()
model.load_state_dict(torch.load('path/to/best_model.pth'))
model.to(device)
model.eval()
</code></pre>

<h3>3. Make Predictions</h3>
<pre><code># Transform for inference
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load and predict
image = Image.open('digit.jpg')
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    predicted_digit = torch.argmax(output, dim=1).item()
    print(f"Predicted Digit: {predicted_digit}")
</code></pre>

<hr>

<h2>📁 Project Structure</h2>

<pre><code>MNIST-Digit-Classifier/
│
├── MNIST_digit.ipynb              # Complete implementation
├── README.md                       # Documentation
│
├── Saved_Models/
│   ├── Project05_MNIST_Bestmodel.pth      # Best model
│   └── Project05_MNIST_currentmodel.pth   # Latest checkpoint
│
├── test_images/                   # Sample images for testing
│   ├── zero.jpg
│   ├── one.jpg
│   ├── three.jpg
│   └── ...
│
└── requirements.txt               # Dependencies
</code></pre>

<hr>

<h2>🔑 Key Implementation Details</h2>

<h3>Device Configuration</h3>
<pre><code>device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)

# Move model and data to GPU
model.to(device)
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
</code></pre>

<h3>Early Stopping Mechanism</h3>
<pre><code>best_val_loss = float('inf')
patience = 5
wait = 0

for epoch in range(epochs):
    # Training and validation...
    
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        wait += 1
        if wait >= patience:
            print(f"⛔ Early stopping at epoch {epoch + 1}")
            break
</code></pre>

<h3>Custom Dataset Class</h3>
<pre><code>class customdataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X.iloc[index]
        label = self.Y.iloc[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
</code></pre>

<hr>

<h2>📊 Training Progress</h2>

<p>The model training includes:</p>

<ul>
  <li>📉 <b>Loss Tracking:</b> Both training and validation loss monitored per epoch</li>
  <li>📈 <b>Visualization:</b> Real-time plotting of loss curves</li>
  <li>✅ <b>Early Stopping:</b> Automatically stops at epoch 7 when validation loss stops improving</li>
  <li>💾 <b>Best Model Saved:</b> Model with lowest validation loss is preserved</li>
</ul>

<h3>Final Training Metrics:</h3>
<ul>
  <li><b>Epochs Trained:</b> 7 (out of max 100)</li>
  <li><b>Final Training Loss:</b> ~0.064</li>
  <li><b>Final Validation Loss:</b> ~0.059</li>
  <li><b>Test Loss:</b> 0.05182</li>
  <li><b>Test Accuracy:</b> 98.32%</li>
</ul>

<hr>

<h2>🎯 Model Performance</h2>

<h3>Strengths:</h3>
<ul>
  <li>✅ Excellent accuracy (98.32%) on unseen test data</li>
  <li>✅ Fast convergence with early stopping</li>
  <li>✅ Robust to various handwriting styles due to augmentation</li>
  <li>✅ Lightweight architecture (~133K parameters)</li>
  <li>✅ Low overfitting risk with dropout and early stopping</li>
  <li>✅ GPU-accelerated training</li>
</ul>

<h3>Technical Highlights:</h3>
<ul>
  <li>🎯 Low test loss (0.05182) indicates good generalization</li>
  <li>📊 Training stopped early (epoch 7) showing efficient learning</li>
  <li>🔄 Elastic transforms add realistic handwriting variations</li>
  <li>⚡ Adam optimizer provides adaptive learning rates</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
  <li>🚀 <b>Web Deployment:</b> Flask/Streamlit app with drawing canvas</li>
  <li>📱 <b>Mobile App:</b> Convert to ONNX/TFLite format</li>
  <li>🎨 <b>Advanced Augmentation:</b> CutOut, MixUp, CutMix techniques</li>
  <li>🏗️ <b>Architecture Experiments:</b> Batch normalization, residual connections</li>
  <li>📊 <b>Learning Rate Scheduling:</b> Cosine annealing or step decay</li>
  <li>🔄 <b>Ensemble Methods:</b> Multiple models for higher accuracy</li>
  <li>💡 <b>Explainability:</b> Grad-CAM visualization</li>
</ul>

<hr>

<h2>📝 Notebook Features</h2>

<p>The complete Jupyter notebook includes:</p>

<ol>
  <li>📥 <b>Data Loading:</b> MNIST dataset import from torchvision</li>
  <li>🔍 <b>Data Exploration:</b> Visualization and statistics</li>
  <li>✂️ <b>Data Splitting:</b> Train/validation split using sklearn</li>
  <li>🎨 <b>Augmentation Pipeline:</b> Comprehensive transforms</li>
  <li>🏗️ <b>Model Definition:</b> Custom CNN architecture</li>
  <li>🎓 <b>Training Loop:</b> With early stopping and checkpointing</li>
  <li>📊 <b>Evaluation:</b> Test set metrics and accuracy</li>
  <li>📈 <b>Visualization:</b> Loss curves over epochs</li>
  <li>💾 <b>Model Persistence:</b> Save and load functionality</li>
  <li>🧪 <b>Inference Testing:</b> Custom image prediction</li>
</ol>

<hr>

<h2>⚠️ Important Notes</h2>

<ul>
  <li>🔧 <b>GPU Highly Recommended:</b> CPU training will be significantly slower</li>
  <li>💾 <b>Google Drive:</b> Models are saved in Google Drive when using Colab</li>
  <li>📏 <b>Input Requirements:</b> Images must be 28×28 grayscale or converted</li>
  <li>🎯 <b>Single Digit:</b> Model designed for single digit per image</li>
  <li>🔄 <b>Preprocessing:</b> Custom images need same normalization as training data</li>
</ul>

<hr>

<h2>🐛 Troubleshooting</h2>

<h3>Common Issues:</h3>

<p><b>Issue:</b> CUDA out of memory</p>
<pre><code># Solution: Reduce batch size
batch_size = 64  # Instead of 128
</code></pre>

<p><b>Issue:</b> Poor prediction on custom images</p>
<pre><code># Solution: Ensure proper preprocessing
# 1. Convert to grayscale
# 2. Resize to 28x28
# 3. Apply same normalization (mean=0.1307, std=0.3081)
</code></pre>

<p><b>Issue:</b> Model not loading</p>
<pre><code># Solution: Check device consistency
model = customMNISTcnn()
model.load_state_dict(torch.load('model.pth', map_location=device))
</code></pre>

<hr>

<h2>👨‍💻 Author</h2>

<p align="center">
  <b>Prayash Ranjan Mohanty</b><br>
  B.Tech in Computer Science (AI & ML)<br>
  Kalinga Institute of Industrial Technology, Bhubaneswar<br>
  📧 <a href="mailto:prayashranjanmohanty11@gmail.com">prayashranjanmohanty11@gmail.com</a>
</p>

<p align="center">
  <a href="https://github.com/prayashmohanty">
    <img src="https://img.shields.io/badge/GitHub-PrayashRanjanMohanty-black?logo=github" alt="GitHub">
  </a>
</p>

<hr>

<h2>📄 License</h2>

<p align="center">
  This project is licensed under the <b>MIT License</b> - free for personal and academic use.
</p>

<hr>

<h2>🙏 Acknowledgments</h2>

<ul>
  <li>📚 <b>MNIST Dataset:</b> Yann LeCun, Corinna Cortes, and Christopher J.C. Burges</li>
  <li>🔥 <b>PyTorch Team:</b> For the exceptional deep learning framework</li>
  <li>🎓 <b>KIIT University:</b> For academic guidance and resources</li>
  <li>💡 <b>Open Source Community:</b> For inspiration and learning resources</li>
</ul>

<hr>

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b><br>
  <i>Made with ❤️ using PyTorch</i>
</p>

</body>
</html>
