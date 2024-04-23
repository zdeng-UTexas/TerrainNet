import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np

class CustomResNet50(nn.Module):
    def __init__(self, embedding_size=128):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet50 model with new syntax
        weights = ResNet50_Weights.DEFAULT  # This loads the default weights, similar to pretrained=True previously
        original_model = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Remove the last layer
        self.embed = nn.Linear(original_model.fc.in_features, embedding_size)  # New embedding layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.embed(x)  # Embedding layer
        return x

def load_image_paths(txt_file):
    with open(txt_file, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
    return paths

def generate_embeddings(image_paths, model, transform, output_csv="output.csv"):
    model = nn.DataParallel(model)  # Enable DataParallel to use multiple GPUs
    with open(output_csv, "w") as f:
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            if torch.cuda.is_available():
                image = image.cuda()
            with torch.no_grad():
                embedding = model(image).cpu().numpy()
            
            # Flatten the embedding and convert it to a string of comma-separated values
            embedding_str = ",".join(map(str, embedding.flatten()))
            f.write(f"{path},{embedding_str}\n")  # Write the path and embedding to the CSV

# Example usage
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = load_image_paths("../../experiment/20240401_lejeune_emount_global_cost_map/path_to_patch_5_demo.txt")  # Load paths from your .txt file
model = CustomResNet50(embedding_size=128)  # Initialize the model with 128-dimensional embeddings
if torch.cuda.is_available():
    model = model.cuda()
model.eval()  # Set the model to evaluation mode

# Specify the output CSV file location
output_csv_path = "../../experiment/embeddings_5_demo.csv"
generate_embeddings(image_paths, model, transform, output_csv=output_csv_path)  # Generate embeddings and save to CSV
