import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Remove the last fully connected layer for embedding purpose
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Move the model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization parameters for ImageNet
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    img_tensor = transform(img)
    return img_tensor

def embed_terrain_patch(patch_path):
    image_tensor = load_image(patch_path)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()  # Move the tensor to GPU

    with torch.no_grad():
        embedding = model(image_tensor)
        embedding = embedding.view(embedding.size(0), -1)  # Flatten the embedding

    return embedding.cpu().numpy()  # Move the embedding back to CPU and convert to numpy array

# Example usage
embedding = embed_terrain_patch("../../dataset/processed_image/unity_lejeune_emout_splitted_5/grid_0_0.png")
print(embedding)
