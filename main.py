import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_l_16, ViT_L_16_Weights
import numpy as np

BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import the ViT-b model
model = vit_l_16(weights=ViT_L_16_Weights).to(DEVICE)
# Download the CIFAR10 dataset
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Set up the dataloader
dataloader = DataLoader(cifar10, batch_size=BATCH_SIZE, shuffle=True)

# Create a tensor to store the CKA values
cka = torch.zeros((len(model.encoder.layers), len(model.encoder.layers))).to(DEVICE)

# Forward each batch through the model
with torch.no_grad():
  for i, (inputs, labels) in enumerate(tqdm(dataloader)):
      # Extract the feature representations at each layer
      x = model._process_input(inputs.to(DEVICE))
      n = x.shape[0]

      # Expand the class token to the full batch
      batch_class_token = model.class_token.expand(n, -1, -1)
      x = torch.cat([batch_class_token, x], dim=1)
      # Extract the feature representations at each layer
      features = model.encoder(x)

      for j in range(len(model.encoder.layers)):
          for k in range(len(model.encoder.layers)):
              cka[j, k] += (torch.mean(features[j] * features[k]) - torch.mean(features[j]) * torch.mean(features[k]))

cka /= len(dataloader)