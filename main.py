import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32, ViT_B_32_Weights
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import the ViT-b model
model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).to(DEVICE)
# Download the CIFAR10 dataset
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Set up the dataloader
dataloader = DataLoader(cifar10, batch_size=BATCH_SIZE, shuffle=True)

# Create a tensor to store the CKA values
cka = torch.zeros((len(model.encoder.layers), len(model.encoder.layers))).to(DEVICE)\
# Forward each batch through the model
with torch.no_grad():
    HSIC_kl = torch.zeros(len(model.encoder.layers), len(model.encoder.layers)).to(DEVICE)
    HSIC_kk = torch.zeros(len(model.encoder.layers)).to(DEVICE)
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        # Extract the feature representations at each layer
        x = model._process_input(inputs.to(DEVICE))
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Extract the feature representations at each layer
        features = []
        H = torch.eye(n).to(DEVICE) - 1 / n
        input = x
        for layer in model.encoder.layers:
            input = layer(input)
            features.append(input)

        for j in range(len(features)):
            K_p = H @ torch.einsum("ikl, jmn->ij", features[j], features[j]) @ H
            for k in range(len(features)):
                L_p = H @ torch.einsum("ikl, jmn->ij",features[k], features[k]) @ H
                HSIC_kl[j, k] += torch.sum(K_p * L_p) / (n - 1) ** 2
            HSIC_kk[j] += torch.sum(K_p * K_p) / (n - 1) ** 2

cka = HSIC_kl/torch.kron(HSIC_kk.reshape(1, -1), HSIC_kk.reshape(-1, 1))**0.5
sns.heatmap(cka.cpu().numpy(), annot=False)
plt.gca().invert_yaxis()
plt.show()