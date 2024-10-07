# %%
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import pil_loader
from torchvision.models import resnet50, ResNet50_Weights

import torch
import numpy as np
import matplotlib.pyplot as plt

# %%
# Data preprocessing
IMN_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
dataset = ImageNet(root='/om/user/yu_xie/data/ImageNet', split='val', transform=IMN_transform)

# %%
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# %%
def show_image_predictions(image, label):
    # Convert the image tensor to a numpy array and transpose it to (H, W, C) format
    img_np = image.numpy().transpose(1, 2, 0)

    # Denormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denorm = std * img_np + mean

    # Clip values to be between 0 and 1
    img_denorm = np.clip(img_denorm, 0, 1)

    # True label
    if label is not None:
        true_label = dataset.classes[label]
    else:
        true_label = 'None'
    # Display the image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_denorm)
    ax.axis('off')
    ax.set_title(f"True Label: {true_label}")

    # Get model predictions
    with torch.inference_mode():
        output = model(image.unsqueeze(0))
    
    # Get top 10 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top10_prob, top10_catid = torch.topk(probabilities, 10)
    classes = [dataset.classes[i][0] for i in top10_catid]
    probabilities = [p.item() for p in top10_prob]

    # Plot top 10 predictions
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(10), probabilities, alpha=0.6)
    ax.set_xticks(range(10), classes, rotation=30, ha='right')
    ax.set_xlabel('Class labels')
    ax.set_ylabel('Soft-maxed activations')
    ax.set_title('Top 10 predictions from ResNet50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('./top10_predictions.pdf', bbox_inches='tight', transparent=True)


# %%
# Get the image and label
img_idx = np.random.randint(0, len(dataset))
image, label = dataset[img_idx]
show_image_predictions(image, label)

# %%
image = IMN_transform(pil_loader('./dog_kart.jpg'))
show_image_predictions(image, None)

# %%
image = IMN_transform(pil_loader('./dog_image.jpg'))
show_image_predictions(image, None)

# %%



