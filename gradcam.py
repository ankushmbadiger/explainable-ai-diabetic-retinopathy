import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms


MODEL_PATH = "models/efficientnet_b0_aptos_ddr_best.pth"

CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}


# --------------------------------------------------
# Model
# --------------------------------------------------

model = efficientnet_b0(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Last convolutional feature layer
target_layer = model.features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


def generate_gradcam(image_path):

    global activations, gradients

    # Load original image
    original_image = Image.open(image_path).convert("RGB")

    # Resize image for display
    display_image = original_image.resize((224, 224))

    # Preprocess
    image_tensor = transform(original_image)
    image_tensor = image_tensor.unsqueeze(0)

    # Forward pass
    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(
        probabilities, dim=1
    )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    # Clear previous gradients
    model.zero_grad()

    # Backpropagate predicted class
    score = output[0, predicted_class]
    score.backward()

    # --------------------------------------------------
    # Calculate Grad-CAM
    # --------------------------------------------------

    # gradients:
    # [1, channels, height, width]

    # Average gradient over spatial dimensions
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted feature maps
    cam = (weights * activations).sum(dim=1)

    # ReLU
    cam = torch.relu(cam)

    # Remove batch dimension
    cam = cam.squeeze(0)

    # Normalize
    cam -= cam.min()

    if cam.max() != 0:
        cam /= cam.max()

    # Convert to numpy
    cam = cam.detach().numpy()

    # Resize heatmap to image size
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap = np.array(cam_image) / 255.0

    # --------------------------------------------------
    # Create overlay
    # --------------------------------------------------

    original_array = np.array(display_image) / 255.0

    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(original_array)
    plt.title("Original Fundus Image")
    plt.axis("off")

    # Grad-CAM
    plt.subplot(1, 2, 2)

    plt.imshow(original_array)
    plt.imshow(
        heatmap,
        alpha=0.5,
        cmap="jet"
    )

    plt.title(
        f"Grad-CAM\n"
        f"{CLASS_NAMES[predicted_class]} "
        f"({confidence * 100:.2f}%)"
    )

    plt.axis("off")

    plt.tight_layout()

    # Save result
    image_name_arr = (Path(image_path).name).split(".")

    output_path = f"images/test_gradcam{image_name_arr[0]}.jpg"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Prediction")
    print("----------")
    print(f"Class:      {predicted_class}")
    print(f"Diagnosis:  {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print()
    print(f"Grad-CAM saved to: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python gradcam.py <image>")
        sys.exit(1)

    generate_gradcam(sys.argv[1])
