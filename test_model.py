import sys
import torch
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

MODEL_PATH = "models/efficientnet_b0_aptos_ddr_best.pth"

CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}

# Create the same architecture used during training
model = efficientnet_b0(weights=None)

# EfficientNet-B0's classifier
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    5
)

# Load your trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    print("\nPrediction")
    print("----------")
    print(f"Class:       {predicted_class}")
    print(f"Diagnosis:   {CLASS_NAMES[predicted_class]}")
    print(f"Confidence:  {confidence * 100:.2f}%")

    print("\nAll class probabilities")
    print("-----------------------")

    for i, probability in enumerate(probabilities[0]):
        print(
            f"{i}: {CLASS_NAMES[i]:35} "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python test_model.py <image>")
        sys.exit(1)

    predict(sys.argv[1])
