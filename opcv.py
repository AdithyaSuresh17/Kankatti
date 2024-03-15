import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os

# Define the SimpleClassifier class
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the trained model
num_classes = 7  # Assuming you have 10 classes for different denominations
model = SimpleClassifier(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define data transformation for input frames
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define function to get denomination label
def get_denomination_label(prediction):
    denominations = ["10","20","50","100","200","500","2000"]  # Update with your class labels
    return denominations[prediction]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Preprocess the frame
    img = data_transform(frame).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(img)

    _, predicted_class_idx = torch.max(output, 1)
    denomination = get_denomination_label(predicted_class_idx.item())

    # Display denomination on the frame
    cv2.putText(frame, denomination, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Currency Denomination', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
