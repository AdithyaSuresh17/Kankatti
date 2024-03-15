import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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

# Define data transformations for test data
data_transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the test dataset
test_data = datasets.ImageFolder(root='C:\\Users\\rjsli\\Desktop\\hackathon\\Test', transform=data_transforms_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load the trained model
num_classes = len(test_data.classes)
model = SimpleClassifier(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Test loop
total_accuracy = 0.0
num_batches = len(test_loader)
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    accuracy = calculate_accuracy(outputs, labels)
    total_accuracy += accuracy

average_accuracy = total_accuracy / num_batches
print(f'Average Accuracy on Test Set: {average_accuracy:.4f}')
