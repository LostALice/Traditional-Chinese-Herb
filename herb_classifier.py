import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image


class ChineseHerbDataset(Dataset):
    """
    Custom Dataset for loading Chinese herb images
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the herb images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.herb_images = []
        self.herb_labels = []
        self.class_to_idx = {}

        # Build the dataset by scanning the directory
        for i, herb_name in enumerate(os.listdir(root_dir)):
            herb_path = os.path.join(root_dir, herb_name)

            # Ensure it's a directory
            if os.path.isdir(herb_path):
                self.class_to_idx[herb_name] = i

                # Collect image paths for this herb
                for img_name in os.listdir(herb_path):
                    img_path = os.path.join(herb_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.herb_images.append(img_path)
                        self.herb_labels.append(i)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.herb_images)

    def __getitem__(self, idx):
        img_path = self.herb_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.herb_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class HerbClassifier(nn.Module):
    """
    Custom CNN for Chinese Herb Classification
    """

    def __init__(self, num_classes):
        super(HerbClassifier, self).__init__()

        # Use a pre-trained ResNet as feature extractor
        self.feature_extractor = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ChineseHerbClassificationModel:
    def __init__(self, image_dir: str = "./image"):
        """
        Initialize the classification model

        Args:
            image_dir (str): Directory containing herb subdirectories
        """
        # Prepare dataset and dataloader
        self.dataset = ChineseHerbDataset(image_dir)

        # Create mappings
        self.idx_to_class = {v: k for k,
                             v in self.dataset.class_to_idx.items()}

        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False)

        # Initialize model
        self.model = HerbClassifier(num_classes=len(self.dataset.class_to_idx))

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=10):
        """
        Train the Chinese herb classification model

        Args:
            epochs (int): Number of training epochs
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch+1}: Train Loss {train_loss/len(self.train_loader)}, "
                  f"Val Loss {val_loss/len(self.val_loader)}, "
                  f"Accuracy {100 * correct / total}%")

    def save_model(self, path='./model/chinese_herb_classifier.pth'):
        """
        Save the trained model
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='chinese_herb_classifier.pth'):
        """
        Load a pre-trained model
        """
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def classifier(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict the herb class for a given image

        Args:
            image (PIL.Image): Input herb image

        Returns:
            str: Predicted herb name
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return self.idx_to_class[predicted.item()], predicted.item()

    # def train():
    #     # Initialize and train the model
    #     herb_classifier = ChineseHerbClassificationModel('./image')
    #     herb_classifier.train()

    #     # Save the model
    #     herb_classifier.save_model()

    # def predict_img(self, model_path: str = "./model/chinese_herb_classifier.pth", target_path: str = "./test/test.jpg") -> str:
    #     model = ChineseHerbClassificationModel("./image")
    #     model.load_model(path=model_path)
    #     test_image = Image.open(image_path)

    #     result = model.classifier(test_image)
    #     assert result, "None result in classifier"

    #     return result


if __name__ == "__main__":
    ...
