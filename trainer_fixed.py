# trainer_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from enhanced_pdf_processor import EnhancedPDFProcessor

class DynamogramDataset(Dataset):
    """
    Dataset class for dynamogram images
    """
    
    def __init__(self, data_list, transform=None):
        """
        data_list: List of tuples (image_path, class_label)
        transform: Image transformations
        """
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DynamogramClassifier(nn.Module):
    """
    CNN model for dynamogram classification
    """
    
    def __init__(self, num_classes=30):
        super(DynamogramClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class DynamogramTrainer:
    """
    Fixed trainer class for the dynamogram classification model
    """
    
    def __init__(self, model_save_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced for dynamograms
            transforms.RandomRotation(degrees=5),    # Small rotation only
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self):
        """
        Prepare training and validation data using Enhanced PDF Processor
        """
        processor = EnhancedPDFProcessor()
        
        # Check if data is already processed
        dataset_info = processor.get_dataset_info()
        if not dataset_info:
            print("🔄 Processing PDF files...")
            processor.process_all_pdfs()
            dataset_info = processor.get_dataset_info()
        
        if not dataset_info:
            raise ValueError("No dataset info found after processing.")
        
        # Collect all available images
        all_data = []
        total_images = 0
        
        print("📊 Available data per class:")
        for class_id in range(1, 31):
            if str(class_id) in dataset_info:
                info = dataset_info[str(class_id)]
            elif class_id in dataset_info:
                info = dataset_info[class_id]
            else:
                continue
            
            images = info.get('extracted_images', [])
            count = len(images)
            total_images += count
            
            if count > 0:
                print(f"   Class {class_id}: {count} images")
                # Add images to dataset (0-based indexing for model)
                for img_path in images:
                    if Path(img_path).exists():
                        all_data.append((img_path, class_id - 1))
            else:
                print(f"   Class {class_id}: 0 images ⚠️")
        
        if not all_data:
            raise ValueError("No training data found. Make sure PDF files are in data/ folder.")
        
        print(f"\n📈 Total available samples: {len(all_data)}")
        
        # Create train/val split
        # Since we have only 1 image per class, we'll use different strategy
        import random
        random.shuffle(all_data)
        
        if len(all_data) < 10:
            # Very few samples - use most for training, few for validation
            val_size = max(1, len(all_data) // 10)  # At least 1 for validation
            train_data = all_data[val_size:]
            val_data = all_data[:val_size]
        else:
            # Normal split
            split_idx = int(len(all_data) * 0.8)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
        
        # Ensure we have both train and val data
        if not train_data:
            # Move some validation data to training
            train_data = val_data[:-1] if len(val_data) > 1 else val_data
            val_data = val_data[-1:] if len(val_data) > 1 else []
        
        if not val_data:
            # Move some training data to validation
            if len(train_data) > 1:
                val_data = train_data[-1:]
                train_data = train_data[:-1]
            else:
                # Duplicate the single sample for validation
                val_data = train_data.copy()
        
        print(f"📊 Data split:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        # Create datasets with data augmentation for training
        train_dataset = DynamogramDataset(train_data, transform=self.train_transform)
        val_dataset = DynamogramDataset(val_data, transform=self.val_transform)
        
        # Create data loaders with smaller batch size for small datasets
        batch_size = min(8, len(train_data))  # Adaptive batch size
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Avoid multiprocessing issues on macOS
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        print(f"📊 Data loaders created:")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        print(f"   Batch size: {batch_size}")
        
        return len(train_data), len(val_data)
    
    def initialize_model(self, num_classes=30):
        """
        Initialize the model, optimizer, and loss function
        """
        self.model = DynamogramClassifier(num_classes=num_classes)
        self.model.to(self.device)
        
        # Optimizer with lower learning rate for small dataset
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0001,  # Lower learning rate
            weight_decay=1e-4
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=15,  # Longer step size
            gamma=0.1
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🏗️  Model initialized with {total_params:,} parameters")
    
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 5 == 0:  # More frequent updates for small datasets
                print(f'   Batch {batch_idx+1}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """
        Validate for one epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, num_epochs=30):
        """
        Main training loop adapted for small datasets
        """
        print(f"🎯 Starting training for {num_epochs} epochs...")
        
        best_val_acc = 0.0
        patience = 15  # Longer patience for small datasets
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"📊 Results:")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(f"best_model_acc_{val_acc:.2f}.pth")
                print(f"💾 New best model saved! Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        
        # Generate final report
        self.generate_training_report(val_preds, val_targets)
    
    def save_model(self, filename):
        """
        Save the trained model
        """
        model_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_architecture': 'DynamogramClassifier',
            'num_classes': 30
        }, model_path)
        
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, filename):
        """
        Load a trained model
        """
        model_path = self.model_save_dir / filename
        if not model_path.exists():
            print(f"❌ Model file {model_path} not found")
            return False
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if not self.model:
            self.initialize_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f"✅ Model loaded from {model_path}")
        return True
    
    def generate_training_report(self, predictions, targets):
        """
        Generate training report with plots and metrics
        """
        # Create plots directory
        plots_dir = self.model_save_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Confusion Matrix (if we have predictions)
        if len(predictions) > 0 and len(targets) > 0:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(targets, predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='Blues')
            ax3.set_title('Confusion Matrix')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        else:
            ax3.text(0.5, 0.5, 'No validation predictions', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Confusion Matrix')
        
        # Sample distribution
        unique_classes = list(set(targets)) if targets else []
        if unique_classes:
            class_counts = [targets.count(c) for c in unique_classes]
            ax4.bar([f"Class {c+1}" for c in unique_classes], class_counts)
            ax4.set_title('Class Distribution in Validation Set')
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Count')
            plt.setp(ax4.get_xticklabels(), rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No class distribution data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Class Distribution')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification report
        if len(predictions) > 0 and len(targets) > 0:
            from sklearn.metrics import classification_report
            unique_labels = sorted(list(set(targets + predictions)))
            class_names = [f"Class_{i+1}" for i in unique_labels]
            report = classification_report(targets, predictions, 
                                         labels=unique_labels,
                                         target_names=class_names,
                                         zero_division=0)
            
            with open(plots_dir / "classification_report.txt", 'w') as f:
                f.write("Dynamogram Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Training samples: {len(self.train_loader.dataset)}\n")
                f.write(f"Validation samples: {len(self.val_loader.dataset)}\n")
                f.write(f"Total epochs: {len(self.train_losses)}\n")
                f.write(f"Best validation accuracy: {max(self.val_accuracies):.2f}%\n\n")
                f.write(report)
        
        print(f"📊 Training report saved to {plots_dir}")

# Example usage
if __name__ == "__main__":
    trainer = DynamogramTrainer()
    
    # Prepare data
    train_size, val_size = trainer.prepare_data()
    
    # Initialize model
    trainer.initialize_model()
    
    # Train
    trainer.train(num_epochs=30)
    
    print("🚀 Training complete! Model saved in models/ directory")