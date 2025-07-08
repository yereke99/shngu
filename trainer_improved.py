# trainer_improved.py
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
import matplotlib
matplotlib.use('Agg')  # Fix threading issue
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from enhanced_pdf_processor import EnhancedPDFProcessor
import random

class DynamogramDataset(Dataset):
    """Enhanced dataset with better augmentation for small datasets"""
    
    def __init__(self, data_list, transform=None, augment_factor=10):
        """
        data_list: List of tuples (image_path, class_label)
        transform: Image transformations
        augment_factor: How many augmented versions per image
        """
        self.original_data = data_list
        self.transform = transform
        self.augment_factor = augment_factor
        
        # Create augmented dataset
        self.data_list = []
        for image_path, label in data_list:
            # Add original
            self.data_list.append((image_path, label, 0))
            # Add augmented versions
            for i in range(augment_factor - 1):
                self.data_list.append((image_path, label, i + 1))
        
        print(f"📊 Dataset created: {len(self.original_data)} original → {len(self.data_list)} augmented samples")
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_path, label, aug_idx = self.data_list[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply stronger augmentation for augmented samples
        if aug_idx > 0 and self.transform:
            # Create stronger augmentation
            strong_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = strong_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleDynamogramClassifier(nn.Module):
    """Simplified model for small datasets"""
    
    def __init__(self, num_classes=30):
        super(SimpleDynamogramClassifier, self).__init__()
        
        # Simpler feature extraction
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImprovedDynamogramTrainer:
    """Improved trainer for small datasets"""
    
    def __init__(self, model_save_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Using device: {self.device}")
        
        # Improved transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self, augment_factor=15):
        """Prepare data with heavy augmentation"""
        processor = EnhancedPDFProcessor()
        
        # Get dataset info
        dataset_info = processor.get_dataset_info()
        if not dataset_info:
            print("🔄 Processing PDF files...")
            processor.process_all_pdfs()
            dataset_info = processor.get_dataset_info()
        
        if not dataset_info:
            raise ValueError("No dataset info found after processing.")
        
        # Collect all available images
        all_data = []
        classes_with_data = []
        
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
            
            if count > 0:
                print(f"   Class {class_id}: {count} images")
                classes_with_data.append(class_id)
                for img_path in images:
                    if Path(img_path).exists():
                        all_data.append((img_path, class_id - 1))  # 0-based indexing
        
        if not all_data:
            raise ValueError("No training data found.")
        
        print(f"\n📈 Total classes with data: {len(classes_with_data)}")
        print(f"📈 Total base samples: {len(all_data)}")
        
        # Smart train/val split to ensure each class has representation
        train_data = []
        val_data = []
        
        # Group by class
        class_data = {}
        for img_path, label in all_data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((img_path, label))
        
        # Split each class
        for label, samples in class_data.items():
            random.shuffle(samples)
            if len(samples) == 1:
                # Single sample: use for training, duplicate for validation
                train_data.extend(samples)
                val_data.extend(samples)  # Same sample for validation
            else:
                # Multiple samples: normal split
                split_idx = max(1, int(len(samples) * 0.8))
                train_data.extend(samples[:split_idx])
                val_data.extend(samples[split_idx:])
        
        print(f"📊 Data split:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = DynamogramDataset(
            train_data, 
            transform=self.train_transform, 
            augment_factor=augment_factor
        )
        val_dataset = DynamogramDataset(
            val_data, 
            transform=self.val_transform, 
            augment_factor=1  # No augmentation for validation
        )
        
        # Create data loaders
        batch_size = 16  # Fixed batch size
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        print(f"📊 Final dataset:")
        print(f"   Training samples: {len(train_dataset)} (augmented)")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        print(f"   Batch size: {batch_size}")
        
        return len(train_dataset), len(val_dataset)
    
    def initialize_model(self, num_classes=30):
        """Initialize the simplified model"""
        # Use simplified model for small dataset
        self.model = SimpleDynamogramClassifier(num_classes=num_classes)
        self.model.to(self.device)
        
        # Optimizer with higher learning rate for small dataset
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001,  # Higher learning rate
            weight_decay=1e-5  # Less regularization
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Monitor validation accuracy
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🏗️  Simplified model initialized with {total_params:,} parameters")
    
    def train_epoch(self):
        """Train for one epoch"""
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
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
    
    def train(self, num_epochs=50):
        """Main training loop with improvements"""
        print(f"🎯 Starting improved training for {num_epochs} epochs...")
        
        best_val_acc = 0.0
        patience = 20  # Longer patience
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_acc)
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
            
            # Stop if learning rate gets too small
            if current_lr < 1e-7:
                print(f"⏹️  Learning rate too small, stopping training")
                break
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        
        # Generate final report (non-blocking)
        self.generate_training_report(val_preds, val_targets)
    
    def save_model(self, filename):
        """Save the trained model"""
        model_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_architecture': 'SimpleDynamogramClassifier',
            'num_classes': 30
        }, model_path)
        
        print(f"💾 Model saved to {model_path}")
    
    def generate_training_report(self, predictions, targets):
        """Generate training report without blocking"""
        try:
            # Create plots directory
            plots_dir = self.model_save_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Create plots
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Loss curves
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Accuracy improvement
            if len(self.val_accuracies) > 1:
                best_acc = max(self.val_accuracies)
                best_epoch = self.val_accuracies.index(best_acc) + 1
                ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.1f}%')
                ax2.legend()
            
            # Learning curve analysis
            ax3.text(0.1, 0.8, f'Training Summary:', transform=ax3.transAxes, fontsize=12, fontweight='bold')
            ax3.text(0.1, 0.7, f'• Total Epochs: {len(self.train_losses)}', transform=ax3.transAxes, fontsize=10)
            ax3.text(0.1, 0.6, f'• Best Val Accuracy: {max(self.val_accuracies):.2f}%', transform=ax3.transAxes, fontsize=10)
            ax3.text(0.1, 0.5, f'• Final Train Accuracy: {self.train_accuracies[-1]:.2f}%', transform=ax3.transAxes, fontsize=10)
            ax3.text(0.1, 0.4, f'• Final Val Accuracy: {self.val_accuracies[-1]:.2f}%', transform=ax3.transAxes, fontsize=10)
            
            # Model info
            ax3.text(0.1, 0.2, f'Model: SimpleDynamogramClassifier', transform=ax3.transAxes, fontsize=10)
            ax3.text(0.1, 0.1, f'Device: {self.device}', transform=ax3.transAxes, fontsize=10)
            ax3.set_title('Training Summary', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            # Data distribution
            if predictions and targets:
                unique_classes = list(set(targets))
                class_counts = [targets.count(c) for c in unique_classes]
                ax4.bar([f"C{c+1}" for c in unique_classes], class_counts, alpha=0.7)
                ax4.set_title('Class Distribution in Validation', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Class')
                ax4.set_ylabel('Count')
                plt.setp(ax4.get_xticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_report.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()  # Important: close to free memory
            
            print(f"📊 Training report saved to {plots_dir}")
            
        except Exception as e:
            print(f"⚠️  Could not generate training report: {str(e)}")

# Replace the original trainer
DynamogramTrainer = ImprovedDynamogramTrainer

if __name__ == "__main__":
    trainer = ImprovedDynamogramTrainer()
    train_size, val_size = trainer.prepare_data()
    trainer.initialize_model()
    trainer.train(num_epochs=30)
    print("🚀 Training complete!")