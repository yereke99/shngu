# train_model.py
#!/usr/bin/env python3
"""
Manual training script for Dynamogram Classification
Run this first to train your AI model before using the GUI

Usage:
    python3.8 train_model.py --epochs 30
    python3.8 train_model.py --epochs 50 --batch-size 8
    python3.8 train_model.py --help
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pdf_processor import EnhancedPDFProcessor
from trainer_fixed import DynamogramTrainer

def check_requirements():
    """Check if all required components are available"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check other dependencies
    required_modules = ['PIL', 'numpy', 'sklearn', 'matplotlib', 'fitz']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} not installed")
            return False
    
    return True

def check_data_folder():
    """Check if PDF data is available"""
    print("\n📊 Checking training data...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ folder not found")
        print("   Create 'data' folder and place your PDF files (1.pdf to 30.pdf)")
        return False
    
    pdf_files = []
    missing_files = []
    
    for i in range(1, 31):
        pdf_file = data_dir / f"{i}.pdf"
        if pdf_file.exists():
            pdf_files.append(pdf_file)
            print(f"✅ {pdf_file.name}")
        else:
            missing_files.append(f"{i}.pdf")
    
    if missing_files:
        print(f"\n⚠️  Missing PDF files: {', '.join(missing_files[:5])}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
    
    print(f"\n📈 Found {len(pdf_files)} PDF files out of 30")
    
    if len(pdf_files) < 5:
        print("❌ Need at least 5 PDF files to train")
        return False
    
    return True

def process_pdfs():
    """Process PDF files to extract images"""
    print("\n🔄 Processing PDF files...")
    
    try:
        processor = PDFDynamogramProcessor()
        result = processor.process_all_pdfs()
        
        total_images = sum(info['image_count'] for info in result.values() if isinstance(info, dict))
        print(f"✅ Extracted {total_images} images from PDFs")
        
        if total_images < 10:
            print("⚠️  Very few images extracted. Check if PDFs contain actual images.")
            return False
        
        # Create train/val split
        train_data, val_data = processor.create_training_split(train_ratio=0.8)
        print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        return len(train_data) > 0
        
    except Exception as e:
        print(f"❌ Error processing PDFs: {str(e)}")
        return False

def train_model(epochs=30, batch_size=16):
    """Train the dynamogram classification model"""
    print(f"\n🎓 Starting model training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    try:
        # Initialize trainer
        trainer = DynamogramTrainer(model_save_dir="models")
        
        # Prepare data
        print("📊 Preparing training data...")
        train_size, val_size = trainer.prepare_data()
        
        if train_size == 0:
            print("❌ No training data found")
            return False
        
        print(f"   Training samples: {train_size}")
        print(f"   Validation samples: {val_size}")
        
        # Initialize model
        print("🏗️  Initializing model...")
        trainer.initialize_model(num_classes=30)
        
        # Modify batch size if requested
        if batch_size != 16:
            from torch.utils.data import DataLoader
            # Recreate data loaders with new batch size
            train_dataset = trainer.train_loader.dataset
            val_dataset = trainer.val_loader.dataset
            
            trainer.train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2,
                pin_memory=True if trainer.device.type == 'cuda' else False
            )
            
            trainer.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2,
                pin_memory=True if trainer.device.type == 'cuda' else False
            )
        
        # Start training
        print("🚀 Training started...")
        trainer.train(num_epochs=epochs)
        
        print("🎉 Training completed successfully!")
        
        # List created models
        models_dir = Path("models")
        model_files = list(models_dir.glob("best_model*.pth"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"💾 Best model saved: {latest_model.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train Dynamogram Classification Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.8 train_model.py                    # Train with default settings
  python3.8 train_model.py --epochs 50        # Train for 50 epochs
  python3.8 train_model.py --batch-size 8     # Use smaller batch size
  python3.8 train_model.py --skip-processing  # Skip PDF processing
        """
    )
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip PDF processing (use existing processed images)')
    parser.add_argument('--force', action='store_true',
                       help='Force training even with warnings')
    
    args = parser.parse_args()
    
    print("🤖 AI-PumpDiag Manual Training")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("processed_images").mkdir(exist_ok=True)
    
    # Check data
    if not check_data_folder():
        if not args.force:
            print("\n❌ Data check failed. Add PDF files to data/ folder")
            sys.exit(1)
        else:
            print("⚠️  Continuing with --force flag...")
    
    # Process PDFs (unless skipped)
    if not args.skip_processing:
        if not process_pdfs():
            if not args.force:
                print("\n❌ PDF processing failed")
                sys.exit(1)
            else:
                print("⚠️  Continuing with --force flag...")
    else:
        print("⏭️  Skipping PDF processing...")
    
    # Train model
    if train_model(epochs=args.epochs, batch_size=args.batch_size):
        print("\n🎉 SUCCESS! Model training completed.")
        print("\nNext steps:")
        print("1. Run 'python main_complete.py' to open the GUI")
        print("2. Go to 'Диагностика' tab to test predictions")
        print("3. Load any dynamogram image and click 'Анализировать'")
    else:
        print("\n❌ FAILED! Training was not successful.")
        sys.exit(1)

if __name__ == "__main__":
    main()