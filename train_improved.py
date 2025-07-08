# train_improved.py
#!/usr/bin/env python3
"""
Improved training script with better performance for small datasets
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pdf_processor import EnhancedPDFProcessor
from trainer_improved import ImprovedDynamogramTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Improved Dynamogram Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--augment', type=int, default=15, help='Augmentation factor')
    parser.add_argument('--skip-processing', action='store_true', help='Skip PDF processing')
    
    args = parser.parse_args()
    
    print("🚀 AI-PumpDiag - Improved Training")
    print("=" * 50)
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("processed_images").mkdir(exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = ImprovedDynamogramTrainer()
        
        # Process data if needed
        if not args.skip_processing:
            print("🔄 Processing PDF files...")
            processor = EnhancedPDFProcessor()
            processor.process_all_pdfs()
        
        # Prepare data with heavy augmentation
        print(f"📊 Preparing data with {args.augment}x augmentation...")
        train_size, val_size = trainer.prepare_data(augment_factor=args.augment)
        
        # Initialize model
        print("🏗️  Initializing improved model...")
        trainer.initialize_model()
        
        # Train
        print(f"🎯 Starting training for {args.epochs} epochs...")
        trainer.train(num_epochs=args.epochs)
        
        print("🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Run GUI: python3.8 main_fixed.py")
        print("2. Test prediction on any dynamogram image")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()