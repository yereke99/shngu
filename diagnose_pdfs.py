# diagnose_pdfs.py
#!/usr/bin/env python3
"""
PDF Diagnostic Tool - Check what's inside your PDF files
Run this to understand why no images were extracted

Usage:
    python3.8 diagnose_pdfs.py
    python3.8 diagnose_pdfs.py --detailed
    python3.8 diagnose_pdfs.py --fix
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pdf_processor import EnhancedPDFProcessor

def main():
    parser = argparse.ArgumentParser(description="Diagnose PDF files for image extraction")
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    parser.add_argument('--fix', action='store_true', help='Try to fix and re-extract images')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for image rendering (default: 200)')
    
    args = parser.parse_args()
    
    print("🔍 PDF Diagnostic Tool")
    print("=" * 50)
    
    # Check if data folder exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ folder not found")
        return
    
    # Initialize processor
    processor = EnhancedPDFProcessor()
    
    # Run diagnostics
    print("\n📊 Analyzing PDF files...")
    processor.diagnose_pdfs()
    
    if args.detailed:
        print("\n🔍 Detailed Analysis:")
        analyze_detailed(processor)
    
    if args.fix:
        print(f"\n🛠️  Attempting to fix extraction (DPI: {args.dpi})...")
        fix_extraction(processor, args.dpi)

def analyze_detailed(processor):
    """Perform detailed analysis of PDF files"""
    summary = {
        'total_files': 0,
        'has_images': 0,
        'text_only': 0,
        'empty': 0,
        'encrypted': 0,
        'corrupted': 0
    }
    
    for class_id in range(1, 31):
        pdf_file = processor.data_folder / f"{class_id}.pdf"
        
        if not pdf_file.exists():
            continue
        
        summary['total_files'] += 1
        analysis = processor.analyze_pdf_content(pdf_file)
        
        if 'error' in analysis:
            summary['corrupted'] += 1
            print(f"❌ {class_id}.pdf: {analysis['error']}")
            continue
        
        if analysis['is_encrypted']:
            summary['encrypted'] += 1
        elif analysis['has_images']:
            summary['has_images'] += 1
        elif analysis['has_text']:
            summary['text_only'] += 1
        else:
            summary['empty'] += 1
    
    print("\n📈 Summary:")
    print(f"   Total PDF files: {summary['total_files']}")
    print(f"   Files with images: {summary['has_images']}")
    print(f"   Text-only files: {summary['text_only']}")
    print(f"   Empty files: {summary['empty']}")
    print(f"   Encrypted files: {summary['encrypted']}")
    print(f"   Corrupted files: {summary['corrupted']}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if summary['has_images'] > 0:
        print(f"   ✅ {summary['has_images']} files have extractable images")
    
    if summary['text_only'] > 0:
        print(f"   📝 {summary['text_only']} files are text-based - will render as images")
    
    if summary['empty'] > 0:
        print(f"   ⚠️  {summary['empty']} files are empty - check these files")
    
    if summary['encrypted'] > 0:
        print(f"   🔐 {summary['encrypted']} files are encrypted - may need password")
    
    if summary['corrupted'] > 0:
        print(f"   💥 {summary['corrupted']} files are corrupted - replace these files")

def fix_extraction(processor, dpi):
    """Attempt to fix extraction with enhanced method"""
    print("🔄 Re-processing with enhanced extraction...")
    
    # Process with higher DPI and enhanced methods
    result = processor.process_all_pdfs(dpi=dpi)
    
    total_images = sum(info['image_count'] for info in result.values() if isinstance(info, dict))
    
    if total_images > 0:
        print(f"✅ Success! Extracted {total_images} images")
        
        # Show per-class breakdown
        print("\n📊 Images per class:")
        for class_id in range(1, 31):
            if class_id in result:
                count = result[class_id]['image_count']
                if count > 0:
                    print(f"   Class {class_id}: {count} images")
        
        # Create training split
        train_data, val_data = processor.create_training_split()
        if train_data:
            print(f"\n🎯 Ready for training!")
            print(f"   Training samples: {len(train_data)}")
            print(f"   Validation samples: {len(val_data)}")
            print("\nNext step: python3.8 train_model.py --skip-processing")
        
    else:
        print("❌ Still no images extracted")
        print("\n🔧 Possible solutions:")
        print("1. Check if PDFs open correctly in a PDF viewer")
        print("2. Try converting PDFs to images first:")
        print("   - Use online PDF to PNG converter")
        print("   - Save as high-resolution images")
        print("   - Place images in processed_images/class_X/ folders manually")
        print("3. Increase DPI: python3.8 diagnose_pdfs.py --fix --dpi 300")

if __name__ == "__main__":
    main()