# enhanced_pdf_processor.py
import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import json
from pathlib import Path
import hashlib
import io

class EnhancedPDFProcessor:
    """
    Enhanced PDF processor that handles multiple PDF types:
    - Embedded images
    - Scanned documents (page as image)
    - Vector graphics converted to images
    """
    
    def __init__(self, data_folder="data", processed_folder="processed_images"):
        self.data_folder = Path(data_folder)
        self.processed_folder = Path(processed_folder)
        self.processed_folder.mkdir(exist_ok=True)
        
        # Create subfolders for each class
        for i in range(1, 31):
            (self.processed_folder / f"class_{i}").mkdir(exist_ok=True)
        
        self.class_definitions = self._load_class_definitions()
        
    def _load_class_definitions(self):
        """Load the 30 pump condition definitions"""
        return {
            1: "Недостаточный приток жидкости",
            2: "Недостаточный приток жидкости и повышенное трение", 
            3: "Газовое влияние",
            4: "Сильная вибрация",
            5: "Внезапные незначительные колебания притока жидкости",
            6: "Внезапное общее газовое влияние",
            7: "Внезапная утечка через проходной клапан",
            8: "Вытягивание плунжера из цилиндра",
            9: "Внезапный обрыв штанг",
            10: "Работа в режиме высокой производительности",
            11: "Критически недостаточный приток жидкости",
            12: "Удар плунжера о насос",
            13: "Газовое влияние с вибрацией",
            14: "Подозрение на отказ штангового зацепа",
            15: "Внезапные значительные колебания притока жидкости",
            16: "Внезапное сильное газовое влияние, воздушная пробка",
            17: "Внезапный отказ открытия приёмного клапана",
            18: "Внезапная утечка в колонне НКТ",
            19: "Попадание постороннего предмета в насос",
            20: "Естественный приток (фонтанирование)",
            21: "Недостаточный приток жидкости с вибрацией",
            22: "Удар плунжера и вибрация",
            23: "Вибрация",
            24: "Работа под полной нагрузкой",
            25: "Внезапное резкое снижение притока жидкости",
            26: "Внезапный отказ открытия проходного клапана",
            27: "Внезапная утечка через приёмный клапан",
            28: "Внезапное увеличение трения",
            29: "Сильное газовое влияние",
            30: "Утечка в насосе"
        }
    
    def extract_images_from_pdf(self, pdf_path, class_id, dpi=150):
        """
        Enhanced image extraction from PDF with multiple methods
        """
        extracted_images = []
        
        try:
            print(f"Processing {pdf_path} for class {class_id}...")
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Method 1: Extract embedded images
                embedded_images = self._extract_embedded_images(doc, page, page_num, class_id)
                extracted_images.extend(embedded_images)
                
                # Method 2: Render page as image (for scanned docs)
                if len(embedded_images) == 0:  # Only if no embedded images found
                    page_images = self._render_page_as_image(page, page_num, class_id, dpi)
                    extracted_images.extend(page_images)
                
                # Method 3: Extract vector graphics regions (if needed)
                # This is more complex and can be added if needed
            
            doc.close()
            print(f"✅ Class {class_id}: {len(extracted_images)} images extracted")
            return extracted_images
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            return []
    
    def _extract_embedded_images(self, doc, page, page_num, class_id):
        """Extract embedded images from PDF page"""
        extracted_images = []
        
        try:
            # Get embedded images
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip if not RGB/GRAY
                    if pix.n - pix.alpha >= 4:
                        pix = None
                        continue
                    
                    # Convert to PIL Image
                    if pix.alpha:
                        # RGBA
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    else:
                        # RGB/GRAY
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Filter out very small images (likely icons/logos)
                    if pil_image.width < 100 or pil_image.height < 100:
                        pix = None
                        continue
                    
                    # Generate filename
                    img_hash = hashlib.md5(img_data).hexdigest()[:8]
                    filename = f"class_{class_id}_page_{page_num}_embedded_{img_index}_{img_hash}.png"
                    
                    # Save image
                    save_path = self.processed_folder / f"class_{class_id}" / filename
                    pil_image.save(save_path, "PNG")
                    extracted_images.append(str(save_path))
                    
                    print(f"  📎 Embedded: {filename}")
                    pix = None
                    
                except Exception as e:
                    print(f"  ❌ Error with embedded image {img_index}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"  ❌ Error extracting embedded images: {str(e)}")
        
        return extracted_images
    
    def _render_page_as_image(self, page, page_num, class_id, dpi=150):
        """Render entire PDF page as image (for scanned documents)"""
        extracted_images = []
        
        try:
            # Render page to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Skip if image is too small
            if pil_image.width < 200 or pil_image.height < 200:
                pix = None
                return extracted_images
            
            # Generate filename
            img_hash = hashlib.md5(img_data).hexdigest()[:8]
            filename = f"class_{class_id}_page_{page_num}_rendered_{img_hash}.png"
            
            # Save image
            save_path = self.processed_folder / f"class_{class_id}" / filename
            pil_image.save(save_path, "PNG")
            extracted_images.append(str(save_path))
            
            print(f"  📄 Page rendered: {filename}")
            pix = None
            
        except Exception as e:
            print(f"  ❌ Error rendering page {page_num}: {str(e)}")
        
        return extracted_images
    
    def process_all_pdfs(self, dpi=150):
        """
        Process all PDF files with enhanced extraction
        """
        processed_info = {}
        total_images = 0
        
        print("🔄 Enhanced PDF Processing Started...")
        print(f"   DPI: {dpi}")
        print(f"   Output: {self.processed_folder}")
        
        for class_id in range(1, 31):
            pdf_file = self.data_folder / f"{class_id}.pdf"
            
            if pdf_file.exists():
                extracted_images = self.extract_images_from_pdf(pdf_file, class_id, dpi)
                
                processed_info[class_id] = {
                    'class_name': self.class_definitions[class_id],
                    'pdf_file': str(pdf_file),
                    'extracted_images': extracted_images,
                    'image_count': len(extracted_images)
                }
                
                total_images += len(extracted_images)
                
            else:
                print(f"⚠️  File {pdf_file} not found")
                processed_info[class_id] = {
                    'class_name': self.class_definitions[class_id],
                    'pdf_file': str(pdf_file),
                    'extracted_images': [],
                    'image_count': 0
                }
        
        # Save processing info
        info_file = self.processed_folder / "processing_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(processed_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total images extracted: {total_images}")
        print(f"   Average per class: {total_images/30:.1f}")
        print(f"   Info saved to: {info_file}")
        
        if total_images == 0:
            print("\n❌ No images extracted! Possible issues:")
            print("   1. PDFs contain only text (no images)")
            print("   2. PDFs are protected/encrypted")
            print("   3. Images are in unsupported format")
            print("\n💡 Solutions:")
            print("   - Try higher DPI: python3.8 train_model.py --force")
            print("   - Convert PDFs to image format first")
            print("   - Check if PDFs open correctly in PDF viewer")
        
        return processed_info
    
    def create_training_split(self, train_ratio=0.8):
        """Create train/validation split from processed images"""
        import random
        
        dataset_info = self.get_dataset_info()
        if not dataset_info:
            print("❌ No dataset info found. Run process_all_pdfs() first.")
            return None, None
        
        train_data = []
        val_data = []
        
        for class_id, info in dataset_info.items():
            if isinstance(class_id, str):
                class_id = int(class_id)
            
            images = info['extracted_images']
            if len(images) == 0:
                continue
            
            # Shuffle images
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            
            # Split data
            train_images = images[:split_idx] if split_idx > 0 else []
            val_images = images[split_idx:] if split_idx < len(images) else images[:1] if images else []
            
            # Add to datasets
            for img_path in train_images:
                train_data.append((img_path, class_id - 1))  # 0-based indexing
            
            for img_path in val_images:
                val_data.append((img_path, class_id - 1))
        
        print(f"📊 Dataset split:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def get_dataset_info(self):
        """Get information about the processed dataset"""
        info_file = self.processed_folder / "processing_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def analyze_pdf_content(self, pdf_path):
        """Analyze what's inside a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            analysis = {
                'pages': len(doc),
                'has_images': False,
                'has_text': False,
                'is_encrypted': doc.is_encrypted,
                'images_per_page': [],
                'text_length_per_page': []
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check for images
                images = page.get_images()
                analysis['images_per_page'].append(len(images))
                if len(images) > 0:
                    analysis['has_images'] = True
                
                # Check for text
                text = page.get_text()
                analysis['text_length_per_page'].append(len(text))
                if len(text.strip()) > 0:
                    analysis['has_text'] = True
            
            doc.close()
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def diagnose_pdfs(self):
        """Diagnose all PDF files to understand their content"""
        print("🔍 Diagnosing PDF files...")
        
        for class_id in range(1, 31):
            pdf_file = self.data_folder / f"{class_id}.pdf"
            
            if pdf_file.exists():
                analysis = self.analyze_pdf_content(pdf_file)
                
                if 'error' in analysis:
                    print(f"❌ {class_id}.pdf: {analysis['error']}")
                    continue
                
                print(f"📄 {class_id}.pdf:")
                print(f"   Pages: {analysis['pages']}")
                print(f"   Has images: {analysis['has_images']}")
                print(f"   Has text: {analysis['has_text']}")
                print(f"   Encrypted: {analysis['is_encrypted']}")
                print(f"   Images per page: {analysis['images_per_page']}")
                
                if not analysis['has_images'] and not analysis['has_text']:
                    print("   ⚠️  Empty or corrupted PDF")
                elif analysis['has_text'] and not analysis['has_images']:
                    print("   📝 Text-based PDF (will render as image)")
                elif analysis['has_images']:
                    print("   🖼️  Contains images")

# Update the original pdf_processor.py to use enhanced version
PDFDynamogramProcessor = EnhancedPDFProcessor

# Example usage
if __name__ == "__main__":
    processor = EnhancedPDFProcessor()
    
    # Diagnose PDFs first
    processor.diagnose_pdfs()
    
    # Process with enhanced extraction
    info = processor.process_all_pdfs(dpi=150)
    
    # Create training split
    train_data, val_data = processor.create_training_split()
    
    if train_data:
        print(f"\n🎯 Ready for training with {len(train_data)} samples!")
    else:
        print("\n❌ No training data available")