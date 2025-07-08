# pdf_processor.py
import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import json
from pathlib import Path
import hashlib

class PDFDynamogramProcessor:
    """
    Process PDF files containing dynamogram images and extract them for training/prediction
    """
    
    def __init__(self, data_folder="data", processed_folder="processed_images"):
        self.data_folder = Path(data_folder)
        self.processed_folder = Path(processed_folder)
        self.processed_folder.mkdir(exist_ok=True)
        
        # Create subfolders for each class
        for i in range(1, 31):
            (self.processed_folder / f"class_{i}").mkdir(exist_ok=True)
        
        # Load class definitions from your documentation
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
    
    def extract_images_from_pdf(self, pdf_path, class_id):
        """
        Extract dynamogram images from PDF file
        """
        try:
            doc = fitz.open(pdf_path)
            extracted_images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Generate unique filename
                        img_hash = hashlib.md5(img_data).hexdigest()[:8]
                        filename = f"class_{class_id}_page_{page_num}_img_{img_index}_{img_hash}.png"
                        
                        # Save to class folder
                        save_path = self.processed_folder / f"class_{class_id}" / filename
                        pil_image.save(save_path)
                        extracted_images.append(str(save_path))
                        
                        print(f"Extracted: {filename}")
                    
                    pix = None
            
            doc.close()
            return extracted_images
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def process_all_pdfs(self):
        """
        Process all PDF files in the data folder (1.pdf to 30.pdf)
        """
        processed_info = {}
        
        for class_id in range(1, 31):
            pdf_file = self.data_folder / f"{class_id}.pdf"
            
            if pdf_file.exists():
                print(f"Processing {pdf_file} for class {class_id}...")
                extracted_images = self.extract_images_from_pdf(pdf_file, class_id)
                
                processed_info[class_id] = {
                    'class_name': self.class_definitions[class_id],
                    'pdf_file': str(pdf_file),
                    'extracted_images': extracted_images,
                    'image_count': len(extracted_images)
                }
                
                print(f"✅ Class {class_id}: {len(extracted_images)} images extracted")
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
        
        print(f"\n📊 Processing complete! Info saved to {info_file}")
        return processed_info
    
    def get_dataset_info(self):
        """
        Get information about the processed dataset
        """
        info_file = self.processed_folder / "processing_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def create_training_split(self, train_ratio=0.8):
        """
        Create train/validation split from processed images
        """
        import random
        
        dataset_info = self.get_dataset_info()
        if not dataset_info:
            print("No dataset info found. Run process_all_pdfs() first.")
            return None
        
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
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Add to datasets
            for img_path in train_images:
                train_data.append((img_path, class_id - 1))  # 0-based indexing
            
            for img_path in val_images:
                val_data.append((img_path, class_id - 1))
        
        print(f"📊 Dataset split:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        return train_data, val_data

# Example usage
if __name__ == "__main__":
    import io
    
    processor = PDFDynamogramProcessor()
    
    # Process all PDFs
    info = processor.process_all_pdfs()
    
    # Create training split
    train_data, val_data = processor.create_training_split()
    
    print("\n🎯 Ready for training!")