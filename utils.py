# utils.py
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTime

def is_dark_mode():
    """Simple dark mode detection based on time"""
    current_time = QTime.currentTime()
    return current_time.hour() < 6 or current_time.hour() > 18

def load_image(file_path):
    """Load image and return QPixmap"""
    if not os.path.exists(file_path):
        return None
    
    pixmap = QPixmap(file_path)
    return pixmap if not pixmap.isNull() else None