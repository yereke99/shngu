# main.py
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from styles import get_style
from utils import is_dark_mode, load_image

class PumpDiagApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.init_ui()
        self.apply_theme()
        
    def init_ui(self):
        self.setWindowTitle('AI-PumpDiag - Диагностика ШГНУ')
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('assets/icon.png') if os.path.exists('assets/icon.png') else QIcon())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel (image display)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('Диагностика насоса')
        title.setObjectName('title')
        layout.addWidget(title)
        
        # File upload section
        upload_group = QGroupBox('Загрузка данных / Деректерді жүктеу')
        upload_layout = QVBoxLayout(upload_group)
        
        self.file_label = QLabel('Файл не выбран / Файл таңдалмаған')
        self.file_label.setObjectName('fileLabel')
        upload_layout.addWidget(self.file_label)
        
        upload_btn = QPushButton('📁 Выбрать динамограмму')
        upload_btn.setObjectName('primaryBtn')
        upload_btn.clicked.connect(self.load_file)
        upload_layout.addWidget(upload_btn)
        
        layout.addWidget(upload_group)
        
        # Analysis section
        analysis_group = QGroupBox('Анализ / Талдау')
        analysis_layout = QVBoxLayout(analysis_group)
        
        analyze_btn = QPushButton('🔍 Анализировать')
        analyze_btn.setObjectName('primaryBtn')
        analyze_btn.clicked.connect(self.analyze_graph)
        analysis_layout.addWidget(analyze_btn)
        
        # Results
        self.result_label = QLabel('Результат анализа появится здесь\nТалдау нәтижесі осында көрсетіледі')
        self.result_label.setObjectName('resultLabel')
        self.result_label.setWordWrap(True)
        analysis_layout.addWidget(self.result_label)
        
        layout.addWidget(analysis_group)
        
        # Recommendations
        rec_group = QGroupBox('Рекомендации / Ұсыныстар')
        rec_layout = QVBoxLayout(rec_group)
        
        self.rec_text = QTextEdit()
        self.rec_text.setObjectName('recText')
        self.rec_text.setMaximumHeight(200)
        self.rec_text.setPlaceholderText('Рекомендации будут показаны после анализа...\nҰсыныстар талдаудан кейін көрсетіледі...')
        rec_layout.addWidget(self.rec_text)
        
        layout.addWidget(rec_group)
        
        layout.addStretch()
        
        # Theme toggle
        theme_btn = QPushButton('🌙 Темная тема')
        theme_btn.setObjectName('secondaryBtn')
        theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(theme_btn)
        
        return panel
        
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setObjectName('imageDisplay')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText('Загрузите динамограмму для отображения\nДинамограмманы жүктеу үшін көрсету')
        self.image_label.setMinimumHeight(400)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return panel
        
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Выберите динамограмму / Динамограмманы таңдаңыз',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)'
        )
        
        if file_path:
            self.current_image = load_image(file_path)
            if self.current_image:
                self.display_image(self.current_image)
                self.file_label.setText(f'Загружено: {os.path.basename(file_path)}')
            else:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось загрузить изображение')
                
    def display_image(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
    def analyze_graph(self):
        if not self.current_image:
            QMessageBox.warning(self, 'Предупреждение', 'Сначала загрузите динамограмму')
            return
            
        # Simple mock analysis
        self.result_label.setText('✅ Анализ завершен\n✅ Талдау аяқталды')
        
        # Mock recommendations based on your document
        recommendations = """Обнаружено: Недостаточный приток жидкости
Табылды: Сұйықтықтың жеткіліксіз ағымы

Рекомендации:
• Проверить герметичность затрубного пространства
• Увеличить дебит притока  
• Скорректировать частоту хода и давление забоя

Ұсыныстар:
• Құбыр астындағы кеңістіктің тығыздығын тексеру
• Ағын дебитін арттыру
• Жүру жиілігі мен түбіндегі қысымды түзету"""
        
        self.rec_text.setText(recommendations)
        
    def toggle_theme(self):
        # This is a simplified theme toggle
        self.apply_theme(not is_dark_mode())
        
    def apply_theme(self, dark=None):
        if dark is None:
            dark = is_dark_mode()
        self.setStyleSheet(get_style(dark))
