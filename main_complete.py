# main_fixed.py
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pathlib import Path
import json
from datetime import datetime

# Import the FIXED modules
from enhanced_pdf_processor import EnhancedPDFProcessor
from trainer_fixed import DynamogramTrainer

class TrainingWorker(QThread):
    """Worker thread for model training"""
    
    progress_update = pyqtSignal(str)
    training_complete = pyqtSignal(bool, str)
    
    def __init__(self, num_epochs=30):
        super().__init__()
        self.num_epochs = num_epochs
        
    def run(self):
        try:
            self.progress_update.emit("🔄 Initializing trainer...")
            trainer = DynamogramTrainer()
            
            self.progress_update.emit("📊 Preparing data from PDFs...")
            train_size, val_size = trainer.prepare_data()
            
            if train_size == 0:
                self.training_complete.emit(False, "No training data found. Please add PDF files to data/ folder.")
                return
            
            self.progress_update.emit(f"🏗️  Initializing model...")
            trainer.initialize_model()
            
            self.progress_update.emit(f"🎯 Starting training ({train_size} train, {val_size} val samples)...")
            trainer.train(num_epochs=self.num_epochs)
            
            self.training_complete.emit(True, "Training completed successfully!")
            
        except Exception as e:
            self.training_complete.emit(False, f"Training failed: {str(e)}")

class PredictionWorker(QThread):
    """Worker thread for AI prediction"""
    
    prediction_complete = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
    
    def run(self):
        try:
            # Use the fixed predictor
            from predictor_fixed import DynamogramPredictor
            predictor = DynamogramPredictor()
            result = predictor.predict(self.image_path, top_k=3)
            self.prediction_complete.emit(result)
        except Exception as e:
            self.prediction_error.emit(str(e))

class PDFProcessingWorker(QThread):
    """Worker thread for PDF processing"""
    
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def run(self):
        try:
            self.progress_update.emit("🔄 Processing PDF files...")
            processor = EnhancedPDFProcessor()
            result = processor.process_all_pdfs()
            self.processing_complete.emit(result)
        except Exception as e:
            self.processing_error.emit(str(e))

class PumpDiagApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.predictor = None
        self.analysis_history = []
        
        self.init_ui()
        self.apply_theme()
        self.check_model_status()
        
    def init_ui(self):
        self.setWindowTitle('AI-PumpDiag - Диагностика ШГНУ с обучением на PDF')
        self.setGeometry(100, 100, 1500, 1000)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_prediction_tab()
        self.create_training_tab()
        self.create_data_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Готов к работе / Ready")
        self.status_bar.addWidget(self.status_label)
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('Файл')
        
        open_action = QAction('Открыть изображение', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Экспорт результатов', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Model menu
        model_menu = menubar.addMenu('Модель')
        
        train_action = QAction('Обучить модель', self)
        train_action.triggered.connect(self.start_training)
        model_menu.addAction(train_action)
        
        model_menu.addSeparator()
        
        reload_action = QAction('Перезагрузить модель', self)
        reload_action.triggered.connect(self.reload_model)
        model_menu.addAction(reload_action)
        
        # Help menu
        help_menu = menubar.addMenu('Помощь')
        
        about_action = QAction('О программе', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_prediction_tab(self):
        """Create the prediction tab"""
        prediction_tab = QWidget()
        self.tab_widget.addTab(prediction_tab, "🔍 Диагностика")
        
        layout = QHBoxLayout(prediction_tab)
        
        # Left panel
        left_panel = self.create_prediction_left_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel
        right_panel = self.create_prediction_right_panel()
        layout.addWidget(right_panel, 2)
    
    def create_prediction_left_panel(self):
        """Create left panel for prediction tab"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('AI Диагностика насоса')
        title.setObjectName('title')
        layout.addWidget(title)
        
        # Model status
        self.model_status_group = QGroupBox('Статус модели')
        model_status_layout = QVBoxLayout(self.model_status_group)
        
        self.model_status_label = QLabel('Проверка модели...')
        self.model_status_label.setWordWrap(True)
        model_status_layout.addWidget(self.model_status_label)
        
        reload_model_btn = QPushButton('🔄 Перезагрузить модель')
        reload_model_btn.setObjectName('secondaryBtn')
        reload_model_btn.clicked.connect(self.reload_model)
        model_status_layout.addWidget(reload_model_btn)
        
        layout.addWidget(self.model_status_group)
        
        # File upload section
        upload_group = QGroupBox('Загрузка изображения')
        upload_layout = QVBoxLayout(upload_group)
        
        self.file_label = QLabel('Файл не выбран')
        self.file_label.setObjectName('fileLabel')
        upload_layout.addWidget(self.file_label)
        
        upload_btn = QPushButton('📁 Выбрать динамограмму')
        upload_btn.setObjectName('primaryBtn')
        upload_btn.clicked.connect(self.load_file)
        upload_layout.addWidget(upload_btn)
        
        layout.addWidget(upload_group)
        
        # Analysis section
        analysis_group = QGroupBox('AI Анализ')
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton('🔍 Анализировать')
        self.analyze_btn.setObjectName('primaryBtn')
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        
        self.stop_btn = QPushButton('⏹ Остановить')
        self.stop_btn.setObjectName('secondaryBtn')
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        analysis_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)
        
        # Main result
        self.result_label = QLabel('Результат анализа появится здесь')
        self.result_label.setObjectName('resultLabel')
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(80)
        analysis_layout.addWidget(self.result_label)
        
        # Confidence bar
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel('Уверенность:'))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setVisible(False)
        confidence_layout.addWidget(self.confidence_bar)
        analysis_layout.addLayout(confidence_layout)
        
        layout.addWidget(analysis_group)
        
        # Top predictions
        predictions_group = QGroupBox('Топ-3 предсказания')
        predictions_layout = QVBoxLayout(predictions_group)
        
        self.predictions_list = QListWidget()
        self.predictions_list.setMaximumHeight(120)
        predictions_layout.addWidget(self.predictions_list)
        
        layout.addWidget(predictions_group)
        
        # Recommendations
        rec_group = QGroupBox('Рекомендации')
        rec_layout = QVBoxLayout(rec_group)
        
        self.rec_text = QTextEdit()
        self.rec_text.setObjectName('recText')
        self.rec_text.setMaximumHeight(200)
        self.rec_text.setPlaceholderText('Рекомендации будут показаны после анализа...')
        rec_layout.addWidget(self.rec_text)
        
        layout.addWidget(rec_group)
        
        layout.addStretch()
        
        return panel
    
    def create_prediction_right_panel(self):
        """Create right panel for prediction tab"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image display header
        header_layout = QHBoxLayout()
        image_title = QLabel('Динамограмма')
        image_title.setObjectName('imageTitle')
        header_layout.addWidget(image_title)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton('🔍+')
        zoom_in_btn.setObjectName('smallBtn')
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton('🔍-')
        zoom_out_btn.setObjectName('smallBtn')
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        reset_zoom_btn = QPushButton('↻')
        reset_zoom_btn.setObjectName('smallBtn')
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(reset_zoom_btn)
        
        header_layout.addStretch()
        header_layout.addLayout(zoom_layout)
        layout.addLayout(header_layout)
        
        # Image display
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setObjectName('imageDisplay')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText('Загрузите динамограмму для отображения')
        self.image_label.setMinimumHeight(400)
        
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Image info
        info_group = QGroupBox('Информация об изображении')
        info_layout = QVBoxLayout(info_group)
        
        self.image_info_label = QLabel('Файл не загружен')
        self.image_info_label.setWordWrap(True)
        info_layout.addWidget(self.image_info_label)
        
        layout.addWidget(info_group)
        
        return panel
    
    def create_training_tab(self):
        """Create the training tab"""
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "🎓 Обучение")
        
        layout = QVBoxLayout(training_tab)
        
        # Training info
        info_group = QGroupBox('Информация об обучении')
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
        Для обучения модели поместите PDF файлы в папку data/:
        
        📁 data/
        ├── 1.pdf  - Недостаточный приток жидкости
        ├── 2.pdf  - Недостаточный приток жидкости и повышенное трение
        ├── 3.pdf  - Газовое влияние
        ├── ...
        └── 30.pdf - Утечка в насосе
        
        Каждый PDF должен содержать примеры динамограмм соответствующего класса.
        """)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Training controls
        controls_group = QGroupBox('Управление обучением')
        controls_layout = QVBoxLayout(controls_group)
        
        # Epochs selection
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel('Количество эпох:'))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 100)
        self.epochs_spin.setValue(30)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        controls_layout.addLayout(epochs_layout)
        
        # Training buttons
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton('🎓 Начать обучение')
        self.train_btn.setObjectName('primaryBtn')
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)
        
        self.stop_train_btn = QPushButton('⏹ Остановить обучение')
        self.stop_train_btn.setObjectName('secondaryBtn')
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        button_layout.addWidget(self.stop_train_btn)
        
        controls_layout.addLayout(button_layout)
        
        # Training progress
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setVisible(False)
        controls_layout.addWidget(self.train_progress_bar)
        
        layout.addWidget(controls_group)
        
        # Training log
        log_group = QGroupBox('Лог обучения')
        log_layout = QVBoxLayout(log_group)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(300)
        log_layout.addWidget(self.training_log)
        
        clear_log_btn = QPushButton('🗑️ Очистить лог')
        clear_log_btn.setObjectName('secondaryBtn')
        clear_log_btn.clicked.connect(self.training_log.clear)
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
    
    def create_data_tab(self):
        """Create the data management tab"""
        data_tab = QWidget()
        self.tab_widget.addTab(data_tab, "📊 Данные")
        
        layout = QVBoxLayout(data_tab)
        
        # Data status
        status_group = QGroupBox('Статус данных')
        status_layout = QVBoxLayout(status_group)
        
        self.data_status_label = QLabel('Проверка данных...')
        self.data_status_label.setWordWrap(True)
        status_layout.addWidget(self.data_status_label)
        
        # Data processing buttons
        button_layout = QHBoxLayout()
        
        process_btn = QPushButton('🔄 Обработать PDF файлы')
        process_btn.setObjectName('primaryBtn')
        process_btn.clicked.connect(self.process_pdfs)
        button_layout.addWidget(process_btn)
        
        check_data_btn = QPushButton('📊 Проверить данные')
        check_data_btn.setObjectName('secondaryBtn')
        check_data_btn.clicked.connect(self.check_data_status)
        button_layout.addWidget(check_data_btn)
        
        status_layout.addLayout(button_layout)
        
        layout.addWidget(status_group)
        
        # Data statistics
        stats_group = QGroupBox('Статистика данных')
        stats_layout = QVBoxLayout(stats_group)
        
        self.data_stats_table = QTableWidget()
        self.data_stats_table.setColumnCount(4)
        self.data_stats_table.setHorizontalHeaderLabels(['Класс', 'Название', 'PDF файл', 'Изображений'])
        self.data_stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.data_stats_table)
        
        layout.addWidget(stats_group)
        
        # Processing log
        proc_log_group = QGroupBox('Лог обработки')
        proc_log_layout = QVBoxLayout(proc_log_group)
        
        self.processing_log = QTextEdit()
        self.processing_log.setReadOnly(True)
        self.processing_log.setMaximumHeight(200)
        proc_log_layout.addWidget(self.processing_log)
        
        layout.addWidget(proc_log_group)
    
    def check_model_status(self):
        """Check if trained model exists"""
        try:
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("best_model*.pth"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.model_status_label.setText(f"✅ Модель найдена: {latest_model.name}")
                    # Initialize predictor with fixed version
                    try:
                        from predictor_fixed import DynamogramPredictor
                        self.predictor = DynamogramPredictor()
                        return True
                    except Exception as e:
                        self.model_status_label.setText(f"⚠️ Модель найдена, но не загружается: {str(e)}")
                        return False
                else:
                    self.model_status_label.setText("❌ Обученная модель не найдена. Требуется обучение.")
            else:
                self.model_status_label.setText("❌ Папка models не найдена. Требуется обучение.")
        except Exception as e:
            self.model_status_label.setText(f"❌ Ошибка проверки модели: {str(e)}")
        
        return False
    
    def check_data_status(self):
        """Check data folder status"""
        try:
            data_dir = Path("data")
            if not data_dir.exists():
                self.data_status_label.setText("❌ Папка data не найдена. Создайте папку data и поместите PDF файлы.")
                return False
            
            pdf_files = list(data_dir.glob("*.pdf"))
            expected_files = [f"{i}.pdf" for i in range(1, 31)]
            found_files = [f.name for f in pdf_files if f.name in expected_files]
            
            status_text = f"📊 Найдено {len(found_files)} из 30 необходимых PDF файлов:\n"
            
            # Show only first 10 and last 5 for compact display
            for i in range(1, 11):
                filename = f"{i}.pdf"
                if filename in found_files:
                    status_text += f"✅ {filename} "
                else:
                    status_text += f"❌ {filename} "
            
            status_text += f"\n... и ещё {len(found_files) - 10 if len(found_files) > 10 else 0} файлов"
            
            self.data_status_label.setText(status_text)
            
            # Update data statistics table
            self.update_data_stats_table()
            
            return len(found_files) > 0
            
        except Exception as e:
            self.data_status_label.setText(f"❌ Ошибка проверки данных: {str(e)}")
            return False
    
    def update_data_stats_table(self):
        """Update the data statistics table"""
        try:
            processor = EnhancedPDFProcessor()
            dataset_info = processor.get_dataset_info()
            
            if dataset_info:
                self.data_stats_table.setRowCount(30)
                
                for i, (class_id, info) in enumerate(dataset_info.items()):
                    if isinstance(class_id, str):
                        class_id = int(class_id)
                    
                    self.data_stats_table.setItem(i, 0, QTableWidgetItem(str(class_id)))
                    self.data_stats_table.setItem(i, 1, QTableWidgetItem(info['class_name']))
                    
                    pdf_status = "✅ Есть" if Path(info['pdf_file']).exists() else "❌ Нет"
                    self.data_stats_table.setItem(i, 2, QTableWidgetItem(pdf_status))
                    
                    self.data_stats_table.setItem(i, 3, QTableWidgetItem(str(info['image_count'])))
            else:
                self.data_stats_table.setRowCount(0)
                
        except Exception as e:
            self.processing_log.append(f"❌ Ошибка обновления статистики: {str(e)}")
    
    def process_pdfs(self):
        """Process PDF files to extract images"""
        if not self.check_data_status():
            QMessageBox.warning(self, "Предупреждение", "Нет PDF файлов для обработки")
            return
        
        self.processing_log.append("🔄 Начинается обработка PDF файлов...")
        
        # Start processing in worker thread
        self.pdf_worker = PDFProcessingWorker()
        self.pdf_worker.processing_complete.connect(self.on_pdf_processing_complete)
        self.pdf_worker.processing_error.connect(self.on_pdf_processing_error)
        self.pdf_worker.progress_update.connect(self.processing_log.append)
        self.pdf_worker.start()
    
    def on_pdf_processing_complete(self, result):
        """Handle PDF processing completion"""
        total_images = sum(info['image_count'] for info in result.values() if isinstance(info, dict))
        self.processing_log.append(f"✅ Обработка завершена! Извлечено {total_images} изображений")
        
        # Update data statistics
        self.update_data_stats_table()
        
        QMessageBox.information(self, "Успешно", f"PDF файлы обработаны! Извлечено {total_images} изображений")
    
    def on_pdf_processing_error(self, error):
        """Handle PDF processing error"""
        self.processing_log.append(f"❌ Ошибка: {error}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка обработки PDF: {error}")
    
    def start_training(self):
        """Start model training"""
        # First check if we have processed images
        processor = EnhancedPDFProcessor()
        dataset_info = processor.get_dataset_info()
        
        if not dataset_info:
            QMessageBox.warning(self, "Предупреждение", "Сначала обработайте PDF файлы на вкладке 'Данные'")
            return
        
        total_images = sum(info['image_count'] for info in dataset_info.values() if isinstance(info, dict))
        if total_images == 0:
            QMessageBox.warning(self, "Предупреждение", "Нет извлеченных изображений для обучения")
            return
        
        reply = QMessageBox.question(
            self, 
            'Подтверждение', 
            f'Начать обучение модели?\nИзображений для обучения: {total_images}\nЭто может занять длительное время.',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Disable training button
            self.train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            self.train_progress_bar.setVisible(True)
            self.train_progress_bar.setRange(0, 0)  # Indeterminate
            
            # Clear log
            self.training_log.clear()
            
            # Start training worker
            num_epochs = self.epochs_spin.value()
            self.training_worker = TrainingWorker(num_epochs)
            self.training_worker.progress_update.connect(self.on_training_progress)
            self.training_worker.training_complete.connect(self.on_training_complete)
            self.training_worker.start()
    
    def stop_training(self):
        """Stop model training"""
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            self.training_worker.terminate()
            self.training_worker.wait()
            self.on_training_complete(False, "Обучение остановлено пользователем")
    
    def on_training_progress(self, message):
        """Handle training progress updates"""
        self.training_log.append(message)
        self.status_label.setText(message)
        # Auto-scroll to bottom
        self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
    
    def on_training_complete(self, success, message):
        """Handle training completion"""
        # Reset UI
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_progress_bar.setVisible(False)
        
        # Update log
        self.training_log.append(f"\n{'✅' if success else '❌'} {message}")
        
        if success:
            QMessageBox.information(self, "Успешно", "Обучение завершено успешно!")
            self.check_model_status()  # Refresh model status
        else:
            QMessageBox.critical(self, "Ошибка", f"Обучение не удалось: {message}")
        
        self.status_label.setText("Готов к работе")
    
    def load_file(self):
        """Load image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Выберите динамограмму',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)'
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display image"""
        try:
            from PIL import Image
            image = Image.open(file_path)
            
            # Convert to QPixmap
            from PyQt5.QtGui import QPixmap
            pixmap = QPixmap(file_path)
            
            self.current_image = pixmap
            self.current_image_path = file_path
            
            # Display image
            self.display_image(pixmap)
            
            # Update file info
            file_info = self.get_file_info(file_path)
            self.file_label.setText(f'Загружено: {Path(file_path).name}')
            self.image_info_label.setText(file_info)
            
            # Enable analysis if model is available
            if self.predictor:
                self.analyze_btn.setEnabled(True)
                self.status_label.setText("Файл загружен, готов к анализу")
            else:
                self.analyze_btn.setEnabled(False)
                self.status_label.setText("Файл загружен, но модель не доступна")
            
            # Clear previous results
            self.clear_results()
            
        except Exception as e:
            QMessageBox.warning(self, 'Ошибка', f'Не удалось загрузить изображение: {str(e)}')
    
    def get_file_info(self, file_path):
        """Get file information"""
        try:
            stat = os.stat(file_path)
            size_mb = stat.st_size / (1024 * 1024)
            
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
            
            return f"""Файл: {Path(file_path).name}
Размер: {size_mb:.2f} MB
Разрешение: {width} x {height}
Формат: {format_name}"""
        except Exception as e:
            return f"Ошибка получения информации: {str(e)}"
    
    def display_image(self, pixmap):
        """Display image in label"""
        if pixmap:
            # Scale to fit display
            max_size = 800
            if pixmap.width() > max_size or pixmap.height() > max_size:
                scaled_pixmap = pixmap.scaled(
                    max_size, max_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = pixmap
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
    
    def zoom_in(self):
        """Zoom in image"""
        if self.current_image:
            current_pixmap = self.image_label.pixmap()
            if current_pixmap:
                new_size = current_pixmap.size() * 1.2
                scaled_pixmap = self.current_image.scaled(
                    new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.resize(scaled_pixmap.size())
    
    def zoom_out(self):
        """Zoom out image"""
        if self.current_image:
            current_pixmap = self.image_label.pixmap()
            if current_pixmap:
                new_size = current_pixmap.size() * 0.8
                scaled_pixmap = self.current_image.scaled(
                    new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.resize(scaled_pixmap.size())
    
    def reset_zoom(self):
        """Reset image zoom"""
        if self.current_image:
            self.display_image(self.current_image)
    
    def clear_results(self):
        """Clear analysis results"""
        self.result_label.setText('Результат анализа появится здесь')
        self.rec_text.clear()
        self.predictions_list.clear()
        self.confidence_bar.setVisible(False)
    
    def analyze_image(self):
        """Analyze image with AI"""
        if not self.current_image_path:
            QMessageBox.warning(self, 'Предупреждение', 'Сначала загрузите изображение')
            return
        
        if not self.predictor:
            QMessageBox.warning(self, 'Ошибка', 'Модель не загружена. Обучите модель сначала.')
            return
        
        # Start analysis
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Анализ в процессе...")
        
        # Start prediction worker
        self.prediction_worker = PredictionWorker(self.current_image_path)
        self.prediction_worker.prediction_complete.connect(self.on_prediction_complete)
        self.prediction_worker.prediction_error.connect(self.on_prediction_error)
        self.prediction_worker.start()
    
    def stop_analysis(self):
        """Stop analysis"""
        if hasattr(self, 'prediction_worker') and self.prediction_worker.isRunning():
            self.prediction_worker.terminate()
            self.prediction_worker.wait()
        
        self.reset_analysis_ui()
        self.status_label.setText("Анализ остановлен")
    
    def reset_analysis_ui(self):
        """Reset analysis UI"""
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_prediction_complete(self, result):
        """Handle prediction completion"""
        self.reset_analysis_ui()
        
        if result.get('success'):
            main_pred = result['main_prediction']
            
            # Update main result
            self.result_label.setText(
                f"✅ Диагноз: {main_pred['name']}\n"
                f"🎯 Уверенность: {main_pred['confidence']:.1%}\n"
                f"📋 {main_pred['description']}"
            )
            
            # Update confidence bar
            self.confidence_bar.setVisible(True)
            self.confidence_bar.setValue(int(main_pred['confidence'] * 100))
            
            # Update recommendations
            self.rec_text.setText(f"""Рекомендации по устранению:
{main_pred['recommendations']}

---

Дополнительная информация:
• Класс проблемы: №{main_pred['class_id']}
• Время анализа: {datetime.now().strftime('%H:%M:%S')}
• Файл: {Path(self.current_image_path).name}""")
            
            # Update predictions list
            self.predictions_list.clear()
            for i, pred in enumerate(result['predictions']):
                item_text = f"#{i+1}: {pred['name']} ({pred['confidence']:.1%})"
                item = QListWidgetItem(item_text)
                if i == 0:
                    item.setBackground(QBrush(QColor(200, 255, 200)))
                self.predictions_list.addItem(item)
            
            # Add to history
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'file_path': self.current_image_path,
                'result': result
            })
            
            self.status_label.setText("Анализ завершен успешно ✅")
            
        else:
            error_msg = result.get('error', 'Неизвестная ошибка')
            self.on_prediction_error(error_msg)
    
    def on_prediction_error(self, error):
        """Handle prediction error"""
        self.reset_analysis_ui()
        self.status_label.setText(f"Ошибка анализа: {error}")
        QMessageBox.critical(self, "Ошибка анализа", error)
    
    def reload_model(self):
        """Reload the AI model"""
        try:
            from predictor_fixed import DynamogramPredictor
            self.predictor = DynamogramPredictor()
            self.check_model_status()
            QMessageBox.information(self, "Успешно", "Модель перезагружена")
        except Exception as e:
            self.predictor = None
            QMessageBox.critical(self, "Ошибка", f"Не удалось перезагрузить модель: {str(e)}")
    
    def export_results(self):
        """Export analysis results"""
        if not self.analysis_history:
            QMessageBox.warning(self, "Нет данных", "Нет результатов для экспорта")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            'Сохранить результаты',
            f'pump_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            'JSON Files (*.json);;All Files (*)'
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_history, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "Успешно", f"Результаты сохранены в {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "О программе", """
AI-PumpDiag - Диагностика ШГНУ

Система автоматической диагностики штанговых глубинных 
насосных установок на основе анализа динамограмм.

Особенности:
• Обучение на PDF файлах с динамограммами
• 30 классов диагностики
• Глубокое обучение с PyTorch
• Экспорт результатов

Версия: 1.0
        """)
    
    def apply_theme(self):
        """Apply application theme"""
        self.setStyleSheet(self.get_stylesheet())
    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
            color: #333333;
        }
        
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin: 10px 0;
            padding-top: 10px;
            background-color: white;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton#primaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2196F3, stop:1 #1976D2);
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
            margin: 5px;
        }
        
        QPushButton#primaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #1976D2, stop:1 #1565C0);
        }
        
        QPushButton#primaryBtn:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        
        QPushButton#secondaryBtn {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #cccccc;
            padding: 8px;
            border-radius: 4px;
            margin: 5px;
        }
        
        QPushButton#secondaryBtn:hover {
            background-color: #f0f0f0;
        }
        
        QPushButton#smallBtn {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #cccccc;
            padding: 4px 8px;
            border-radius: 3px;
            margin: 2px;
            max-width: 40px;
        }
        
        QLabel#title {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
            margin: 10px 0;
        }
        
        QLabel#imageTitle {
            font-size: 16px;
            font-weight: bold;
            color: #333333;
            margin: 5px 0;
        }
        
        QLabel#fileLabel, QLabel#resultLabel {
            padding: 10px;
            background-color: #ffffff;
            border-radius: 4px;
            border: 1px solid #cccccc;
            font-weight: bold;
        }
        
        QLabel#imageDisplay {
            background-color: #ffffff;
            border: 2px dashed #cccccc;
            border-radius: 8px;
            padding: 20px;
        }
        
        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
        }
        
        QListWidget {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        
        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #eeeeee;
        }
        
        QListWidget::item:hover {
            background-color: #f0f0f0;
        }
        
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 4px;
            text-align: center;
            background-color: #ffffff;
        }
        
        QProgressBar::chunk {
            background-color: #2196F3;
            border-radius: 3px;
        }
        
        QTableWidget {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            gridline-color: #eeeeee;
        }
        
        QTableWidget::item {
            padding: 5px;
        }
        
        QScrollArea {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        
        QStatusBar {
            background-color: #ffffff;
            border-top: 1px solid #cccccc;
        }
        """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("processed_images").mkdir(exist_ok=True)
    
    window = PumpDiagApp()
    window.show()
    
    # Check data status on startup
    window.check_data_status()
    
    sys.exit(app.exec_())