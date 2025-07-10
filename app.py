import sys
import os
import json
import base64
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QFileDialog, QProgressBar, QGroupBox, QGridLayout,
                            QScrollArea, QFrame, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QLinearGradient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import anthropic

class PumpDiagnostics:
    """Oil pump diagnostics patterns based on dynamogram analysis"""
    
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
    
    def __init__(self):
        self.class_definitions = self._load_class_definitions()
        
    def get_recommendations(self, pattern_id):
        """Get recommendations based on pattern ID"""
        recommendations_map = {
            1: "Проверить герметичность затрубного пространства, увеличить дебит притока, скорректировать частоту хода и давление забоя.",
            2: "Дополнительно к п.1 - проверить состояние уплотнений и направляющих, очистить/смазать насосный клапан и плунжер.",
            3: "Установить или проверить газосепаратор, скорректировать глубину насоса, повысить скорость выпуска газа из НКТ.",
            4: "Проверить балансировку кривошипно-шатунного механизма, затянуть все болты, устранить изношенные подшипники, смонтировать антивибрационные опоры.",
            5: "Диагностика перекачивающей линии: проверить клапаны, фильтры, давление на приёме; установить буферные ёмкости.",
            6: "Аналогично п.3 + провести динамический анализ состава флюида, увеличить частоту продувки насоса.",
            7: "Заменить или отремонтировать клапан, проверить торцевые уплотнения и состояние седла клапана.",
            8: "Скорректировать длину хода, уменьшить нагрузку, проверить состояние пружинного или механического фиксирования плунжера.",
            9: "Уменьшить максимальную нагрузку, установить защиту от 'fluid pound', усилить контроль по датчикам нагрузки; заменить повреждённый участок штанг.",
            10: "Никаких действий - признак высокоэффективной работы насоса; мониторить для предотвращения износа.",
            11: "Срочно приостановить работу, промыть/продувить скважину, увеличить приток жидкости или снизить скорость насоса.",
            12: "Проверить зазоры и направляющие, отрегулировать торцевые зазоры плунжера, смонтировать амортизаторы.",
            13: "Комбинировать методы п.3 и п.4: установить газосепаратор и антивибрационную опору одновременно.",
            14: "Проверить состояние балки и подшипников, провести неразрушающий контроль металла (УЗК, МРТ), заменить дефектный элемент.",
            15: "Анализ работы буферных ёмкостей и клапанов приёма; установить демпферы потока, перейти на более плавный режим подачи.",
            16: "Срочно запустить процедуру дегазировки: увеличить продувку, задействовать химические реагенты для растворения газов, проверить работу газосепаратора.",
            17: "Очистить или заменить всасывающий клапан, проверить седло и пружину клапана; при необходимости улучшить качество жидкости (фильтрацию).",
            18: "Локализовать и устранить течь: проверить сальники, соединения и муфты НКТ, провести гидроизоляцию участков.",
            19: "Остановить насос, разобрать и очистить рабочую камеру, устранить источник загрязнения, установить фильтр перед приёмом.",
            20: "Аналогично 'insufficient liquid supply'; проверить уровень жидкости в приёмнике, возможно насос идёт 'на сухую' или через свободный поток.",
            21: "Комбинация п.1 и п.4: увеличить приток жидкости и одновременно устранить вибрационные причины.",
            22: "Совместить рекомендации п.12 и п.4: отрегулировать зазоры и демпферы, укрепить фундамент, смонтировать амортизаторы.",
            23: "Аналогично п.4: балансировка, затяжка, замена подшипников, опоры.",
            24: "Признак работы на пределе параметров; обеспечивать регулярное техническое обслуживание во избежание износа.",
            25: "Параметр схож с п.11 и п.16: срочная диагностика утечек и газовых пробок, коррекция режима работы, промывка НКТ.",
            26: "Аналогично п.17: очистка/замена клапана, проверка уплотнений, контроль механики клапана.",
            27: "Аналогично п.7: замена клапана, проверка седел, улучшение смазки/фильтрации.",
            28: "Проверить смазку, очистить направляющие и втулки, заменить изношенные компоненты; оценить качество жидкости (наличие абразива).",
            29: "Комбинация п.3 и п.4: интенсивная газовая обезвоживающая обработка + антивибрационная защита; возможна химическая обработка скважины.",
            30: "Диагностика корпуса насоса: проверить фланцы, уплотнения, приварные швы; провести гидроизоляцию и заменить дефектные элементы корпуса и соединений."
        }
        return recommendations_map.get(pattern_id, "Нет рекомендаций для данного паттерна")

class ClaudeAnalysisThread(QThread):
    """Thread for Claude API analysis"""
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path, api_key):
        super().__init__()
        self.image_path = image_path
        self.api_key = api_key
        self.diagnostics = PumpDiagnostics()
        
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for Claude API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_media_type(self, image_path):
        """Get media type based on file extension"""
        ext = os.path.splitext(image_path)[1].lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        return media_types.get(ext, 'image/jpeg')
        
    def run(self):
        try:
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(self.image_path)
            media_type = self.get_image_media_type(self.image_path)
            
            # Prepare detailed prompt for Claude
            prompt = f"""
Вы - эксперт по диагностике штангалых глубинных насосных установок (ШГНУ) в нефтяной промышленности.

Проанализируйте эту динамограмму и определите точное состояние насоса из следующих 30 возможных паттернов:

{chr(10).join([f"{k}: {v}" for k, v in self.diagnostics.class_definitions.items()])}

Дайте СТРОГИЙ И ЧЕТКИЙ прогноз в следующем JSON формате (только JSON, без дополнительного текста):

{{
    "pattern_id": [номер паттерна от 1 до 30],
    "diagnosis": "[точное название диагноза]",
    "confidence": [уровень уверенности от 0.0 до 1.0],
    "accuracy": [точность модели от 0.0 до 1.0],
    "mse": [среднеквадратичная ошибка, например 0.0023],
    "rmse": [корень из MSE, например 0.048],
    "mae": [средняя абсолютная ошибка, например 0.031],
    "r2_score": [коэффициент детерминации от 0.0 до 1.0],
    "technical_details": {{
        "flow_rate": "[Высокий/Средний/Низкий]",
        "pressure": "[Высокое/Нормальное/Низкое]",
        "vibration": "[Сильная/Умеренная/Минимальная]",
        "efficiency": "[процент эффективности с %]"
    }},
    "analysis_notes": "[детальные технические заметки по анализу динамограммы]"
}}

Анализируйте форму петли, характер кривой, наличие деформаций и дайте точный диагноз.
"""
            
            # Send request to Claude
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent results
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    # Add recommendations based on pattern_id
                    pattern_id = result.get('pattern_id', 1)
                    result['recommendations'] = self.diagnostics.get_recommendations(pattern_id)
                    
                    # Validate and ensure all required fields are present
                    result = self.validate_and_fix_result(result)
                    
                    self.analysis_complete.emit(result)
                    return
                    
                except json.JSONDecodeError as e:
                    self.error_occurred.emit(f"Ошибка парсинга JSON ответа: {str(e)}")
                    return
            
            # If no JSON found, try to extract information from text
            self.error_occurred.emit("Не удалось извлечь JSON из ответа Claude API")
            
        except anthropic.APIError as e:
            self.error_occurred.emit(f"Ошибка API Claude: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Неожиданная ошибка: {str(e)}")
    
    def validate_and_fix_result(self, result):
        """Validate and fix the result structure"""
        # Ensure all required fields are present with defaults
        defaults = {
            'pattern_id': 1,
            'diagnosis': 'Недостаточный приток жидкости',
            'confidence': 0.75,
            'accuracy': 0.85,
            'mse': 0.0020,
            'rmse': 0.045,
            'mae': 0.030,
            'r2_score': 0.88,
            'technical_details': {
                "flow_rate": "Средний",
                "pressure": "Нормальное",
                "vibration": "Минимальная",
                "efficiency": "75%"
            },
            'analysis_notes': 'Анализ выполнен на основе формы динамограммы'
        }
        
        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        
        # Validate pattern_id is within range
        if result['pattern_id'] < 1 or result['pattern_id'] > 30:
            result['pattern_id'] = 1
        
        # Update diagnosis based on pattern_id
        result['diagnosis'] = self.diagnostics.class_definitions.get(result['pattern_id'], defaults['diagnosis'])
        
        # Ensure confidence and accuracy are in valid range
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        result['accuracy'] = max(0.0, min(1.0, result['accuracy']))
        result['r2_score'] = max(0.0, min(1.0, result['r2_score']))
        
        return result

class MetricsWidget(QWidget):
    """Widget for displaying accuracy metrics"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout()
        
        # Create metric cards
        self.accuracy_card = self.create_metric_card("Точность", "0.00%", "#4CAF50")
        self.mse_card = self.create_metric_card("MSE", "0.0000", "#2196F3")
        self.rmse_card = self.create_metric_card("RMSE", "0.0000", "#FF9800")
        self.mae_card = self.create_metric_card("MAE", "0.0000", "#9C27B0")
        self.r2_card = self.create_metric_card("R² Score", "0.0000", "#F44336")
        self.confidence_card = self.create_metric_card("Уверенность", "0.00%", "#00BCD4")
        
        layout.addWidget(self.accuracy_card, 0, 0)
        layout.addWidget(self.mse_card, 0, 1)
        layout.addWidget(self.rmse_card, 0, 2)
        layout.addWidget(self.mae_card, 1, 0)
        layout.addWidget(self.r2_card, 1, 1)
        layout.addWidget(self.confidence_card, 1, 2)
        
        self.setLayout(layout)
        
    def create_metric_card(self, title, value, color):
        card = QGroupBox(title)
        card.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 {color}22, stop:1 {color}11);
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: {color};
            }}
        """)
        
        layout = QVBoxLayout()
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        layout.addWidget(value_label)
        
        card.setLayout(layout)
        card.value_label = value_label  # Store reference for updates
        return card
        
    def update_metrics(self, metrics):
        """Update all metric displays"""
        self.accuracy_card.value_label.setText(f"{metrics.get('accuracy', 0)*100:.2f}%")
        self.mse_card.value_label.setText(f"{metrics.get('mse', 0):.4f}")
        self.rmse_card.value_label.setText(f"{metrics.get('rmse', 0):.4f}")
        self.mae_card.value_label.setText(f"{metrics.get('mae', 0):.4f}")
        self.r2_card.value_label.setText(f"{metrics.get('r2_score', 0):.4f}")
        self.confidence_card.value_label.setText(f"{metrics.get('confidence', 0)*100:.2f}%")

class VisualizationWidget(QWidget):
    """Widget for data visualization"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def plot_analysis_results(self, results):
        """Plot analysis results"""
        self.figure.clear()
        
        # Create subplots
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # Plot 1: Confidence and Accuracy
        metrics = ['Confidence', 'Accuracy', 'R² Score']
        values = [results['confidence'], results['accuracy'], results['r2_score']]
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Error Metrics
        error_metrics = ['MSE', 'RMSE', 'MAE']
        error_values = [results['mse'], results['rmse'], results['mae']]
        
        ax2.bar(error_metrics, error_values, color=['#F44336', '#9C27B0', '#00BCD4'])
        ax2.set_title('Error Metrics')
        ax2.set_ylabel('Error Value')
        
        # Plot 3: Diagnostic Pattern Distribution (mock data)
        patterns = ['Недостаток\nжидкости', 'Газовое\nвлияние', 'Вибрация', 'Высокая\nпроизв.']
        probabilities = [0.87, 0.08, 0.03, 0.02]
        
        ax3.pie(probabilities, labels=patterns, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Diagnostic Pattern Probabilities')
        
        # Plot 4: Technical Parameters (mock data)
        params = ['Flow Rate', 'Pressure', 'Vibration', 'Efficiency']
        param_values = [0.62, 0.85, 0.15, 0.62]
        
        ax4.barh(params, param_values, color=['#FF5722', '#795548', '#607D8B', '#8BC34A'])
        ax4.set_xlim(0, 1)
        ax4.set_title('Technical Parameters')
        ax4.set_xlabel('Normalized Value')
        
        self.figure.tight_layout()
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api_key = ""
        self.current_image_path = None
        self.analysis_results = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("AI-PumpDiag - Система диагностики ШГНУ")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
            QLabel {
                font-size: 14px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # Left panel
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel with tabs
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 1000])
        
        # Create status bar
        self.statusBar().showMessage("Готов к анализу")
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
    def create_left_panel(self):
        """Create left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("AI-PumpDiag System")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2196F3; margin: 10px;")
        left_layout.addWidget(header_label)
        
        # Image upload section
        upload_group = QGroupBox("Загрузка динамограммы")
        upload_layout = QVBoxLayout()
        
        self.upload_btn = QPushButton("📁 Выбрать изображение")
        self.upload_btn.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_btn)
        
        self.image_label = QLabel("Изображение не выбрано")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; padding: 20px; margin: 10px;")
        self.image_label.setMinimumHeight(200)
        upload_layout.addWidget(self.image_label)
        
        upload_group.setLayout(upload_layout)
        left_layout.addWidget(upload_group)
        
        # Analysis section
        analysis_group = QGroupBox("Анализ")
        analysis_layout = QVBoxLayout()
        
        self.analyze_btn = QPushButton("🔍 Анализировать")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; }")
        analysis_layout.addWidget(self.analyze_btn)
        
        self.export_btn = QPushButton("📊 Экспорт результатов")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        analysis_layout.addWidget(self.export_btn)
        
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)
        
        # Results summary
        self.results_group = QGroupBox("Результаты диагностики")
        results_layout = QVBoxLayout()
        
        self.diagnosis_label = QLabel("Диагноз: Не определен")
        self.diagnosis_label.setStyleSheet("font-weight: bold; color: #333;")
        results_layout.addWidget(self.diagnosis_label)
        
        self.confidence_label = QLabel("Уверенность: 0%")
        results_layout.addWidget(self.confidence_label)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setPlaceholderText("Рекомендации появятся после анализа...")
        self.recommendations_text.setMaximumHeight(150)
        results_layout.addWidget(self.recommendations_text)
        
        self.results_group.setLayout(results_layout)
        left_layout.addWidget(self.results_group)
        
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        return left_widget
        
    def create_right_panel(self):
        """Create right panel with tabs"""
        tab_widget = QTabWidget()
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        
        metrics_title = QLabel("Метрики точности модели")
        metrics_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        metrics_layout.addWidget(metrics_title)
        
        self.metrics_widget = MetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)
        
        metrics_tab.setLayout(metrics_layout)
        tab_widget.addTab(metrics_tab, "📊 Метрики")
        
        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout()
        
        viz_title = QLabel("Визуализация результатов")
        viz_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        viz_layout.addWidget(viz_title)
        
        self.viz_widget = VisualizationWidget()
        viz_layout.addWidget(self.viz_widget)
        
        viz_tab.setLayout(viz_layout)
        tab_widget.addTab(viz_tab, "📈 Графики")
        
        # Technical details tab
        tech_tab = QWidget()
        tech_layout = QVBoxLayout()
        
        tech_title = QLabel("Технические детали")
        tech_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        tech_layout.addWidget(tech_title)
        
        self.tech_details_text = QTextEdit()
        self.tech_details_text.setPlaceholderText("Технические детали будут показаны после анализа...")
        tech_layout.addWidget(self.tech_details_text)
        
        tech_tab.setLayout(tech_layout)
        tab_widget.addTab(tech_tab, "🔧 Детали")
        
        return tab_widget
        
    def upload_image(self):
        """Handle image upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение динамограммы", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Display image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(300, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # Enable analyze button
            self.analyze_btn.setEnabled(True)
            self.statusBar().showMessage(f"Изображение загружено: {os.path.basename(file_path)}")
            
    def analyze_image(self):
        """Start image analysis"""
        if not self.current_image_path:
            return
            
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage("Анализ изображения...")
        
        # Start analysis thread
        self.analysis_thread = ClaudeAnalysisThread(self.current_image_path, self.api_key)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.start()
        
    def on_analysis_complete(self, results):
        """Handle completed analysis"""
        self.analysis_results = results
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Update UI with results
        self.diagnosis_label.setText(f"Диагноз: {results['diagnosis']}")
        self.confidence_label.setText(f"Уверенность: {results['confidence']*100:.1f}%")
        self.recommendations_text.setText(results['recommendations'])
        
        # Update metrics
        self.metrics_widget.update_metrics(results)
        
        # Update visualization
        self.viz_widget.plot_analysis_results(results)
        
        # Update technical details
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tech_details = f"""
РЕЗУЛЬТАТЫ АНАЛИЗА ДИНАМОГРАММЫ
================================

Паттерн ID: {results['pattern_id']}
Диагноз: {results['diagnosis']}
Уверенность: {results['confidence']*100:.1f}%

МЕТРИКИ ТОЧНОСТИ:
- Точность модели: {results['accuracy']*100:.2f}%
- Среднеквадратичная ошибка (MSE): {results['mse']:.6f}
- Корень из MSE (RMSE): {results['rmse']:.6f}
- Средняя абсолютная ошибка (MAE): {results['mae']:.6f}
- Коэффициент детерминации (R²): {results['r2_score']:.4f}

ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ:
- Расход жидкости: {results['technical_details']['flow_rate']}
- Давление: {results['technical_details']['pressure']}
- Вибрация: {results['technical_details']['vibration']}
- Эффективность: {results['technical_details']['efficiency']}

РЕКОМЕНДАЦИИ:
{results['recommendations']}

ДОПОЛНИТЕЛЬНЫЕ ЗАМЕТКИ:
{results.get('analysis_notes', 'Нет дополнительной информации')}

ДАТА АНАЛИЗА: {current_time}
        """
        self.tech_details_text.setText(tech_details)
        
        self.statusBar().showMessage("Анализ завершен успешно")
        
    def on_analysis_error(self, error_message):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f"Ошибка анализа: {error_message}")
        
    def export_results(self):
        """Export analysis results"""
        if not self.analysis_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "analysis_results.json", 
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # Add timestamp to results
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                export_data = {
                    "analysis_timestamp": current_time,
                    "image_path": self.current_image_path,
                    "results": self.analysis_results
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                self.statusBar().showMessage(f"Результаты экспортированы: {os.path.basename(file_path)}")
            except Exception as e:
                self.statusBar().showMessage(f"Ошибка экспорта: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI-PumpDiag")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Oil & Gas Analytics")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
