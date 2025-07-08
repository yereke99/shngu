# predictor_simple.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from trainer_improved import SimpleDynamogramClassifier

class SimpleDynamogramPredictor:
    """Simple predictor for the improved model"""
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_definitions = self._load_class_definitions()
        self.load_model(model_path)
        
    def _load_class_definitions(self):
        """Load class definitions"""
        return {
            1: {
                "name": "Недостаточный приток жидкости",
                "description": "Нехватка жидкости - насос не заполняется до полного хода",
                "recommendations": "Проверить герметичность затрубного пространства, увеличить дебит притока, скорректировать частоту хода и давление забоя."
            },
            2: {
                "name": "Недостаточный приток жидкости и повышенное трение",
                "description": "Нехватка жидкости + повышенное трение (смазка/износ)",
                "recommendations": "Дополнительно к п.1 - проверить состояние уплотнений и направляющих, очистить/смазать насосный клапан и плунжер."
            },
            3: {
                "name": "Газовое влияние",
                "description": "Попадание газа (частичная ловушка газа)",
                "recommendations": "Установить или проверить газосепаратор, скорректировать глубину насоса, повысить скорость выпуска газа из НКТ."
            },
            4: {
                "name": "Сильная вибрация",
                "description": "Сильная вибрация (неровная, «зигзаг» петля)",
                "recommendations": "Проверить балансировку кривошипно-шатунного механизма, затянуть все болты, устранить изношенные подшипники, смонтировать антивибрационные опоры."
            },
            5: {
                "name": "Внезапные незначительные колебания притока жидкости",
                "description": "Небольшие кратковременные изменения притока жидкости",
                "recommendations": "Диагностика перекачивающей линии: проверить клапаны, фильтры, давление на приёме; установить буферные ёмкости."
            },
            6: {
                "name": "Внезапное общее газовое влияние",
                "description": "Внезапный общий «газовый шок» - вся петля смещается вниз",
                "recommendations": "Аналогично п.3 + провести динамический анализ состава флюида, увеличить частоту продувки насоса."
            },
            7: {
                "name": "Внезапная утечка через проходной клапан",
                "description": "Протечка пробкового клапана - нижняя ветвь петли «подплывает»",
                "recommendations": "Заменить или отремонтировать клапан, проверить торцевые уплотнения и состояние седла клапана."
            },
            8: {
                "name": "Вытягивание плунжера из цилиндра",
                "description": "Срывание плунжера из корпуса - петля «слайдится»",
                "recommendations": "Скорректировать длину хода, уменьшить нагрузку, проверить состояние пружинного или механического фиксирования плунжера."
            },
            9: {
                "name": "Внезапный обрыв штанг",
                "description": "Обрыв штанги - загрузка резко падает до нуля",
                "recommendations": "Уменьшить максимальную нагрузку, установить защиту от «fluid pound», усилить контроль по датчикам нагрузки; заменить повреждённый участок штанг."
            },
            10: {
                "name": "Работа в режиме высокой производительности",
                "description": "Полный режим (максимальная нагрузка)",
                "recommendations": "Никаких действий - признак высокоэффективной работы насоса; мониторить для предотвращения износа."
            },
            11: {
                "name": "Критически недостаточный приток жидкости",
                "description": "Критическая нехватка жидкости - почти плоская петля",
                "recommendations": "Срочно приостановить работу, промыть/продувить скважину, увеличить приток жидкости или снизить скорость насоса."
            },
            12: {
                "name": "Удар плунжера о насос",
                "description": "Удар плунжера о корпус (столкновение) - резкий пик на петле",
                "recommendations": "Проверить зазоры и направляющие, отрегулировать торцевые зазоры плунжера, смонтировать амортизаторы."
            },
            13: {
                "name": "Газовое влияние с вибрацией",
                "description": "Газ + вибрация - «зигзаг» с укороченной нижней ветвью",
                "recommendations": "Комбинировать методы п.3 и п.4: установить газосепаратор и антивибрационную опору одновременно."
            },
            14: {
                "name": "Подозрение на отказ штангового зацепа",
                "description": "Вероятный излом несущей балки - асимметричная смещённая петля",
                "recommendations": "Проверить состояние балки и подшипников, провести неразрушающий контроль металла (УЗК, МРТ), заменить дефектный элемент."
            },
            15: {
                "name": "Внезапные значительные колебания притока жидкости",
                "description": "Внезапные крупные скачки притока - ударные перепады нижней ветви",
                "recommendations": "Анализ работы буферных ёмкостей и клапанов приёма; установить демпферы потока, перейти на более плавный режим подачи."
            },
            16: {
                "name": "Внезапное сильное газовое влияние, воздушная пробка",
                "description": "Резкое газовое блокирование (air lock) - нижняя ветвь почти полностью «срезана»",
                "recommendations": "Срочно запустить процедуру дегазировки: увеличить продувку, задействовать химические реагенты для растворения газов, проверить работу газосепаратора."
            },
            17: {
                "name": "Внезапный отказ открытия приёмного клапана",
                "description": "Залипание всасывающего клапана - нижняя ветвь «провисает»",
                "recommendations": "Очистить или заменить всасывающий клапан, проверить седло и пружину клапана; при необходимости улучшить качество жидкости (фильтрацию)."
            },
            18: {
                "name": "Внезапная утечка в колонне НКТ",
                "description": "Утечка в НКТ - петля смещена, протечки при всасывании",
                "recommendations": "Локализовать и устранить течь: проверить сальники, соединения и муфты НКТ, провести гидроизоляцию участков."
            },
            19: {
                "name": "Попадание постороннего предмета в насос",
                "description": "Посторонний предмет в насосе - «зубцы» или хаотичные выступы на ветвях",
                "recommendations": "Остановить насос, разобрать и очистить рабочую камеру, устранить источник загрязнения, установить фильтр перед приёмом."
            },
            20: {
                "name": "Естественный приток (фонтанирование)",
                "description": "Самотечный режим (без сопротивления) - почти прямая линия",
                "recommendations": "Аналогично «insufficient liquid supply»; проверить уровень жидкости в приёмнике, возможно насос идёт «на сухую» или через свободный поток."
            },
            21: {
                "name": "Недостаточный приток жидкости с вибрацией",
                "description": "Нехватка жидкости + вибрация",
                "recommendations": "Комбинация п.1 и п.4: увеличить приток жидкости и одновременно устранить вибрационные причины."
            },
            22: {
                "name": "Удар плунжера и вибрация",
                "description": "Столкновение + вибрация",
                "recommendations": "Совместить рекомендации п.12 и п.4: отрегулировать зазоры и демпферы, укрепить фундамент, смонтировать амортизаторы."
            },
            23: {
                "name": "Вибрация",
                "description": "Общая вибрация",
                "recommendations": "Аналогично п.4: балансировка, затяжка, замена подшипников, опоры."
            },
            24: {
                "name": "Работа под полной нагрузкой",
                "description": "Максимальная производительность - расширенный, «полный» параллелограмм",
                "recommendations": "Признак работы на пределе параметров; обеспечивать регулярное техническое обслуживание во избежание износа."
            },
            25: {
                "name": "Внезапное резкое снижение притока жидкости",
                "description": "Внезапный сильный спад притока - резко уменьшенная нижняя ветвь",
                "recommendations": "Параметр схож с п.11 и п.16: срочная диагностика утечек и газовых пробок, коррекция режима работы, промывка НКТ."
            },
            26: {
                "name": "Внезапный отказ открытия проходного клапана",
                "description": "Заедание обратного клапана - участок не заполняется",
                "recommendations": "Аналогично п.17: очистка/замена клапана, проверка уплотнений, контроль механики клапана."
            },
            27: {
                "name": "Внезапная утечка через приёмный клапан",
                "description": "Протечка обратного клапана - нижняя ветвь «плывёт»",
                "recommendations": "Аналогично п.7: замена клапана, проверка седел, улучшение смазки/фильтрации."
            },
            28: {
                "name": "Внезапное увеличение трения",
                "description": "Внезапный рост трения - верхняя ветвь «зигзаг» или «ступеньки»",
                "recommendations": "Проверить смазку, очистить направляющие и втулки, заменить изношенные компоненты; оценить качество жидкости (наличие абразива)."
            },
            29: {
                "name": "Сильное газовое влияние",
                "description": "Тяжёлое газовое воздействие - хаотичная резкая «зигзаг»-ветвь внизу и вверху",
                "recommendations": "Комбинация п.3 и п.4: интенсивная газовая обезвоживающая обработка + антивибрационная защита; возможна химическая обработка скважины."
            },
            30: {
                "name": "Утечка в насосе",
                "description": "Течь самого насоса - смещённая/суженная петля",
                "recommendations": "Диагностика корпуса насоса: проверить фланцы, уплотнения, приварные швы; провести гидроизоляцию и заменить дефектные элементы корпуса и соединений."
            }
        }
    
    def load_model(self, model_path=None):
        """Load the trained model"""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                print("❌ Models directory not found")
                return False
            
            # Find model file
            if model_path and Path(model_path).exists():
                model_file = Path(model_path)
            else:
                model_files = list(models_dir.glob("best_model*.pth"))
                if not model_files:
                    print("❌ No trained model found")
                    return False
                model_file = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Initialize model
            self.model = SimpleDynamogramClassifier(num_classes=30)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model = None
            return False
    
    def predict(self, image_path, top_k=3):
        """Predict pump condition"""
        if not self.model:
            return {
                'success': False,
                'error': 'Model not loaded',
                'predictions': []
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, min(top_k, 30), dim=1)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                
                # Format results
                predictions = []
                for i in range(len(top_probs)):
                    class_id = top_indices[i] + 1
                    confidence = float(top_probs[i])
                    
                    class_info = self.class_definitions.get(class_id, {
                        'name': f'Class {class_id}',
                        'description': 'No description',
                        'recommendations': 'No recommendations'
                    })
                    
                    predictions.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'name': class_info['name'],
                        'description': class_info['description'],
                        'recommendations': class_info['recommendations']
                    })
                
                return {
                    'success': True,
                    'image_path': str(image_path),
                    'predictions': predictions,
                    'main_prediction': predictions[0] if predictions else None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'predictions': []
            }

# Make it the default predictor
DynamogramPredictor = SimpleDynamogramPredictor