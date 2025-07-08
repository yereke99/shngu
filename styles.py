# styles.py
def get_style(dark=False):
    if dark:
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 8px;
            margin: 10px 0;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton#primaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #4CAF50, stop:1 #45a049);
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
            margin: 5px;
        }
        QPushButton#primaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #45a049, stop:1 #3d8b40);
        }
        QPushButton#secondaryBtn {
            background-color: #555555;
            color: white;
            border: 1px solid #777777;
            padding: 8px;
            border-radius: 4px;
            margin: 5px;
        }
        QLabel#title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }
        QLabel#fileLabel {
            padding: 10px;
            background-color: #3c3c3c;
            border-radius: 4px;
            border: 1px solid #555555;
        }
        QLabel#resultLabel {
            padding: 10px;
            background-color: #3c3c3c;
            border-radius: 4px;
            border: 1px solid #555555;
            font-weight: bold;
        }
        QLabel#imageDisplay {
            background-color: #3c3c3c;
            border: 2px dashed #555555;
            border-radius: 8px;
            padding: 20px;
        }
        QTextEdit#recText {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 5px;
        }
        """
    else:
        return """
        QMainWindow {
            background-color: #f5f5f5;
            color: #333333;
        }
        QWidget {
            background-color: #f5f5f5;
            color: #333333;
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
        QPushButton#secondaryBtn {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #cccccc;
            padding: 8px;
            border-radius: 4px;
            margin: 5px;
        }
        QLabel#title {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
            margin: 10px 0;
        }
        QLabel#fileLabel {
            padding: 10px;
            background-color: #ffffff;
            border-radius: 4px;
            border: 1px solid #cccccc;
        }
        QLabel#resultLabel {
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
        QTextEdit#recText {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
        }
        """
