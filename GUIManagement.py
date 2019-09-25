import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt for timing'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
    
    def startGUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        # Create widget
        labelStart = QLabel(self)
        labelEnd = QLabel(self)
        pixmapStart = QPixmap('startFrame.jpeg')
        pixmapEnd = QPixmap('startEnd.jpeg')
        labelStart.setPixmap(pixmapStart)
        labelEnd.setPixmap(pixmapEnd)
        screenWidth = 1024
        screenHeight = 800
        self.resize(screenWidth, screenHeight)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())