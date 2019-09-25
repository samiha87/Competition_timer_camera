import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon, QPixmap

class App():

    def __init__(self):
        super().__init__()
        self.title = 'PyQt for timing'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.gui = QApplication([])
        self.widget = QWidget()
    
    def start(self):

        self.widget.setWindowTitle(self.title)
        self.widget.setGeometry(self.left, self.top, self.width, self.height)
        # Create layout
        layoutMain = QVBoxLayout()
        layoutVideo = QHBoxLayout()
        layoutButton = QHBoxLayout()
        
        # Create buttons
        startButton = QPushButton('Start')
        stopButton = QPushButton('End')
        exitButton = QPushButton('Exit')
        # Create Start and Stop windows
        labelStart = QLabel()
        labelEnd = QLabel()

        pixmapStart = QPixmap('startFrame.jpeg')
        pixmapEnd = QPixmap('startEnd.jpeg')

        labelStart.setPixmap(pixmapStart)
        labelEnd.setPixmap(pixmapEnd)

        layoutVideo.addWidget(labelStart)
        layoutVideo.addWidget(labelEnd)

        layoutButton.addWidget(startButton)
        layoutButton.addWidget(stopButton)
        layoutButton.addWidget(exitButton)

        # Init layout
        layoutMain.addLayout(layoutVideo)
        layoutMain.addLayout(layoutButton)
        self.widget.setLayout(layoutMain)
        screenWidth = 1024
        screenHeight = 800
        self.widget.resize(screenWidth, screenHeight)
        self.widget.show()
        self.gui.exec_()
