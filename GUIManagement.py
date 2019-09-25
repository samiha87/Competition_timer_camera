import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import QObject

class App():

    def __init__(self):
        super().__init__()
        self.title = 'PyQt for timing'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.gui = QApplication(sys.argv)
        self.widget = QWidget()
        self.pixmapStart = None
        self.pixmapEnd = None
    
    def updateStream(self, pathStart, pathEnd):
        if pathStart:
            self.pixmapStart.Load(pathStart)
            print("updateStream: start")
        if pathEnd:
            self.pixmapEnd.Load(pathEnd)
            print("updateStream: end")

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
        # Signals
        startButton.clicked.connect(self.bStart_clicked)
        stopButton.clicked.connect(self.bStop_clicked)
        exitButton.clicked.connect(self.bExit_clicked)
        # Create Start and Stop windows
        labelStart = QLabel()
        labelEnd = QLabel()

        pixmapStart = QPixmap('buffer/startFrame.jpg')
        pixmapEnd = QPixmap('buffer/endFrame.jpg')
        # Scale start frame
        pixmapStart = pixmapStart.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        # Scale end frame
        pixmapEnd = pixmapEnd.scaled(500, 500, QtCore.Qt.KeepAspectRatio)

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
        sys.exit(self.gui.exec_())

    def bStart_clicked(self):
        print("Button 1 clicked")

    def bStop_clicked(self):    
        print("Button 2 clicked")

    def bExit_clicked(self):
        print("Exit program")
        self.gui.quit()