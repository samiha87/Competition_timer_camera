import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import QObject
from detectors import yolo_detect_both, getLatestEndFrame, getLatestStartFrame

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
        self.labelEnd = None
        self.labelStart = None
        self.videoTimer = QtCore.QTimer()

        #Init signals
        self.videoTimer.timeout.connect(self.updateStream)

    def updateStream(self):
        return
        self.pixmapEnd.load("buffer/bufferend.jpg")
        self.pixmapStart.load("buffer/bufferstart.jpg")

        self.pixmapEnd = self.pixmapEnd.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        self.labelEnd.setPixmap(self.pixmapEnd)    
  
        self.pixmapStart = self.pixmapStart.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        self.labelStart.setPixmap(self.pixmapStart)

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
        self.labelStart = QLabel()
        self.labelEnd = QLabel()
        self.pixmapStart = QPixmap('buffer/startFrame.jpg')
        self.pixmapEnd = QPixmap('buffer/endFrame.jpg')
        # Scale start frame
        self.pixmapStart = self.pixmapStart.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        # Scale end frame
        self.pixmapEnd = self.pixmapEnd.scaled(500, 500, QtCore.Qt.KeepAspectRatio)

        self.labelStart.setPixmap(self.pixmapStart)
        self.labelEnd.setPixmap(self.pixmapEnd)

        # Set widgets for layouts
        layoutVideo.addWidget(self.labelStart)
        layoutVideo.addWidget(self.labelEnd)

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
        # Set frame rate to 25fps -> 40ms 
        self.videoTimer.start(40)   
        yolo_detect_both("videos/start_sample.mp4", "videos/start_sample.mp4", "both", "yolo-coco", 270, [-200, 10], 0.5, 0.3, 5)

    def bStop_clicked(self):    
        print("Button 2 clicked")

    def bExit_clicked(self):
        print("Exit program")
        self.gui.quit()