import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5 import uic
from PyQt5 import QtWidgets, QtGui

qtCreatorFile = "design.ui"  # Enter file here.
global ImageFile
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.browse.clicked.connect(self.Test)
        self.close.clicked.connect(self.Close)

    def Test(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        ImageFile = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image To Process", "","All Files (*);;Image Files(*.jpg *.gif)", options=options)
        exec(open('main.py').read())

    def Close(self):
        self.destroy()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())