##################################################################################################
# TOOL

import sys

from mltool.classifiers import MLPClassifier

DrawingTool = None
try:
    from PyQt5.QtCore import Qt, QPoint
    from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
    from PyQt5.QtGui import QImage, QPainter, QPen, QKeyEvent, QFont

    class _DrawingTool(QMainWindow):
        """
        A window that allows the user to draw something and see how the classifier responds
        This was (stolen and) adapted from: https://stackoverflow.com/a/51475353
        """

        def __init__(self, classifier:MLPClassifier, img_dims):
            super().__init__()
            self.setWindowTitle(f"DrawingTool -- {classifier}")
            self.classifier = classifier
            self.drawing = False
            self.lastPoint = QPoint()
            self.image:QImage = QImage(img_dims[0], img_dims[1], QImage.Format.Format_Grayscale8)
            self.image.fill(Qt.black)
            self.scale = 20
            self.bottom_h = 20
            self.resize(self.image.width()*self.scale, self.image.height()*self.scale+self.bottom_h)
            self.draw_instructions = True
            self.reclassify()
            self.show()

        def keyPressEvent(self, a0: QKeyEvent) -> None:
            if a0.text().lower() == 'c':
                self.draw_instructions = True
                self.image.fill(Qt.black)
                self.reclassify()
                self.update()
                return
            elif a0.text().lower() == 'l':
                fname = QFileDialog.getOpenFileName(self, "Open an image")
                if not fname:
                    return
                loaded = QImage(fname[0])
                self.image = loaded.scaled(self.image.width(), self.image.height())
                self.image.convertTo(QImage.Format.Format_Grayscale8)
                self.reclassify()
                self.update()
                return
            elif a0.text().lower() == 'r':
                fname = QFileDialog.getOpenFileName(self, "Open an image")
                if not fname:
                    return
                loaded = QImage(fname[0])
                self.image = loaded.scaled(self.image.width(), self.image.height())
                self.image.convertTo(QImage.Format.Format_Grayscale8)
                self.reclassify()
                self.update()
                return
            return super().keyPressEvent(a0)

        def paintEvent(self, event):
            painter = QPainter(self)
            myrect = self.rect()
            myrect.setHeight(myrect.height()-self.bottom_h)
            painter.drawImage(myrect, self.image)
            painter.setFont(QFont("Monospace,mono,serif", 12, [-1, QFont.Bold][0], True))
            painter.fillRect(0,self.width(),self.width(),self.bottom_h,Qt.lightGray)
            txtpoint = QPoint(10,self.width()+15)
            painter.setPen(self.caption_color)
            painter.drawText(txtpoint, self.caption)
            if self.draw_instructions:
                painter.setPen(Qt.white)
                painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignVCenter) + int(Qt.AlignmentFlag.AlignHCenter), "Draw a number with the mouse\npress 'l' to open an image\npress 'r' to load a random object from the dataset")


        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.lastPoint = event.pos()/self.scale

        def mouseMoveEvent(self, event):
            self.draw_instructions = False
            if event.buttons() and Qt.LeftButton and self.drawing:
                painter = QPainter(self.image)
                painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))
                currpos = event.pos()/self.scale
                painter.drawLine(self.lastPoint, currpos)
                self.lastPoint = currpos
                self.reclassify()
                self.update()

        def mouseReleaseEvent(self, event):
            if event.button == Qt.LeftButton:
                self.drawing = False

        def reclassify(self):
            input = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
            ptr = input.constBits()
            ptr.setsize(input.height() * input.width())
            arr:np.ndarray = np.frombuffer(ptr, np.uint8) 
            arrfloat = arr.astype(np.float32)
            print(arrfloat, arrfloat.shape)
            out = self.classifier.classify_single(arrfloat)
            print(out)

            tbl = ""
            ctbl = ""
            for i,c in enumerate(self.classifier._classes):
                val = out[0][i]
                prefix = "\u001b[33m\u001b[1m" if val else ""
                suffix = '\u001b[0m' if val else ''
                ctbl+=f'{prefix}{c}:{val}{suffix}\t'
                if val > 0.66:
                    tbl+=f'{c}:{val} '
            self.caption = tbl if tbl else "(no classes match)"
            if not self.draw_instructions:
                self.caption += " - press 'c' to clear"
            self.caption_color = (Qt.blue if tbl.count(":") == 1 else Qt.black) if tbl else Qt.red
            print(tbl)

        @staticmethod
        def run(classifier, image_dims):
            app = QApplication(sys.argv)
            main = DrawingTool(classifier, image_dims)
            (main) # unused ref
            sys.exit(app.exec_())
    DrawingTool = _DrawingTool
except ImportError as err:
    DrawingTool = err
    


def plot_confusion(classifier):
    pass
