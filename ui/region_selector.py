from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


class RegionSelector(QWidget):
    region_selected = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)

        self.start_pos = None
        self.end_pos = None
        self.is_selecting = False
        self.set_geometry_to_virtual_screen()

    def set_geometry_to_virtual_screen(self):
        desktop = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(desktop)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.end_pos = self.start_pos
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end_pos = event.globalPosition().toPoint()
            rect = QRect(self.start_pos, self.end_pos).normalized()

            if rect.width() > 10 and rect.height() > 10:
                self.region_selected.emit(
                    rect.y(),
                    rect.x(),
                    rect.width(),
                    rect.height(),
                )
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(5, 9, 16, 185))

        header_rect = QRect(max(24, self.width() // 2 - 230), 34, 460, 70)
        painter.setPen(QPen(QColor(51, 65, 85, 230), 1))
        painter.setBrush(QColor(10, 17, 29, 238))
        painter.drawRoundedRect(header_rect, 12, 12)

        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.drawText(
            QRect(header_rect.x(), header_rect.y() + 9, header_rect.width(), 26),
            Qt.AlignCenter,
            "Select capture area",
        )
        painter.setPen(QColor("#8A9AB0"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRect(header_rect.x(), header_rect.y() + 35, header_rect.width(), 22),
            Qt.AlignCenter,
            "Drag to select a region  ·  ESC to cancel",
        )

        if not (self.start_pos and self.end_pos):
            return

        rect = QRect(self.start_pos, self.end_pos).normalized()
        local_rect = QRect(
            self.mapFromGlobal(rect.topLeft()),
            self.mapFromGlobal(rect.bottomRight()),
        )

        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(local_rect, Qt.transparent)

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor("#60A5FA"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(local_rect, 3, 3)

        size_text = f"{rect.width()} × {rect.height()} px"
        pill_width = max(120, 8 * len(size_text))
        pill_rect = QRect(
            local_rect.left(),
            max(116, local_rect.top() - 34),
            pill_width,
            26,
        )
        painter.setPen(QPen(QColor(59, 130, 246, 210), 1))
        painter.setBrush(QColor(15, 23, 42, 240))
        painter.drawRoundedRect(pill_rect, 9, 9)
        painter.setPen(QColor("#DBEAFE"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(pill_rect, Qt.AlignCenter, size_text)

        painter.setPen(QPen(QColor("#BFDBFE"), 1))
        painter.setBrush(QColor("#2563EB"))
        for point in (
            local_rect.topLeft(),
            local_rect.topRight(),
            local_rect.bottomLeft(),
            local_rect.bottomRight(),
        ):
            painter.drawEllipse(point - QPoint(3, 3), 3, 3)
