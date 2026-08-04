from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget, QApplication

class RegionSelector(QWidget):
    # Signal emitted when selection is complete: (top, left, width, height)
    region_selected = Signal(int, int, int, int)
    
    def __init__(self):
        super().__init__()
        
        # Set window flags for full-screen cover
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool | 
            Qt.X11BypassWindowManagerHint
        )
        
        # Make background fully transparent so we can custom paint
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        
        # Geometry selection points
        self.start_pos = None
        self.end_pos = None
        self.is_selecting = False
        
        # Determine full virtual screen size (spanning multiple monitors if any)
        self.setGeometry_to_virtual_screen()

    def setGeometry_to_virtual_screen(self):
        """
        Calculates and sets geometry to cover the entire virtual desktop area (all screens).
        """
        desktop = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(desktop)

    def keyPressEvent(self, event):
        # Escape key cancels selection
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
            
            # Calculate rectangle dimensions
            rect = QRect(self.start_pos, self.end_pos).normalized()
            
            # Return coordinates if selection has some size
            if rect.width() > 10 and rect.height() > 10:
                self.region_selected.emit(
                    rect.y(),
                    rect.x(),
                    rect.width(),
                    rect.height()
                )
                
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Step 1: Draw dark translucent mask over the entire screen
        mask_color = QColor(0, 0, 0, 160) # RGB + Alpha
        painter.fillRect(self.rect(), mask_color)
        
        # Step 2: Clear/cut-out the selected region and highlight it
        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            
            # Subtract/translate local geometry since widget is positioned globally
            local_rect = QRect(
                self.mapFromGlobal(rect.topLeft()),
                self.mapFromGlobal(rect.bottomRight())
            )
            
            # Clear selected rectangle in mask
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(local_rect, Qt.transparent)
            
            # Redraw bounding box around cutout
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            border_pen = QPen(QColor("#00E676"), 2, Qt.DashLine) # Cyber green dash border
            painter.setPen(border_pen)
            painter.drawRect(local_rect)
            
            # Draw dimension helper text next to box
            text = f"{rect.width()}x{rect.height()} px"
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(local_rect.bottomLeft() + QPoint(5, 18), text)
            
        # Step 3: Draw help instructions
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        
        instructions = "Drag a Box over the Area to Capture (e.g. Code Editor, Slides)"
        sub_instructions = "Press ESC to Cancel Selection"
        
        # Draw text at the top-center of the main screen
        center_x = self.width() // 2
        painter.drawText(
            QRect(0, 80, self.width(), 40), 
            Qt.AlignCenter, 
            instructions
        )
        
        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QColor("#FF5252")) # Neon red/pink
        painter.drawText(
            QRect(0, 120, self.width(), 30), 
            Qt.AlignCenter, 
            sub_instructions
        )
