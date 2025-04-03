from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QObject
from PyQt5.QtWidgets import QApplication, QCheckBox

class CheckBoxWidget:
    def __init__(self, parent, label, init_v, var_name, var_parent=None,
                 on_state_changed_callback=None):
        self.parent = parent
        self.check_box = QCheckBox(label, parent)

        self.var_name = var_name
        self.var_parent = var_parent if var_parent is not None else parent

        self.on_state_changed_callback = on_state_changed_callback

        self.check_box.setChecked(init_v)
        self.check_box.stateChanged.connect(self.on_state_changed)

    def on_state_changed(self, state):
        value = (state == Qt.Checked)
        if hasattr(self.var_parent, self.var_name):
            setattr(self.var_parent, self.var_name, value)

        if self.on_state_changed_callback:
            self.on_state_changed_callback()


class KeyTracker(QObject):
    keyPressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, o, e):
        if e.type() == QEvent.KeyPress:
            self.keyPressed.emit(e.key())
            return True
        return super().eventFilter(o, e)
