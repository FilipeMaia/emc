"""Library of widgets used througout the modules of the viewer."""
from QtVersions import QtCore, QtGui
from functools import partial

DEFAULT_MAX = 1000000000

class IntegerControll(QtGui.QWidget):
    """Spinbox with buttons to shift the value."""
    valueChanged = QtCore.Signal(int)
    def __init__(self, min_lim, max_lim=None):
        super(IntegerControll, self).__init__()
        self._min_lim = min_lim
        self._max_lim = max_lim
        self._value = min_lim

        self._prev_button = None
        self._next_button = None
        self._max_label = None
        self._spin_box = None

        self._setup_gui()
        self._setup_connections()

    def _setup_gui(self):
        """Create the gui and set layout."""
        self._spin_box = QtGui.QSpinBox()
        self._update_spinbox_max(self._max_lim)
        #self._spin_box.setRange(self._min_lim, self._max_lim)
        self._spin_box.setValue(self._value)

        self._max_label = QtGui.QLabel("")
        self._update_max_label(self._max_lim)

        self._prev_button = QtGui.QPushButton("Previous")
        self._next_button = QtGui.QPushButton("Next")

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._spin_box)
        layout.addWidget(self._max_label)
        layout.addWidget(self._prev_button)
        layout.addWidget(self._next_button)

        self.setLayout(layout)

    def _update_max_label(self, new_max):
        """Change the label indicating the maximum value. Provide the new max"""
        if new_max is not None:
            self._max_label.setText("(%d)" % new_max)
        else:
            self._max_label.setText("")

    def _update_spinbox_max(self, new_max):
        """Set the newe new maximum value of the spin box."""
        if new_max is None:
            new_max = DEFAULT_MAX
        self._spin_box.setRange(self._min_lim, new_max)

    def get_value(self):
        """Get the current value."""
        return self._value

    def change_value(self, new_value):
        """Set the value."""
        if new_value >= self._min_lim and (self._max_lim is None or new_value <= self._max_lim):
            self._value = new_value
            self._spin_box.setValue(self._value)
            self.valueChanged.emit(self._value)

    def shift_value(self, shift):
        """Set the value by shifting the current one."""
        # if (self._value + shift) >= self._min_lim and (self._value + shift) <= self._max_lim:
        #     new_value = self._value + shift
        # else:
        #     new_value = self._min_lim
        # self.change_value(new_value)
        self.change_value(self._value + shift)

    def set_max(self, new_max):
        """Change the upper limit."""
        self._max_lim = new_max
        #self._spin_box.setRange(self._min_lim, self._max_lim)
        self._update_spinbox_max(self._max_lim)
        if self._value > self._max_lim:
            self.change_value(self._max_lim)
        self._update_max_label(self._max_lim)

    def _setup_connections(self):
        """Create qt connections."""
        self._spin_box.editingFinished.connect(self._on_spin_box_changed)
        self._prev_button.clicked.connect(partial(self.shift_value, -1))
        self._next_button.clicked.connect(partial(self.shift_value,  1))

    def _on_spin_box_changed(self):
        """Handle a value change."""
        self.change_value(self._spin_box.value())
