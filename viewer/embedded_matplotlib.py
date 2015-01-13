"""Small function to embedd matplotlib visualization in qt."""
# try:
#import matplotlib
#print matplotlib.rcParams["text.usetex"]
#matplotlib.rcParams["text.usetex"] = False
from QtVersions import QtGui
# except IOError:
#     from pyface.qt import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

def get_matplotlib_widget():
    """Returns four objects: A widget, a figure, a canvas an mpl_toolbar."""
    widget = QtGui.QWidget()
    fig = Figure((5., 4.))
    canvas = FigureCanvas(fig)
    canvas.setParent(widget)
    mpl_toolbar = NavigationToolbar(canvas, widget)
    layout = QtGui.QVBoxLayout()
    layout.addWidget(canvas)
    layout.addWidget(mpl_toolbar)
    widget.setLayout(layout)
    return widget, fig, canvas, mpl_toolbar
