"""Similar to modelmap module but shows the weight instead of the intensity."""
import module_template
import modelmap_module

class Plugin(module_template.Plugin):
    """Collects all parts of the plugin."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = modelmap_module.ModelmapViewer()
        self._data = modelmap_module.ModelmapData("output/weight")
        self._controll = modelmap_module.ModelmapControll(self._common_controll, self._viewer, self._data)
