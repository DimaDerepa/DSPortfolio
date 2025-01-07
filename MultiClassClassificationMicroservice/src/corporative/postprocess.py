from src.core.interfaces import Postprocessor


class CorporativePostprocessor(Postprocessor):
    """Postprocessor for Corporative flow."""
    def postprocess(self, data, config):
        """
        Add here all postprocessing code for Corporative flow
        Not used for now
        """
        return data
