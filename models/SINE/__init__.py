import os

from models.base_model import BaseModel


def import_model():
    from .SINE_model import SINEModel

    return SINEModel


path_components = os.path.abspath(__file__).split(os.sep)
models_index = path_components.index("models")
if models_index + 1 < len(path_components):
    model_name = path_components[models_index + 1]

BaseModel.register(model_name, import_model)
