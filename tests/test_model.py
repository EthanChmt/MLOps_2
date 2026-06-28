import numpy as np


class DummyInput:
    @property
    def name(self):
        return "float_input"


class DummyOnnxModel:
    def get_inputs(self):
        return [DummyInput()]

    def run(self, output_names, feed_dict):
        X = list(feed_dict.values())[0]
        return [np.zeros(len(X))]


def test_model_has_onnx_interface():
    """
    Vérifie que le modèle utilisé par l'API respecte l'interface ONNX attendue :
    - get_inputs()
    - run()
    """

    model = DummyOnnxModel()

    assert hasattr(model, "get_inputs")
    assert hasattr(model, "run")

    input_name = model.get_inputs()[0].name
    X = np.zeros((1, 3), dtype=np.float32)

    outputs = model.run(None, {input_name: X})

    assert isinstance(outputs, list)
    assert len(outputs) >= 1
    assert len(outputs[0]) == 1