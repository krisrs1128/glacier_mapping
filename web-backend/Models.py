import os
import json
import pathlib

_MODEL_FN = pathlib.Path(
    os.environ["REPO_DIR"],
    "conf",
    "models.json"
)


def _load_model(model):
    if not os.path.exists(_MODEL_FN):
        return False
    return model["model"]


def load_models():
    model_json = json.load(open(_MODEL_FN,"r"))
    models = dict()

    for key, model in model_json.items():
        model_object = _load_model(model)

        if model_object is False:
            print("WARNING: files are missing, we will not be able to serve '%s' model" % (key))
        else:
            models[key] = model_object

    return models
