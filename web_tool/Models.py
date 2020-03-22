import os
import json

ROOT_DIR = os.environ["WEBTOOL_ROOT"]
_MODEL_FN = "models.json"


def _load_model(model):
    if not os.path.exists(model["model"]["fn"]):
        return False
    return model["model"]


def load_models():
    model_json = json.load(open(os.path.join(ROOT_DIR,_MODEL_FN),"r"))
    models = dict()

    for key, model in model_json.items():
        model_object = _load_model(model)

        if model_object is False:
            print("WARNING: files are missing, we will not be able to serve '%s' model" % (key))
        else:
            models[key] = model_object

    return models
