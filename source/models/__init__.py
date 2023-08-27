###############################################################################
# This file contains basic functions: find_model_using_name, create_model, 
# get_option_setter, ... which is used to create the model for training
###############################################################################

import torch
import importlib
from models.base_model import BaseModel

def find_model_using_name(model_name):
    model_filename = "models." + "model_"  + model_name
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
    return model

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance

def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options
