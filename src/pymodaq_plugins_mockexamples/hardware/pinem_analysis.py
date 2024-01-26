# import tensorflow as tf
# from tensorflow import keras
import numpy as np


from typing import Type
import logging
from pathlib import Path
import pytorch_lightning as pl
import torch

def evaluate_model(model, input):
    # assert input.shape[-1] == model.net.nf
    if len(input.shape) == 1:
        input = np.expand_dims(input, axis=0)
    if input.shape[-2] != 1:
        input = np.expand_dims(input, axis=-2)
    assert len(input.shape) <= 3

    torch_input = torch.tensor(input, dtype=torch.float32).to(model.device)

    return model(torch_input).detach().cpu().numpy()


def load_model(type: Type[pl.LightningModule], path: Path, **kwargs):
    """Load the autoencoder model from the save directory.

    Parameters
    ----------
    type : Type[pl.LightningModule]
        The type of the model to load.
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.

    """
    if not path.exists():
        logging.info("Model not found. Returning None.")
        return None
    model = type.load_from_checkpoint(path, **kwargs)
    return model

# def make_predictions(model, input_data):
#     predictions = model.predict(input_data)
    
#     return predictions

class PinemAnalysis : 
    def __init__(self,model_file) : 
        self.model = load_model(model_file)

    def remove_background(self, data) :
        bkgd = data[:102]
        a = np.ones((102,1))
        return data - np.linalg.inv(a.T@a)@a.T@bkgd

    def normalize(self, data) : 
        M = np.max(data)
        m = np.min(data)
        return (data-m)/(M-m)
    
    def predict(self, data) : 
        bdata = self.remove_background(data)
        ndata = self.normalize(bdata)
        ndata = np.expand_dims(ndata, axis=0)
        g, rt = evaluate_model(self.model, ndata)
        return g, rt