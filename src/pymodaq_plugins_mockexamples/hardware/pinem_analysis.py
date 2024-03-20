from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

# TODO : Implement a way for the user to chose what model to load. 
# TODO : Implement some utility function that adapts the code the expected shape from the neural network. (For example)

class PinemAnalysis : 
    """
    Class to use the neural network and perform the data analysis in live.

    Args:
    model_file : str
        Path to the model file (hdf5 format)

    It can be any keras-usable model file.
    """
    def __init__(self,model_file) : 
        self.model = load_model(model_file)

    # TODO : Implement scaling along with the normalization

    def normalize(self, data) :
        """
        Normalize the data between 0 and 1.

        Args:
        data : np.ndarray
            Data to normalize it should be a 1D array.
        """ 
        M = np.max(data)
        m = np.min(data)
        return (data-m)/(M-m)
    
    def predict(self, data) : 
        """
        Predict the shape of the spectrum (i.e. its underlying parameters) using the neural network.

        Args:
        data : np.ndarray
            Data to predict it should be a 1D array.
        """
        ndata = self.normalize(data)
        ndata = np.expand_dims(ndata, axis=0)
        g, rt = self.model.predict(ndata)
        return g, rt