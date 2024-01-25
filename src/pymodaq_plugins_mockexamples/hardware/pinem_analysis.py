import tensorflow as tf
from tensorflow import keras
import numpy as np

def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    
    return predictions

class PinemAnalysis : 
    def __init__(self,model_file) : 
        self.model = keras.models.load_model(model_file)

    def normalize(self, data) : 
        M = np.max(data)
        m = np.min(data)
        return (data-m)/(M-m)
    
    def predict(self, data) : 
        ndata = self.normalize(data)
        ndata = np.expand_dims(ndata, axis=0)
        omg, g, offset, fwhm = make_predictions(self.model, ndata)[0]
        return omg, g, offset, fwhm