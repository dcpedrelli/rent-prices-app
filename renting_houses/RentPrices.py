### Class in Production
import pickle

class RentPrices(object):
  def __init__(self):
    self.one_hot_enc = pickle.load(open('parameters/one_hot_enc.pkl','rb'))
    self.stand_scaler = pickle.load(open('parameters/stand_scaler.pkl','rb'))
    
  def data_preparation(self, df):
    
    # One_Hot_Encoding
    
    X1 = self.one_hot_enc.transform(df)
    
    # Standard_Scaler
    
    X = self.stand_scaler.transform(X1)
    
    return X