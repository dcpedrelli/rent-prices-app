#import os
import pandas as pd
from flask import Flask, request
import pickle
from renting_houses.RentPrices import RentPrices


# load model
model = pickle.load(open( 'model/house_model.pkl', 'rb' ))

# instanciate flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  test_json = request.get_json()
  
  # collect data
  if test_json:
    if isinstance(test_json, dict): #unique value
      
      df_raw = pd.DataFrame(test_json, index=[0])
      
    else:
      df_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
    # instantiate data preparation
    
    pipeline = RentPrices()
    
    # data preparation
    df1 = pipeline.data_preparation(df_raw)
    
    # prediction
    
    pred = model.predict(df1)
    
    df_raw['prediction'] = pred
    
    return df_raw.to_json(orient = 'records')

if __name__ == '__main__':
  # start Flask
  #port = os.environ.get('PORT',5000)
  #app.run(host='localhost', port = port)
  app.run(port=5000,debug=True)