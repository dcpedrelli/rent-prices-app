# Importing necessary libraries
import pandas as pd
from flask import Flask, request, make_response
import json
import pickle
from renting_houses.RentPrices import RentPrices
from flask_cors import cross_origin


# load model
model = pickle.load(open( 'model/house_model.pkl', 'rb' ))

# instanciate flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
@cross_origin()


def webhook():

    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=11)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r
  
  
# processing the request from dialogflow
def processRequest(req):


    result = req.get("queryResult")
    
    def ace_animal(x):
      x=x.lower()
      if x=='sim':
        animal = 'acept'
      elif x=='não':
        animal = 'not acept'
      return animal
    
    def furn(x):
      x=x.lower()
      if x=='sim':
        fur = 'furnished'
      elif x=='não':
        fur = 'not furnished'
      return fur
    
    #Fetching the data points
    parameters = result.get("parameters")
    city = parameters.get("city").title()
    area=parameters.get("area")
    rooms=parameters.get("rooms")
    bathroom=parameters.get("bathroom")
    parkings = parameters.get("parkings")
    floor=parameters.get("floor")
    animal= ace_animal(parameters.get("animal"))
    furniture= furn(parameters.get("furniture"))
    hoa = parameters.get("hoa")
    tax=parameters.get("tax")
    fire=parameters.get("fire")
    values = [city, area, rooms, bathroom, parkings, floor, animal, furniture, hoa, tax, fire]
    
    #Dumping the data into an array
    keys = ["city","area","rooms","bathroom","parking spaces","floor", "animal","furniture","hoa (R$)", "property tax (R$)","fire insurance (R$)"]
    
    final_features = dict(zip(keys,values))
    
    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')
    
    #Fitting out model with the data points
    if (intent=='aluguel-int'):
      
      df_raw = pd.DataFrame([final_features])
      
      pipeline = RentPrices()
      
      # data preparation
      
      df1 = pipeline.data_preparation(df_raw)
      
      # prediction
      
      prediction = model.predict(df1)
      
      #Returning back the fullfilment text back to DialogFlow
      
      fulfillmentText= "O valor estimado do seu aluguel é: R$ {:.2f} !".format(prediction)
      
      #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
      return {"fulfillmentText": fulfillmentText}


if __name__ == '__main__':
  app.run(port=5000,debug=True)
