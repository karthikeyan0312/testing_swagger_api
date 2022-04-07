import numpy as np
import pickle as pk
import gzip
from flask import Flask, jsonify,json
from flasgger import Swagger
from flask_restful import Api, Resource,request
from flasgger.utils import swag_from
import gc
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
api = Api(app)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/swagger/",
}
template = {
  "swagger": "2.0",
  "info": {
    "title": "RESTful API",
    "description": "cricket score prediction",
    "contact": {
      "responsibleOrganization": "ME",
      "responsibleDeveloper": "Me",
      "email": "me@me.com",
      "url": "www.me.com",
    },
    "termsOfService": "http://me.com/terms",
    "version": "1.0.0"
  },
  "schemes": [
    "http",
    "https"
  ],
}

swagger = Swagger(app,config=swagger_config,template=template)

def load_model():
  

    with gzip.open(r"C:\Users\User\Desktop\ipl-score-predictor-main\artifacts\model.pickle.gz", "rb") as f:
        model = pk.load(f)

    with open(r"C:\Users\User\Desktop\ipl-score-predictor-main\artifacts\scaler(1).pickle", "rb") as f:
        scaler = pk.load(f)
    with open(r"C:\Users\User\Desktop\ipl-score-predictor-main\artifacts\columns.json", "r") as f:
        columns = np.array(json.load(f)["columns"])

    with open(r"C:\Users\User\Desktop\ipl-score-predictor-main\artifacts\encodedteams.json", "r") as f:
        teams = json.load(f) 

    return model,scaler,columns,list(teams.keys()), list(columns[7:]) + ["Barabati Stadium"]




def predict_score(overs, wickets, runs, wickets_last_5, runs_last_5, bat_team, bowl_team, venue):
    try:
        model,scaler,columns,teams, venues = load_model()
        teams.sort()
        venues.sort()
        X_pred = np.zeros(columns.size)

        X_pred[0:7] = [overs, wickets, runs, wickets_last_5, runs_last_5, teams.index(bat_team), teams.index(bowl_team)]

        if venue != "Barabati Stadium":
            # because i removed first columns for prevent dummy variable trap
            # and first column of venue was Barabati Stadium
            venue_index = np.where(columns == venue)[0][0]

        X_pred = scaler.transform([X_pred])

        result = model.predict(X_pred)[0]

        del model,scaler

        return result
    except Exception as e:
        print(e)
        return 1 # error code 1

class Randomforest(Resource):
    @swag_from("swagger_config.yml")
    def post(self):

        data = json.loads(request.get_data())
        
        over= data["over"]
        wickets = data["wickets"]
        runs = data["runs"]
        last_5_over_wickets = data["last_5_over_wickets"]
        last_5_over_runs = data["last_5_over_runs"]
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        venue = data["venue"]
        score =  int(predict_score(over, wickets, runs, last_5_over_wickets, last_5_over_runs, batting_team, bowling_team, venue))
        gc.collect()
        return jsonify({"score": score})

api.add_resource(Randomforest, '/v1/model')

if __name__ == "__main__":
    #print(predict_score(7, 0, 52, 0, 24, "Sunrisers Hyderabad", "Delhi Capitals", "Sheikh Zayed Stadium"))

    app.run(debug=True)
    