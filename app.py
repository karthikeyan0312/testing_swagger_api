import numpy as np
import pickle as pk
import gzip
from flask import Flask, jsonify,json
from flasgger import Swagger
from flask_restful import Api, Resource,request
from flasgger.utils import swag_from
import gc
from cachetools import cached, TTLCache
import os
from flask_cors import CORS
from flask_cors import cross_origin
#to get the current working directory
directory = os.path.dirname(os.path.realpath(__file__))

print(directory)


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)
cache = TTLCache(maxsize=100, ttl=100)

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
    "specs_route": "/",
}
template = {
  "swaggerUI": "3.28.0",
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
    "https",
    "http"
  ],
}

swagger = Swagger(app,config=swagger_config,template=template)

@cached(cache)
def load_model():
  

    with gzip.open(r"/opt/render/project/src/model/model.pickle.gz", "rb") as f:
        model = pk.load(f)

    with open(r"/opt/render/project/src/model/scaler(1).pickle", "rb") as f:
        scaler = pk.load(f)
    with open(r"/opt/render/project/src/model/columns.json", "r") as f:
        columns = np.array(json.load(f)["columns"])

    with open(r"/opt/render/project/src/model/encodedteams.json", "r") as f:
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
        return 1 # error code 1

class Randomforest(Resource):
    @swag_from("swagger_config.yml")
    @cross_origin()
    def post(self):

        data = json.loads(request.get_data())
        if(len(data)!=8):
            return {"Error":"Invalid Input"},400
        
        over= data["over"]
        wickets = data["wickets"]
        runs = data["runs"]
        last_5_over_wickets = data["last_5_over_wickets"]
        last_5_over_runs = data["last_5_over_runs"]
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        venue = data["venue"]
        score =  int(predict_score(over, wickets, runs, last_5_over_wickets, last_5_over_runs, batting_team, bowling_team, venue))
        if score == 1:
            return {"Error":"Invalid Input"},400
        del data
        gc.collect()
        cache.clear()
        return jsonify({"score": score})
    
    @swag_from("swagger_config2.yml")
    def get(self):
        req=request.args.to_dict()["model_details"]
        data=None
        if req == "Best Parameter":
            data={'criterion': 'mse',
                    'max_depth': 20,
                    'max_features': 'log2',
                    'max_leaf_nodes': None,
                    'min_samples_leaf': 5,
                    'min_samples_split': 15,
                    'n_estimators': 300}
        elif req == "Best Estimators":
            data={'criterion':'mse', 'max_depth':20, 'max_features':'log2',
                      'min_samples_leaf':5, 'min_samples_split':15,
                      'n_estimators':300}
        elif req == "Best Score":
            data={"Best Score": 67.36}
        elif req == "All":
            data={
            "cv":5, 
            "estimator":"RandomForestRegressor()", 
            "n_iter":15,
            "n_jobs":-1,
            "param_distributions": {
            'criterion':['mse', 'friedman_mse'],
            'max_depth': [1, 3, 5, 7, 9, 10, 11, 12,14, 15, 18, 20, 25, 28,30, 33, 38, 40],

            'max_features': ['auto', 'log2', 'sqrt',None],

            'max_leaf_nodes': [None, 10, 20, 30, 40,50, 60, 70, 80, 90],

            'min_samples_leaf': [1, 2, 3, 4, 5, 6,7, 8, 9, 10],

            'min_samples_split': [2, 4, 6, 8, 10,15, 20],

            'n_estimators': [100, 200, 300]},

            "random_state":4, "verbose":2}
        else:
            return {"Error" : "Invalid Input"},400
        return jsonify(data)


api.add_resource(Randomforest, '/v1/model')

if __name__ == "__main__":    
    app.run()
    
