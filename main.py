import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel ,Field,computed_field
from typing import Optional,Annotated
from fastapi.responses import JSONResponse
import joblib

app=FastAPI()
model=joblib.load("model.pkl")
data=joblib.load("processed_dataset.pkl")

class passenger(BaseModel):
    Pclass:int=Field(description='Passenger class',examples=['1','2','3'])
    Sex:str=Field(description='Gender of passenger',examples=['male','female'])
    Age:float=Field(description='Age of passenger',examples=[22,38,26])
    SibSp:int=Field(description='Number of siblings/spouses aboard',examples=[0,1,2])
    Parch:int=Field(description='Number of parents/children aboard',examples=[0,1,2])
    Fare:float=Field(description='Fare paid by passenger',examples=[7.25,71.2833,8.05])
    Cabin:int=Field(description='Do you have  cabin?',examples=[0,1])
    Embarked:str=Field(description='Port of embarkation',examples=['C','Q','S'])

    @computed_field()
    @property
    def family_size(self)->int:
        return 1+ self.SibSp+self.Parch
    
    @computed_field()
    @property
    def is_alone(self)->int:
        return 1 if self.family_size==1 else 0 
@app.post('/predict')
def predict_survial(data:passenger):
    try:
        data=data.model_dump()
        df=pd.DataFrame([data])
        prediction=model.predict(df)
        probability = model.predict_proba(df)[0][1]
        return JSONResponse(status_code=200,content={"prediction": int(prediction[0]), "probability": probability})
    except Exception as e:
        return JSONResponse(status_code=400,content={"error": str(e)})
