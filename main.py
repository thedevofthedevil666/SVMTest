import datetime

import pandas
import pandas as pd
import simplejson
from fastapi import FastAPI
from pydiator_core.mediatr import pydiator
from starlette import status

from controllers.sizecontroller import SizeRequest

app = FastAPI()


@app.get("/")
def hello():
    return {"Bienvenido a la api de pruebas"}


@app.get("/read-dataset")
def dataset():
    import pandas as pd
    with open('sapphire_jeans.csv') as fp:
        headers = fp.readline().strip().split(',')[:-1]
        df = pd.read_csv(fp, header=None, names=headers, dtype=str)
        df.drop(df[df[headers[1:]].isna().all(axis=1)].index, inplace=True)
        return df.to_string()


@app.get("/predict/")
async def prediction_model(req: SizeRequest):
    return await pydiator.send(req=req)


