from typing import Optional
from urllib.request import BaseHandler

import joblib
from pydantic import BaseModel, Field
from pydiator_core.interfaces import BaseRequest, BaseResponse


class Dataset:
    resources: [str] = ['sapphire_jeans.csv']

dataset = Dataset()

class ProductType:
    new_cloth: int = 0
    old_cloth: int = 1
    hourglass: int = 2


class StandardMeasure:
    def __init__(self, waist, hips):
        self.waist = waist
        self.hips = hips


jeans_sizes = ["0-6", "0-8", "0-10", "2-6", "2-8", "2-10",
               "2-12", "4-10", "4-12", "4-14", "6-14", "8-14",
               "10-16", "10-18", "12-16", "12-18", "16-20"]

product_type = ProductType()


##################SizeRequest##########

class SizeRequest(BaseModel, BaseResponse):
    waist: float = Field("waist", title="A")
    hips: float = Field("hips", title="B")
    type: int = Field("type")
    measure_type: str = Optional[str]


class SizeResponse(BaseModel, BaseResponse):
    size: str = None
    pass


def standard_measure(request: SizeRequest) -> StandardMeasure:
    multiplier_c = 2.54
    if request.type == 'cm':
        multiplier_c = 1
    return StandardMeasure(request.waist * multiplier_c, request.hips * multiplier_c)


class PredictSizeDataResponse(BaseModel, BaseRequest):
    size: str = None


class Prediction(BaseHandler):
    async def handle(self, req: SizeRequest) -> PredictSizeDataResponse:
        offset = 20.5

        size = standard_measure(req)
        if req.type == product_type.hourglass:
            offset = 0
        dataset_index = req.type
        loaded_model = joblib.load(dataset.job[dataset_index])

        svm_algorithm = loaded_model.predict([[size.waist, size.hips + offset]])

        if req.type == product_type.hourglass:
            return PredictSizeDataResponse(size=svm_algorithm[0])

        index = svm_algorithm[0] - 1
        return PredictSizeDataResponse(size=jeans_sizes[index])


