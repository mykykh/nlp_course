from fastapi import FastAPI
from pydantic import BaseModel
from src.location_predictor import LocationPredictor

app = FastAPI()

uk_location_predictor = LocationPredictor('models/uk_model/', 'cpu', 0.5)


class Input(BaseModel):
    texts: list[str]


@app.post('/extract_locations')
async def get_locations(input: Input):
    return uk_location_predictor.predict(input.texts)
