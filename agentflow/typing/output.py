from pydantic import BaseModel


class SampleOutput(BaseModel):
    text: str


class RefinedOutput(BaseModel):
    text: str


class CountOutput(BaseModel):
    count: int


class DataItem(BaseModel):
    id: str
    data: dict
