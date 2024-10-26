from typing import Annotated

from fastapi import APIRouter, Body

router = APIRouter()

@router.post("/recognize_group")
def recognize_group(query: Annotated[str, Body()]) -> str:
    ...