"""Request schema for POST /predict, with the client's two rules built in."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from mp074.cleaning import normalise_destination, normalise_service


class PredictRequest(BaseModel):
    distance_km: float = Field(..., ge=0, description="Distance to destination, km")
    weight_kg: float = Field(..., ge=0, description="Parcel weight, kg")
    # Client rule 2: a negative backlog is always a typing mistake.
    hub_backlog: int = Field(..., ge=0, description="Parcels waiting at the hub")
    pin_code_tier: int = Field(..., ge=1, le=3, description="Delivery-area tier")
    # Client rule 1: above 35000 rupees is impossible for us.
    declared_value_rs: float = Field(..., ge=0, le=35000, description="Declared value, rupees")
    transfers: int = Field(..., ge=0, description="Hub-to-hub transfers")
    service: str = Field(..., description="standard, express or economy (spelling is tidied)")
    destination: str = Field(..., description="Metro, Tier-2 or Rural")

    @field_validator("service")
    @classmethod
    def _service_known(cls, value: str) -> str:
        return normalise_service(value)

    @field_validator("destination")
    @classmethod
    def _destination_known(cls, value: str) -> str:
        return normalise_destination(value)


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted delivery_days")
