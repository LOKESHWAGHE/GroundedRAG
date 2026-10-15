from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    version: str = "0.3.0"
