from pydantic import BaseModel, ConfigDict


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chunk_index: int
    page_number: int
    section_title: str | None
    token_count: int
    text: str
