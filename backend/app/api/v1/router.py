"""Aggregates every /api/v1/* router into one."""

from fastapi import APIRouter

from app.api.v1 import chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
