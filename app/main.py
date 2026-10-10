from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import auth, chats, documents, health, runs
from app.db.base import Base
from app.db.session import engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Simplest thing that works for the current timeline: create any
    # missing tables against the existing `researchdesk` database. Never
    # drops or alters existing tables/data.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="ResearchDesk API", lifespan=lifespan)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(documents.router)
app.include_router(runs.router)
