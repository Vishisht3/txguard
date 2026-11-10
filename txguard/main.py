from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from txguard.api.routes import router
from txguard.config import STATIC_DIR


def create_app() -> FastAPI:
    application = FastAPI(
        title="TxGuard × PayPal AI Risk Manager & Agentic Commerce API",
        version="2.0.0",
        description=(
            "Real-Time Two-Tier Cascading Fraud Gate & Gemini Agentic Commerce Platform "
            "built on the PayPal Developer Platform (Orders v2, Authorizations v2, Webhooks v1) "
            "and PyTorch Residual Neural Networks."
        ),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    application.include_router(router)
    return application


app = create_app()

