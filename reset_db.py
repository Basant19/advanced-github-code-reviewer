#E:\advanced-github-code-reviewer\reset_db.py
import sys
import asyncio
from app.db.base import Base
from app.db.session import engine  # This is your AsyncEngine
from app.db.models import * # noqa — ensures all models are registered
from app.core.logger import get_logger

logger = get_logger(__name__)

async def reset_logic():
    async with engine.begin() as conn:
        logger.info("Dropping all tables...")
        # run_sync is required to bridge the async connection 
        # to the sync metadata methods
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("All tables dropped successfully.")

        logger.info("Recreating all tables from models...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("All tables recreated successfully.")

async def reset_task():
    try:
        await reset_logic()
        logger.info("Database reset complete — ready to use.")
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        # Log full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    confirm = input("This will DELETE ALL DATA. Type 'yes' to continue: ")
    if confirm.strip().lower() == "yes":
        # Initialize the async event loop
        asyncio.run(reset_task())
    else:
        print("Aborted.")
        sys.exit(0)