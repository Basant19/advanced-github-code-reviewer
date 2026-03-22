"""
schemas/repository_schema.py

Repository Pydantic Schemas — P4
---------------------------------
Request and response schemas for repository indexing endpoints.

STATUS: Stub — schemas live in app/api/routes/repos.py for now.
Will be migrated here in P6 when all schemas are consolidated.

Current schemas in app/api/routes/repos.py:
    IndexRepoRequest      — POST /repos/index request body
    IndexRepoResponse     — POST /repos/index response
    IndexStatusResponse   — GET  /repos/{owner}/{repo}/status response
    ClearIndexResponse    — DELETE /repos/{owner}/{repo}/index response

Why This File Exists
--------------------
Consistent with the schema pattern established in P1:
    schemas/review_schema.py     ← P1/P2 review schemas
    schemas/chat_schema.py       ← P1 chat schemas
    schemas/repository_schema.py ← P4 repo/indexing schemas (stub)

P6 Migration Plan
-----------------
Move IndexRepoRequest, IndexRepoResponse, IndexStatusResponse,
and ClearIndexResponse from repos.py into this file.
Update repos.py to import from here.
"""

from app.core.logger import get_logger

logger = get_logger(__name__)
logger.debug("repository_schema: stub loaded — schemas live in app/api/routes/repos.py")