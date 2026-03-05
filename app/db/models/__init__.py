"""
Database Models Package

This module imports all ORM models so they are registered
with SQLAlchemy metadata.

When Base.metadata.create_all() runs, SQLAlchemy will
discover all tables from these imports.
"""

from app.db.models.repository import Repository