from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings

# Alembic Config object (reads alembic.ini)
config = context.config

# Inject the DB URL from our typed settings (env-overridable in prod).
# Force the psycopg 3 driver for SQLAlchemy (app code uses plain psycopg,
# so the '+psycopg' suffix lives here, not in the settings URL).
db_url = get_settings().database_url.replace(
    "postgresql://", "postgresql+psycopg://", 1
)
config.set_main_option("sqlalchemy.url", db_url)

# Set up loggers from the ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No ORM models: migrations are hand-written raw SQL, so no autogenerate target
target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection ('--sql' mode)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()