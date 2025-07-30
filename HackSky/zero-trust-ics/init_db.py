# init_db.py (in root folder)

from sqlalchemy import create_engine
from gateway.models import Base

db_path = "logs/ics_data.db"
engine = create_engine(f"sqlite:///{db_path}")

Base.metadata.drop_all(engine)  # 💣 Drop old tables
Base.metadata.create_all(engine)  # ✅ Create new tables
print("✅ Database initialized with new schema.")
