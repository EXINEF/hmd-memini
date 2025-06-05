from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    task_name: str
    due_date: Optional[str] = None  # Natural language date for display
    normalized_due_date: Optional[str] = None  # YYYY-MM-DD format for calculations
    completed: bool = False
