
from pydantic import BaseModel
from scalellm import Priority


def jsonify_model(obj: BaseModel):
    return obj.model_dump_json(exclude_unset=True)
  
def to_priority(priority: str) -> Priority:
  if priority == "low":
      return Priority.LOW
  if priority == "normal":
      return Priority.NORMAL
  if priority == "high":
      return Priority.HIGH
  return Priority.DEFAULT