from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Employees, Admin, SystemLog
from app.auth.auth_utils import hash_password, verify_password, create_access_token
from pydantic import BaseModel

from app.utils.logger import log_event

router = APIRouter(prefix="/admin", tags=["admin"])
@router.get("/employees")
def get_all_employees(db: Session = Depends(get_db)):
    users = db.query(Employees).all()

    return [
        {
            "id": user.emp_id,
            "name": user.firstname + " " + user.lastname,
            "email": user.email,
            "status": user.status
        }
        for user in users
    ]
@router.patch("/employees/{emp_id}/status")
def update_status(emp_id: int, status: str, db: Session = Depends(get_db)):
    user = db.get(Employees, emp_id)

    if not user:
        raise HTTPException(404, "User not found")

    user.status = status
    db.commit()

    return {"message": "Status updated"}

@router.delete("/employees/{emp_id}")
def delete_employee(emp_id: int, db: Session = Depends(get_db)):
    user = db.get(Employees, emp_id)

    if not user:
        raise HTTPException(404, "User not found")
    log_event(
    db,
    actor_type="admin",
    actor_id=1,
    action_type="DELETE_USER",
    description=f"Deleted user {user.email}"
)
    db.delete(user)
    db.commit()


    return {"message": "User deleted"}

@router.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(SystemLog).order_by(SystemLog.timestamp.desc()).all()

    return [
        {
            "id": log.log_id,
            "actor": f"{log.actor_type} ({log.actor_id})",
            "action": log.action_type,
            "description": log.action_description,
            "time": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "status": log.status
        }
        for log in logs
    ]