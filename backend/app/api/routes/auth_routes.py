from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Employees, Admin
from app.auth.auth_utils import hash_password, verify_password, create_access_token, get_current_user
from pydantic import BaseModel
from app.utils.logger import log_event
router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    firstname: str
    lastname: str
    email: str
    password: str


@router.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):

    existing = db.query(Employees).filter(Employees.email == data.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")

    user = Employees(
        firstname=data.firstname,
        lastname=data.lastname,
        email=data.email,
        password_hash=hash_password(data.password),
        status="pending"
    )

    db.add(user)
    db.commit()

    return {"message": "Signup successful"}

class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/admin/login")
def admin_login(data: LoginRequest, db: Session = Depends(get_db)):

    admin = db.query(Admin).filter(Admin.email == data.email).first()

    if not admin or not verify_password(data.password, admin.password_hash):
        raise HTTPException(401, "Invalid admin credentials")

    token = create_access_token({
        "user_id": admin.admin_id,
        "role": "admin"
    })
    log_event(
    db,
    actor_type="admin",
    actor_id=admin.admin_id,
    action_type="LOGIN",
    description=f"{admin.firstname} {admin.lastname} logged in"
)

    return {"access_token": token, "role": "admin", "user_id": admin.admin_id, "name": f"{admin.firstname} {admin.lastname}"}

@router.post("/employee/login")
def employee_login(data: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(Employees).filter(Employees.email == data.email).first()

    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    if user.status != "active":
        raise HTTPException(403, "Account not approved")

    token = create_access_token({
        "user_id": user.emp_id,
        "role": "employee"
    })
    log_event(
    db,
    actor_type="employee",
    actor_id=user.emp_id,
    action_type="LOGIN",
    description=f"{user.firstname} {user.lastname} logged in"
)

    return {"access_token": token, "role": "employee", "user_id": user.emp_id, "name": f"{user.firstname} {user.lastname}"}

@router.patch("/admin/approve/{emp_id}")
def approve_employee(emp_id: int, db: Session = Depends(get_db)):

    user = db.get(Employees, emp_id)

    if not user:
        raise HTTPException(404, "User not found")

    user.status = "active"
    db.commit()

    log_event(
    db,
    actor_type="admin",
    actor_id=1,
    action_type="APPROVE_USER",
    description=f"Approved user {user.email}"
)

    return {"message": "Employee approved"}

class LogoutRequest(BaseModel):
    user_id: int
    role: str

@router.post("/logout")
def logout(current_user = Depends(get_current_user), db: Session = Depends(get_db)):

    user_id = current_user["user_id"]
    role = current_user["role"]

    if role == "employee":
        user = db.get(Employees, user_id)
        name = f"{user.firstname} {user.lastname}" if user else "Unknown Employee"
    else:
        admin = db.get(Admin, user_id)
        name = f"{admin.firstname} {admin.lastname}" if admin else "Unknown Admin"

    log_event(
        db,
        actor_type=role,
        actor_id=user_id,
        action_type="LOGOUT",
        description=f"{name} logged out"
    )

    return {"message": "Logged out"}

class ChangePasswordRequest(BaseModel):
    new_password: str


@router.put("/change-password")
def change_password(
    data: ChangePasswordRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = current_user["user_id"]
    role = current_user["role"]

    if role == "employee":
        user = db.get(Employees, user_id)
    else:
        user = db.get(Admin, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = hash_password(data.new_password)
    db.commit()

    return {"message": "Password updated successfully"}
