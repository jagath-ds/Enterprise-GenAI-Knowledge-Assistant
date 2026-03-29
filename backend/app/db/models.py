from .database import Base
from sqlalchemy import  Integer, String, Text,func,DateTime
from sqlalchemy.dialects.postgresql import UUID,INET
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional
import uuid

class Document(Base):
    __tablename__ = "documents"
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(20), server_default="uploaded",nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, server_default="0",nullable=False)
    uploaded_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    indexed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class Employees(Base):
    __tablename__="employees"

    emp_id : Mapped[int] = mapped_column(Integer, primary_key=True,autoincrement=True)
    firstname: Mapped[str] = mapped_column(String(50), nullable=False)
    lastname: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255),nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    department: Mapped[str] = mapped_column(String(50), nullable=True)
    designation: Mapped[str] = mapped_column(String(50), nullable=True)

    
class Admin(Base):
    __tablename__="admins"

    admin_id : Mapped[int] = mapped_column(Integer, primary_key=True,autoincrement=True)
    firstname: Mapped[str] = mapped_column(String(50), nullable=False)
    lastname: Mapped[str] = mapped_column(String(50), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255),nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    
    
class SystemLog(Base):
    __tablename__ = "system_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_type: Mapped[str] = mapped_column(String(20), nullable=False)  
    actor_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action_description: Mapped[str] = mapped_column(Text, nullable=True)
    affected_table: Mapped[str] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True),server_default=func.now())
    ip_address = mapped_column(INET, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=True)