import hashlib, base64, bcrypt
from app.db.database import sessionLocal
from app.db.models import Admin

def _prepare(password):
    digest = hashlib.sha256(password.encode()).digest()
    return base64.b64encode(digest)

def hash_password(password):
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode()

db = sessionLocal()

# Delete existing admin if any
db.query(Admin).filter(Admin.email == "admin@company.com").delete()
db.commit()

admin = Admin(
    firstname="Admin",
    lastname="User",
    email="admin@company.com",
    password_hash=hash_password("admin123")
)

db.add(admin)
db.commit()
print("Admin created successfully")
print("Email: admin@company.com")
print("Password: admin123")
db.close()