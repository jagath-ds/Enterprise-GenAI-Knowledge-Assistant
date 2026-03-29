from app.db.models import SystemLog

def log_event(db, actor_type, actor_id, action_type, description, status="success"):
    log = SystemLog(
        actor_type=actor_type,
        actor_id=actor_id,
        action_type=action_type,
        action_description=description,
        status=status
    )

    db.add(log)
    db.commit()