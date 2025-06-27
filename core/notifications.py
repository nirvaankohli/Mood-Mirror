from plyer import notification

def notify_user(message: str):
    notification.notify(
        title="Mood Mirror",
        message=message,
        timeout=5
    )
