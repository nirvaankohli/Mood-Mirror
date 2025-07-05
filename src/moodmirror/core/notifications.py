from plyer import notification

# notifications.py

class Notification(Exception):

    def __init__(self):
        pass
        

    def notify_user(message: str):

        notification.notify(
            title="Mood Mirror",
            message=message,
            timeout=5
        )
