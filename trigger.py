from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/9e39be7b-1bbe-4673-bb3c-aa6bdd48caa5/trust.mp4",
        "meetingId": "9e39be7b-1bbe-4673-bb3c-aa6bdd48caa5",
        "type": "analysis"
    }

    # Call the function with the provided parameters
    send_message_to_sqs(
        object_key=message['key'],
        meeting_id=message['meetingId'],
        message_type=message['type'],
        bucket_name=message['bucket'],
        queue_url=meetings_queue_url
    )
