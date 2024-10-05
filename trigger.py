from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/3bd3606f-78b1-4892-a608-fd04c80ea98b/Syneurgy - X Team Tuesday - platform and readiness timeline - 20 August 2024.mp4",
        "meetingId": "3bd3606f-78b1-4892-a608-fd04c80ea98b",
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
