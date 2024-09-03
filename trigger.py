from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/ced743e5-2c46-4cb8-957a-83bcc9ee332f/no_trust.mp4",
        "meetingId": "ced743e5-2c46-4cb8-957a-83bcc9ee332f",
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
