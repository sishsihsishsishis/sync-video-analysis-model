from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/e04e044c-278e-4da6-ac7c-b4cbdf75bd6e/gitlab2_meeting_video_3.mp4",
        "meetingId": "e04e044c-278e-4da6-ac7c-b4cbdf75bd6e",
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
