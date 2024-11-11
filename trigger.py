from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/cf57bf9d-3db6-4efe-adff-11dec9294394/no_trust.mp4",
        "meetingId": "cf57bf9d-3db6-4efe-adff-11dec9294394",
        "type": "analysis"
    }

    # message = {
    #     "bucket": "syneurgy-prod",
    #     "key": "meetings/e5a1d664-2c0e-47aa-9459-260a7eedcc38/Syuneurgy stand up - 16 July 2024.mp4",
    #     "meetingId": "e5a1d664-2c0e-47aa-9459-260a7eedcc38",
    #     "type": "analysis"
    # }

    # message = {
    #     "bucket": "syneurgy-prod",
    #     "key": "meetings/b9ff8c4f-a5a2-46f6-9d32-220b10247fef/Syneurgy - Saturday Review call - Active Speaker Detection _ Platform _ Pdf Report _ Mirror Site - 13 July 2024.mp4",
    #     "meetingId": "b9ff8c4f-a5a2-46f6-9d32-220b10247fef",
    #     "type": "analysis"
    # }

    # Call the function with the provided parameters
    send_message_to_sqs(
        object_key=message['key'],
        meeting_id=message['meetingId'],
        message_type=message['type'],
        bucket_name=message['bucket'],
        queue_url=meetings_queue_url
    )
