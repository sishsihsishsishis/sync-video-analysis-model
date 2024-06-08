import logging
import re
import sys
import boto3
import requests
import time
import redis
import uuid
import os
import shutil
import json
import traceback

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')

# SQS SETUP
aws_access_key_id = 'AKIATRKQV5SYQTJ7DBGF'
aws_secret_access_key = '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'
aws_region = 'us-east-2'

# Initialize SQS client with AWS credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

sqs = session.client('sqs')

# SQS queue URL
on_off_queue_url = 'https://sqs.us-east-2.amazonaws.com/243371732145/sync-main.fifo'
meetings_queue_url = 'https://sqs.us-east-2.amazonaws.com/243371732145/MeetingsQueue.fifo'
cloudfront_base_url = 'https://d2n2ldezfv2tlg.cloudfront.net'

# Receive messages from the SQS queue
def fetch_sqs_messages(group):
    try:
        logging.info("Fetching messages from SQS queue...")
        response = sqs.receive_message(
            QueueUrl=meetings_queue_url,
            MaxNumberOfMessages=10,  # Adjust as needed
            WaitTimeSeconds=20,  # Poll every 20 seconds
            MessageAttributeNames=['All']  # Include all message attributes
        )
        messages = response.get('Messages', [])
        logging.info(f"Found {len(messages)} messages")
        # logging.info(str(messages))
        # Filter messages by the specified group
        filtered_messages = []
        for message in messages:
            body = json.loads(message.get('Body', '{}'))
            print(body)
            if body.get('type') == group:
                filtered_messages.append(message)
        
        if filtered_messages:
            logging.info(f"Found {len(filtered_messages)} messages for group: {group}")
            for message in filtered_messages:
                logging.info(f"Message: {message}")
        else:
            logging.info(f"No messages found for group: {group}")
        
        return filtered_messages
    
    except Exception as e:
        logging.error(f"Error fetching messages: {e}")
        return []

def request_stop_instance(group):
    try:
        # Send a request to stop the instance
        logging.info(f"Requesting stop of {group}")

        message_body = {
            "action": "stop",
            "group": group
        }

        # Send the message to the on_off_queue
        response = sqs.send_message(
            QueueUrl=on_off_queue_url,
            MessageBody=str(message_body),
            MessageGroupId='stop-' + group,  # Ensuring messages are grouped appropriately
            MessageDeduplicationId='stop-' + group + '-' + str(int(time.time()))  # Unique ID to avoid duplication
        )

        logging.info(f"Stop request sent to {on_off_queue_url} with response: {response}")

    except Exception as e:
        logging.error(f"Error requesting stop of instance: {e}")

def delete_message_from_queue(message):
    try:
        # Delete the message from the queue
        sqs.delete_message(
            QueueUrl=meetings_queue_url,
            ReceiptHandle=message['ReceiptHandle']
        )
        logging.info("Message deleted from the queue.")
    except Exception as e:
        logging.error(f"Error deleting message from the queue: {e}")

# RETURN THE S3 INSTANCE
def get_s3_instance():
    # init s3
    s3_instance = boto3.client(service_name='s3', region_name=aws_region, endpoint_url='https://s3.us-east-2.amazonaws.com', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
    return s3_instance

# RETURN THE REDIS CLIENT INSTANCE
def get_redis_instance():
    pool = redis.ConnectionPool(host="18.144.11.243", port=6379, password="123456", db=0)
    redis_instance = redis.StrictRedis(connection_pool=pool)
    return redis_instance

# DOWNLOAD RAW VIDEO FROM S3
def download_resource(resource_path, base_cache_dir, bucket_name='sync5'):
    try:
        logging.info(f"downloading... {resource_path}")
        start = time.time()

        # init cache dir
        if os.path.exists(base_cache_dir):
            if os.path.isdir(base_cache_dir):
                shutil.rmtree(base_cache_dir, ignore_errors=True)
            elif os.path.isfile(base_cache_dir):
                os.remove(base_cache_dir)
        cache_dir = os.path.join(base_cache_dir)

        os.makedirs(cache_dir)

        # Generate a unique file name
        file_name = uuid.uuid4().hex
        cached_path = os.path.join(cache_dir, f'{file_name}.mp4')

        # Construct the full URL to the resource on CloudFront
        url = f"{cloudfront_base_url}/{resource_path}"

        # Download the file from CloudFront
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Write the downloaded content to a local file
        with open(cached_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        finish = time.time()
        logging.info(f"Downloaded in {finish - start}s")
        return file_name

    except: 
        log = traceback.format_exc()
        logging.error(log)

# UPLOAD THE OUTPUT VIDEO TO S3
def upload_resource(filename, key, bucket_name='syneurgy-prod'):
    logging.info('uploading')
    start = time.time()

    # initialize s3 and upload file
    s3_instance = get_s3_instance()
    s3_instance.upload_file(filename, bucket_name, key)

    finish = time.time()
    print('uploaded in ', finish - start, 's')
    return filename

# UPLOAD FILES THROUGH ENDPOINT
def upload_file_to_endpoint(meeting_id, file_path):
    try:
        id_ = re.findall('\d+', meeting_id)[0]
        url = 'http://18.144.11.243:8080/s3/uploadByMeeting/'
        url += id_

        logging.info(f"Uploading file.. {id_} {url}")
        files = {'file': open(file_path, 'rb')}
        response = requests.post(url, files=files)
        if response.ok:
            logging.info(f"File Upload Success :) \n:::: {response.text}")
        else:
            logging.error(f"File Upload Error :) \n:::: {response.status_code} {response.text}")
        logging.info(response.text)
    except:
        log = traceback.format_exc()
        logging.error(log)

# UPLOAD TIMESTAMPS
def upload_timestamps(meeting_id, start, end):
    try: 
        logging.info(f"Uploading timestamps.. {int(start*1000)} {int(end*1000)}")
        data = {
            "model-emotion-detection": [int(start*1000), int(end*1000)]
        }

        id_ = re.findall('\d+', meeting_id)[0]
        url = 'http://18.144.11.243:8080/meeting/analysis-time/' + id_

        logging.info(f"Uploading file to Endpoint.. {id_} {url}")
        response = requests.post(url, json=data)
        if response.ok:
            logging.info(f"Timestamps Upload Success :) \n:::: {response.text}")
        else:
            logging.error(f"Timestamps Upload Error :) \n:::: {response.status_code} {response.text}")
    except:
        log = traceback.format_exc()
        logging.error(log)

# remove double quotes from the string
def remove_quotes(input_string):
    if input_string.startswith('"') and input_string.endswith('"'):
        # If the string starts and ends with double quotes
        # Remove the first and last character (double quotes)
        logging.info("Redis entry has double quotes :(")
        return input_string[1:-1]
    else:
        # If the string does not start and end with double quotes
        return input_string

# INITIALIZE REDIS AS GLOBAL
redis_instance = get_redis_instance()
