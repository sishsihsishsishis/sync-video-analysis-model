# import redis
import base
import subprocess
import time
import logging
import time
import json
import helpers
import rppg

if __name__ == '__main__':
    no_message_count = 0
    max_no_message_count = 5
    group = 'analysis'  # Specify the appropriate group for stopping the instance

    while True:
        try:
            messages = base.fetch_sqs_messages(group)

            if messages:
                no_message_count = 0  # Reset the counter when messages are received

                # Process received messages
                for message in messages:
                    # Do something with the message
                    print('Received message:', message['Body'])
                    body = json.loads(message.get('Body', '{}'))
                    
                    download_url = body.get('key')
                    meeting_id = body.get('meetingId')
                    # Download from s3
                    local_cached_filename = base.download_resource(download_url, './raw')
                    # Delete the message from the queue
                    # base.delete_message_from_queue(message)
                    
                    process_start_time = time.time()
                    # start the talknet subprocess
                    logging.info(f"resampling... {local_cached_filename}")
                    
                    cached_path = "./raw/" + local_cached_filename + ".mp4"
                    resample_path = "./data/" + local_cached_filename + ".mp4"
                    
                    # change fps 
                    helpers.change_fps(cached_path, resample_path, 25)
                    
                    logging.info(f"extracting audio...")
                    wav_file = helpers.extract_audio(resample_path)
                    
                    logging.info(f"generating transcript...")
                    transcription_segments = helpers.asr_transcribe(wav_file)
                    
                    helpers.save_transcription_to_file(transcription_segments, './data/transcription.txt')
                    rppg.gorppg(cached_path, "./data/" + local_cached_filename, 0.7)

            else:
                no_message_count += 1  # Increment the counter if no messages are received

                if no_message_count >= max_no_message_count:
                    # base.request_stop_instance(group)
                    no_message_count = 0  # Reset the counter after requesting stop
            
        except Exception as e:
            logging.error(f"Error: {e}")
        logging.info(f"Message Count: {no_message_count}")
        time.sleep(20)  # Wait for 20 seconds before polling again
        