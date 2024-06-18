# import redis
import subprocess
import os
import time
import logging
import json
import shutil
import base  # basic functions
import helpers  # helper functions
import rppg  # model
from speaker import diarization
from emotion import go_emotion
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    no_message_count = 0
    max_no_message_count = 5
    group = 'analysis'  # Specify the appropriate group for stopping the instance

    while True:
        try:
            logging.info("Fetching messages from SQS...")
            messages = base.fetch_sqs_messages(group)

            if messages:
                no_message_count = 0  # Reset the counter when messages are received
                cache_dir = "./data"
                asr_path = "./data/transcription.txt"
                diarization_path = "./data/diarization.txt"
                # Process received messages
                for message in messages:
                    try:
                        logging.info('Received message: %s', message['Body'])
                        
                        # Reset cache
                        logging.info("Resetting cache...")
                        if os.path.exists(cache_dir):
                            shutil.rmtree(cache_dir)
                        
                        # Extract details from SQS event
                        body = json.loads(message.get('Body', '{}'))
                        download_url = body.get('key')
                        meeting_id = body.get('meetingId')
                        
                        # Download from s3
                        logging.info(f"Downloading resource from S3: {download_url}")
                        local_cached_filename = base.download_video(download_url, './raw')
                        video_path = f"./raw/{local_cached_filename}.mp4"
                        
                        # Delete the message from the queue
                        base.delete_message_from_queue(message)
                    
                        process_start_time = time.time()
                        
                        # Change FPS
                        logging.info(f"Resampling video: {local_cached_filename}")
                        resample_path = f"./data/{local_cached_filename}.mp4"
                        helpers.change_fps(video_path, resample_path, 25)
                        
                        # Run rppg model
                        logging.info(f"Running rppg model: {local_cached_filename}")
                        rppg.go_rppg(video_path, cache_dir, 0.7)
                        
                        # Find the number of speakers in the video
                        logging.info(f"Counting speakers in the video...")
                        speaker_count, jpg_files = helpers.get_speaker_count('data')
                        logging.info(f"Speaker count: {speaker_count}")
                        
                        # Upload rppg files
                        logging.info(f"Uploading detected avatars...")
                        helpers.upload_detected_avatars(jpg_files, meeting_id)
                        
                        # Extract audio
                        logging.info(f"Extracting audio...")
                        wav_file = helpers.extract_audio(resample_path)
                        
                        # Run NLP model
                        logging.info(f"Generating transcript...")
                        transcription_segments = helpers.asr_transcribe(wav_file)
                        
                        logging.info(f'Saving transcript to file...')
                        helpers.save_transcription_to_file(transcription_segments, asr_path)
                        
                        # Run diarization model
                        logging.info(f"Running diarization model...")
                        speaker_chunks = diarization(wav_file, int(speaker_count))
                        
                        logging.info(f"Matching speakers with ASR and combining speaker chunks...")
                        result_path = helpers.match_speakers_asr(speaker_chunks, meeting_id)
                        
                        # Upload NLP text data
                        logging.info(f"Uploading NLP text data to S3...")
                        s3_nlp_upload_path = f'out/{meeting_id}/transcript.txt'
                        base.upload_resource(diarization_path, s3_nlp_upload_path)
                        helpers.send_message_to_sqs(download_url, meeting_id, 'speaker')
                        
                        # Run emotion model
                        logging.info(f"Running emotion model...")
                        diarization_result, emotion_data = helpers.process_diarization_and_emotion(diarization_path)
                        emotion_labels = go_emotion(emotion_data)
                        emotion_path = helpers.save_emotion_results(emotion_labels, diarization_result, meeting_id)

                    except Exception as e:
                        logging.error(f"Error processing message: {e}")
                        traceback.print_exc()
                    
            else:
                no_message_count += 1  # Increment the counter if no messages are received
                if no_message_count >= max_no_message_count:
                    logging.info(f"Requesting stop for the instance in group: {group}")
                    base.request_stop_instance(group)
                    no_message_count = 0  # Reset the counter after requesting stop
            
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            traceback.print_exc()
        logging.info(f"Message Count: {no_message_count}")
        time.sleep(60)  # Wait for 20 seconds before polling again
