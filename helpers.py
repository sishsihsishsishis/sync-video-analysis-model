

import numpy as np
import speaker
from base import dynamodb, s3, upload_resource, sqs, on_off_queue_url
from decimal import Decimal
from decimal import Decimal, getcontext, Inexact, Rounded
import re

#============================================================
import cv2 
import os 
import traceback
import logging
import shutil
import pandas as pd
import torch
import whisper
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

table_name='MeetingTable'

def float_to_decimal(f):
    getcontext().traps[Inexact] = False
    getcontext().traps[Rounded] = False
    return Decimal(str(f))


def send_message_to_sqs(object_key, meeting_id, message_type, bucket_name = 'syneurgy-prod'):
    """
    Send a message to an SQS queue.

    :param bucket_name: Name of the S3 bucket
    :param object_key: Key of the S3 object
    :param meeting_id: ID of the meeting
    :param message_type: Type of the message
    :param queue_url: URL of the SQS queue
    """
    try:
        logging.info(f"sending message to sqs...")
        # Construct the message body
        message_body = {
            "bucket": bucket_name,
            "key": object_key,
            "meetingId": meeting_id,
            "type": message_type,
        }

        # Send the message to the SQS queue
        response = sqs.send_message(
            QueueUrl=on_off_queue_url,
            MessageBody=json.dumps(message_body),
            MessageGroupId=message_type,
            MessageDeduplicationId=f'{message_type}-{meeting_id}'
        )
        print(response)
        logging.info(f'Message sent to SQS queue.')
    except Exception as e:
        print(f'An error occurred: {e}')


def change_fps(source_video_path, output_file_path, target_fps=25):
    try:
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            logging.info(f"Created data directory: {data_folder}")
        
        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {source_video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps is None or original_fps == 0:
            raise ValueError(f"Unable to determine FPS for video file: {source_video_path}")

        logging.info(f"Original FPS: {original_fps}")
        
        if original_fps == target_fps:
            logging.info("Target FPS is the same as the original FPS. Copying the video to the data folder.")
            output_data_path = os.path.join(data_folder, os.path.basename(output_file_path))
            shutil.copy(source_video_path, output_data_path)
            logging.info(f"Video copied to: {output_data_path}")
            cap.release()
            return output_data_path
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Frame dimensions: {frame_width}x{frame_height}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, target_fps, (frame_width, frame_height))
        logging.info(f"Output file path: {output_file_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        logging.info(f"Video with changed FPS saved to: {output_file_path}")

        return output_file_path
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        return None

def extract_audio(video_path, audio_path=None):
    """
    Extracts audio from a video file.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str, optional): Path to save the extracted audio. If not provided, it will be saved
            in the same directory as the video file with a ".wav" extension.

    Returns:
        str: Path to the extracted audio file.
    """
    try:
        if audio_path is None:
            audio_path = os.path.splitext(video_path)[0] + ".wav"

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
        
        logging.info(f"Extracting audio from '{video_path}' to '{audio_path}'")

        # Load the video file
        video_clip = VideoFileClip(video_path)
        
        # Extract and write the audio
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le')

        logging.info(f"Audio successfully extracted to '{audio_path}'")
        return audio_path

    except Exception as e:
        logging.error(f"An error occurred while extracting audio: {e}")
        traceback.print_exc()
        return None


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def asr_transcribe(wav_file):
    """
    Transcribes speech from a WAV file using the Whisper ASR model.

    Args:
        wav_file (str): Path to the input WAV audio file.

    Returns:
        list: List of transcribed segments.
    """
    # Check if CUDA is available, set device accordingly
    device = torch.device(0) if torch.cuda.is_available() else "cpu"

    # Load Whisper ASR model
    WHISPER_MODEL = whisper.load_model("large", in_memory=True, device=device)

    # Transcribe using Whisper model
    result = WHISPER_MODEL.transcribe(wav_file)

    # print(result)
    return result["segments"]

def save_transcription_to_file(transcript_segments, output_path):
    """
    Save the transcribed segments to a CSV file.

    Args:
        transcript_segments (list): List of transcribed segments.
        output_path (str): Path to the output CSV file.
    """
    # Convert the list of segments to a DataFrame and write to CSV
    pd.DataFrame(transcript_segments).to_csv(output_path, index=False, sep="\t")
    print(f'Transcription saved to: {output_path}\n\n\n')


def match_speakers_asr(speaker_chunks, meeting_id, asr_path="./data/transcription.txt", output_dir='./data/'):
    """
    Match speakers with ASR results and save the concatenated results to a file.

    :param speaker_chunks: List of speaker chunks with time information
    :param asr_path: Path to the ASR results file
    :param output_dir: Directory to save the concatenated results file
    :param meeting_id: Meeting ID to be added to each segment
    :return: Path to the saved concatenated results file
    """
    try:
        # Concatenate speaker time chunks and IDs
        speaker_time_chunks, speaker_id_chunks = speaker.concat_speaker_chunks(speaker_chunks)
        
        concat_res = []

        # Read ASR chunks
        asr_chunks = pd.read_csv(asr_path, index_col=None, sep="\t")

        # Match speakers and ASR results
        time_id = 0
        for asr_chunk in asr_chunks.itertuples(index=False):
            asr_text = asr_chunk.text
            asr_start = float(asr_chunk.start)
            asr_end = float(asr_chunk.end)
            mid_time = (asr_start + asr_end) / 2

            for i in range(time_id, len(speaker_time_chunks)):
                speaker_start, speaker_end = speaker_time_chunks[i]
                if mid_time < speaker_start:
                    time_id = 0
                elif mid_time > speaker_end:
                    time_id = -1
                else:
                    time_id = i
                    break
            concat_res.append(
                [speaker_id_chunks[time_id], asr_start, asr_end, asr_text.lower()]
            )

        # Save the concatenated results
        concat_res_path = os.path.join(output_dir, "diarization.txt")
        pd.DataFrame(
            np.array(concat_res),
            columns=["Speaker", "Start", "End", "Sentence"]
        ).to_csv(concat_res_path, index=False, sep="\t")

        # Initialize DynamoDB resource
        table = dynamodb.Table(table_name)

        mapped_res = [
            {
                "speaker": segment[0],
                "start": float_to_decimal(segment[1]),
                "end": float_to_decimal(segment[2]),
                "sentence": segment[3],
            }
            for segment in concat_res
        ]

        # Add meeting_id to each segment and batch write to DynamoDB
        item = {
            'id': meeting_id, 
            'transcript': mapped_res
        }
        
        table.update_item(
            Key={'id': meeting_id},
            UpdateExpression="SET transcript = :transcript",
            ExpressionAttributeValues={':transcript': mapped_res}
        )
        return concat_res_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_speaker_count(directory='data'):
    """
    Counts the number of .jpg files in the specified directory.

    Args:
        directory (str): Path to the directory to check for .jpg files. Defaults to 'data'.

    Returns:
        int: Number of .jpg files found in the directory.
    """
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg')]
        jpg_count = len(jpg_files)

        print(f"Found {jpg_count} .jpg files in '{directory}'")
        return (jpg_count, jpg_files)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return 0

def pad_filename(filename):
    # Regular expression to match the pattern 'user_<number>.jpg'
    match = re.match(r'(user_)(\d+)(\.jpg)', filename)
    if match:
        prefix, number, suffix = match.groups()
        padded_number = f'{int(number):02}'  # Pad the number with leading zeros to ensure at least 2 digits
        return f'{prefix}{padded_number}{suffix}'
    return filename

def upload_detected_avatars(jpg_files, meeting_id, bucket_name='syneurgy-prod'):
    # Create the file-key mapping
    file_key_tuples = [
        (f'./data/{filename}', f'out/{meeting_id}/{pad_filename(filename)}')
        for filename in jpg_files
    ]

    # Upload each file using the base.upload_resource function
    for file_path, key in file_key_tuples:
        print(f'uploading {file_path} to {key} in bucket {bucket_name}')
        upload_resource(file_path, key, bucket_name)
    
    return file_key_tuples


# process diarization result to prepare enotion data
def process_diarization_and_emotion(diarization_path):
    """
    Reads the diarization results from a file and prepares emotion data.
    
    Parameters:
    diarization_path (str): Path to the diarization result file.
    
    Returns:
    tuple: A tuple containing:
        - diarization_result (pd.DataFrame): The DataFrame of diarization results.
        - emotion_data (list): A list of lists where each sublist contains the sentence (lowercased) and the speaker.
    """
    # Read the diarization result file
    diarization_result = pd.read_csv(diarization_path, index_col=None, sep="\t")
    
    # Prepare emotion data
    emotion_data = []
    for _emotion_data in diarization_result.itertuples(index=False):
        emotion_data.append([str(_emotion_data.Sentence).lower(), _emotion_data.Speaker])  # type: ignore
    
    return diarization_result, emotion_data


# save emotion result to csv
def save_emotion_results(emotion_labels, speaker_diarization, meeting_id, output_dir='./data', output_filename="emotion.txt"):
    """
    Save the emotion results combined with speaker diarization data to a text file.
    
    Parameters:
    emotion_labels (list): A list of emotion labels.
    speaker_diarization (pd.DataFrame): A DataFrame containing speaker diarization data.
    output_dir (str): Directory where the output file will be saved. Default is './data'.
    output_filename (str): The name of the output file. Default is 'emotion.txt'.
    
    Returns:
    str: The path to the saved file.
    """
    emotion_res = []
    for emotion_label, speaker_d in zip(emotion_labels, speaker_diarization.values):
        speaker_d = speaker_d.tolist()
        speaker_d.append(emotion_label)
        emotion_res.append(speaker_d)
    
    headers = speaker_diarization.columns.tolist()
    headers.append("Emotion")  # type: ignore
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    emotion_res_path = os.path.join(output_dir, output_filename)
    pd.DataFrame(np.array(emotion_res), columns=headers).to_csv(emotion_res_path, index=False, sep="\t")
    
    mapped_res = [
        {
            "speaker": segment[0],
            "start": float_to_decimal(segment[1]),
            "end": float_to_decimal(segment[2]),
            "sentence": segment[3],
            "emotion": segment[4],
        }
        for segment in emotion_res
    ]
    # Initialize DynamoDB resource
    table = dynamodb.Table(table_name)

    item = {
        'id': meeting_id, 
        'emotionText': mapped_res
    }
    # print(item)
    
    logging.info(f"Saving emotion to DDB... {meeting_id}")
    table.update_item(
        Key={'id': meeting_id},
        UpdateExpression="SET emotionText = :emotionText",
        ExpressionAttributeValues={':emotionText': mapped_res}
    )
    
    return emotion_res_path

# load emotion data for dialogue processing
def load_emotion_data(data_path='./data/emotion.txt'):
    """
    Load emotion data from a text file into a pandas DataFrame.
    
    Parameters:
    data_path (str): Path to the text file containing emotion data.
    
    Returns:
    pd.DataFrame: DataFrame containing loaded emotion data.
    """
    try:
        emotion_data = pd.read_csv(data_path, index_col=None, sep="\t")
        
        text = []
        for _emotion_data in emotion_data.itertuples(index=False):
            text.append(str(_emotion_data.Sentence).lower())  # type: ignore
        
        logging.info(f"Emotion data loaded successfully from {data_path}.")
        return emotion_data, text
    
    except Exception as e:
        logging.error(f"Failed to load emotion data from {data_path}: {str(e)}")
        return None
    
# save dialogue data to csv & DDB
def save_dialogue_act_labels(dialogue_act_labels, emotion_data, meeting_id, cache_dir = './data', output_filename="dialogue.txt"):
    """
    Save the dialogue act labels combined with emotion text data to a text file.
    
    Parameters:
    dialogue_act_labels (list): A list of dialogue act labels.
    emotion_data (pd.DataFrame): A DataFrame containing emotion text data.
    cache_dir (str): Directory where the output file will be saved.
    output_filename (str): The name of the output file. Default is 'dialogue.txt'.
    
    Returns:
    str: The path to the saved file.
    """
    nlp_res = []
    for dialogue_act_label, _e_text in zip(dialogue_act_labels, emotion_data.values):
        _e_text = _e_text.tolist()
        _e_text.append(dialogue_act_label)
        nlp_res.append(_e_text)

    headers = emotion_data.columns.tolist()
    headers.append("DialogueAct")  # type: ignore

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    res_path = os.path.join(cache_dir, output_filename)
    pd.DataFrame(np.array(nlp_res), columns=headers).to_csv(res_path, index=False, sep="\t")
    
    mapped_res = [
        {
            "speaker": segment[0],
            "start": float_to_decimal(segment[1]),
            "end": float_to_decimal(segment[2]),
            "sentence": segment[3],
            "emotion": segment[4],
            "dialogue": segment[5],
        }
        for segment in nlp_res
    ]
    # Initialize DynamoDB resource
    table = dynamodb.Table(table_name)

    item = {
        'id': meeting_id, 
        'dialogue': mapped_res
    }
    # print(item)
    
    logging.info(f"Saving dialogue to DDB... {meeting_id}")
    res = table.update_item(
        Key={'id': meeting_id},
        UpdateExpression="SET dialogue = :dialogue",
        ExpressionAttributeValues={':dialogue': mapped_res}
    )
    
    print(res)
    
    return res_path
