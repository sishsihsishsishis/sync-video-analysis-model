

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
import json
import hashlib
from services.gpt import GptServiceImpl
from services.heatmap import va_heatmap
from services.nlp import word_count, calculate_speaker_time, calculate_speaker_rate_in_chunks, get_pie_and_bar, get_radar_components
from services.pea import get_positive_and_negative

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

table_name='MeetingTable'

def float_to_decimal(f):
    getcontext().traps[Inexact] = False
    getcontext().traps[Rounded] = False
    return Decimal(str(f))


def send_message_to_sqs(object_key, meeting_id, message_type, bucket_name='syneurgy-prod', queue_url=on_off_queue_url):
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
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body),
            MessageGroupId=message_type,
            MessageDeduplicationId=hashlib.md5(f'{message_type}-{meeting_id}'.encode()).hexdigest()
            
        )
        logging.info(response)
        logging.info(f'Message sent to SQS queue.')
    except Exception as e:
        logging.info(f'An error occurred: {e}')


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
        traceback.logging.info_exc()
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
        traceback.logging.info_exc()
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

    # logging.info(result)
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
    logging.info(f'Transcription saved to: {output_path}\n\n\n')


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
        logging.info(f"An error occurred: {e}")
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

        logging.info(f"Found {jpg_count} .jpg files in '{directory}'")
        return (jpg_count, jpg_files)

    except Exception as e:
        logging.info(f"An error occurred: {e}")
        traceback.logging.info_exc()
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
        logging.info(f'uploading {file_path} to {key} in bucket {bucket_name}')
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
    # logging.info(item)
    
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
def save_dialogue_act_labels(dialogue_act_labels, emotion_data, meeting_id, cache_dir='./data', output_filename="dialogue.txt"):
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
    
    # GENERATE MEETING SUMMARY
    gpt_service = GptServiceImpl()
    summary, team_highlights, user_highlights = gpt_service.summary_nlp(mapped_res)
    
    # GET WORD COUNT 
    word_count_data = word_count(mapped_res)
    speaker_time = calculate_speaker_time(mapped_res)
    speaker_rate = calculate_speaker_rate_in_chunks(mapped_res)
    
    word_count_data = {k: {str(ks): Decimal(str(vs)) for ks, vs in v.items()} for k, v in word_count_data.items()}
    speaker_time = {k: Decimal(str(v)) for k, v in speaker_time.items()}
    speaker_rate = {k: {str(ks): Decimal(str(vs)) for ks, vs in v.items()} for k, v in speaker_rate.items()}
    
    # Extract unique speakers from mapped_res
    unique_speakers = list({segment["speaker"] for segment in mapped_res})
    
    # PIE & BAR
    s_keys, s_time, s_rate, e_keys, e_time, e_rate, a_keys, a_time, a_rate, bar_speakers, total, sentences_array = get_pie_and_bar(mapped_res, unique_speakers)
    
    # Initialize empty lists for radar chart data
    r_keys = []
    radar_chart_list = []
    get_radar_components(s_time, total[0], a_time, e_time, sentences_array, radar_chart_list, r_keys, unique_speakers)
    
    # Prepare dimensions for DynamoDB (snake_case)
    raw_dimensions = {key: Decimal(str(value)) for key, value in zip(r_keys, radar_chart_list)}
    dimensions_map = {
        "Absorption or Task Engagement": "absorptionOrTaskEngagement",
        "Enjoyment": "enjoyment",
        "Equal Participation": "equalParticipation",
        "Shared Goal Commitment": "sharedGoalCommitment",
        "Trust and Psychological Safety": "trustAndPsychologicalSafety"
    }

    # Map dimensions to snake_case
    dimensions = {dimensions_map[key]: raw_dimensions[key] for key in raw_dimensions if key in dimensions_map}

    # Initialize DynamoDB resource
    table = dynamodb.Table(table_name)

    logging.info(f"Saving data to DDB... {meeting_id}")
    
    try:
        res = table.update_item(
            Key={'id': meeting_id},
            UpdateExpression="""
                SET 
                    dialogue = :dialogue,
                    summary = :summary,
                    team_highlights = :team_highlights,
                    user_highlights = :user_highlights, 
                    word_count = :word_count_data,
                    speaker_time = :speaker_time,
                    speaker_rate = :speaker_rate,
                    dimensions = :dimensions
            """,
            ExpressionAttributeValues={
                ':dialogue': mapped_res,
                ':summary': summary,
                ':team_highlights': team_highlights,
                ':user_highlights': user_highlights,
                ":word_count_data": word_count_data,
                ":speaker_time": speaker_time,
                ":speaker_rate": speaker_rate,
                ":dimensions": dimensions  # Use snake_case keys
            }
        )
        logging.info(f"dialogue, summary, team_highlights & user_highlights saved to ddb: {res}")
    except Exception as e:
        logging.error(f"Failed to update DynamoDB table for meeting_id {meeting_id}: {e}")
        return None
    
    return unique_speakers


# upload csv files for rppg model 
def float_to_decimal(f):
    return Decimal(str(f))

def rename_headers(df):
    # logging.info(df)
    # logging.info(df.columns[0], " first column")
    # Check if the first header is "Unnamed: 0" and rename it to '_'
    if df.columns[0] == 'Unnamed: 0':
        # logging.info('Renaming the first column to "_"')
        df.columns = ['_'] + df.columns[1:].tolist()
    return df

def process_a_result(df):
    df = rename_headers(df)
    return [
        {
            "_": float_to_decimal(row['_']),
            "a_mean": float_to_decimal(row['a_mean']),
            "a_std": float_to_decimal(row['a_std']),
            "user_0": float_to_decimal(row['user_0']),
            "user_1": float_to_decimal(row['user_1'])
        }
        for index, row in df.iterrows()
    ]

def process_v_result(df):
    df = rename_headers(df)
    return [
        {
            "_": float_to_decimal(row['_']),
            "v_mean": float_to_decimal(row['v_mean']),
            "v_std": float_to_decimal(row['v_std']),
            "user_0": float_to_decimal(row['user_0']),
            "user_1": float_to_decimal(row['user_1'])
        }
        for index, row in df.iterrows()
    ]

def process_rppg_result(df):
    df = rename_headers(df)
    return [
        {
            "_": float_to_decimal(row['_']),
            "rppg_mean": float_to_decimal(row['rppg_mean']),
            "rppg_std": float_to_decimal(row['rppg_std']),
            "user_0": float_to_decimal(row['user_0']),
            "user_1": float_to_decimal(row['user_1'])
        }
        for index, row in df.iterrows()
    ]

def process_anchor_result(df):
    df = rename_headers(df)
    return [
        {
            "_": float_to_decimal(row['_']),
            "user_locs": row['user_locs']
        }
        for index, row in df.iterrows()
    ]

def upload_csv_to_dynamodb(file_path, meeting_id, result_type):
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        # Process the data based on the result_type
        if result_type == 'a_result':
            concat_res = process_a_result(df)
        elif result_type == 'v_result':
            concat_res = process_v_result(df)
        elif result_type == 'rppg_result':
            concat_res = process_rppg_result(df)
        elif result_type == 'anchor_result':
            concat_res = process_anchor_result(df)
        else:
            raise ValueError("Unsupported result type")
        # Initialize DynamoDB resource
        table = dynamodb.Table(table_name)
        
        # Update the item in the DynamoDB table
        table.update_item(
            Key={'id': meeting_id},
            UpdateExpression=f"SET {result_type} = :result",
            ExpressionAttributeValues={f':result': concat_res}
        )
        
        logging.info(f"{result_type} data uploaded successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def upload_participation_emotion(a_results_file_path, v_results_file_path, user_list, meeting_id):
    try:
        # Read the CSV file using pandas
        a_results_raw = pd.read_csv(a_results_file_path)
        v_results_raw = pd.read_csv(v_results_file_path)
        
        #  Convert the data to a array of arrays of floats [[1,2,3],[4,5,6]]
        a_result = a_results_raw.values.tolist()
        v_result = v_results_raw.values.tolist()
        
        posNegRates = get_positive_and_negative(a_result, v_result, user_list, meeting_id)
        
        print("pos_neg_rates_out")
        print(posNegRates)
        # Initialize DynamoDB resource
        table = dynamodb.Table(table_name)
        
        # Update the item in the DynamoDB table
        table.update_item(
            Key={'id': meeting_id},
            UpdateExpression=f"SET posNegRates = :post_neg_rates",
            ExpressionAttributeValues={f':post_neg_rates': posNegRates})
        
        logging.info(f"pos_neg_rates uploaded to DDB successfully.")
        
        return posNegRates
    except Exception as e:
        print(e)
        logging.error(f"An error occurred: {e}")

def convert_to_native_types(data):
    """Convert NumPy types to native Python types."""
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    else:
        return data

def process_heatmap(meeting_id, a_result_path='./data/a_results.csv', v_result_path='./data/v_results.csv'):
    """
    Process heatmap data from the given CSV files and update the results in DynamoDB.

    :param meeting_id: ID of the meeting
    :param a_result_path: Path to the A results CSV file
    :param v_result_path: Path to the V results CSV file
    """
    try:
        logging.info("Processing heatmap...")

        # Read the CSV files
        dfA = pd.read_csv(a_result_path)
        dfV = pd.read_csv(v_result_path)

        # Convert DataFrame to an array of arrays, removing the headers
        dataA = dfA.values.tolist()
        dataV = dfV.values.tolist()

        # Convert each element to a string
        dataA_strings = [[str(item) for item in row] for row in dataA]
        dataV_strings = [[str(item) for item in row] for row in dataV]

        # Print the formatted data (for debugging purposes)
        # print(dataA_strings, dataV_strings)   

        heatmap_result = va_heatmap(meeting_id, dataV_strings, dataA_strings, 100)

        # Convert heatmap_result to native Python types
        heatmap_result = convert_to_native_types(heatmap_result)

        # Access the DynamoDB table
        table = dynamodb.Table(table_name)
        table.update_item(
            Key={'id': meeting_id},
            UpdateExpression="SET heatmap = :result",
            ExpressionAttributeValues={':result': heatmap_result}
        )

        logging.info("Heatmap data uploaded successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
