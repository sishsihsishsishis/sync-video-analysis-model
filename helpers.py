import ffmpeg
import logging
import traceback
import os
import torch
import whisper
import ffmpeg
import pandas as pd

def change_fps(source_video_path, output_file_path, target_fps=25):
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
        
    streams = ffmpeg.probe(source_video_path)["streams"]
    fps = None
    for stream in streams:
        fractions = stream.get("r_frame_rate", "*/0").split("/")
        if float(fractions[1]) == 0:
            continue
        else:
            fps = float(fractions[0]) / float(fractions[1])
            break

    if fps is None:
        return None

    if fps != target_fps:
        source_video = ffmpeg.input(source_video_path)
        out = source_video.output(output_file_path, r=target_fps)
        out.global_args("-loglevel", "warning").run(overwrite_output=True)
        return output_file_path
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
    # If audio_path is not provided, generate it based on the video_path
    if audio_path is None:
        audio_path = video_path[:-3] + "wav"

    # Input stream from video file
    stream = ffmpeg.input(video_path)

    # Output stream to extract audio
    stream = ffmpeg.output(stream, audio_path)

    # Run the ffmpeg command
    ffmpeg.run(stream, overwrite_output=True)

    return audio_path


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

    print(result)
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
