from decimal import Decimal

# Helper function to calculate time slot
def calculate_time_slot(start_time, end_time, chunk_size=30):
    # Calculate the average time and map it to a 30-second time slot
    return int(((start_time + end_time) / 2) // chunk_size * chunk_size)

def word_count(nlp_data, chunk_size=30):
    # Create a dictionary to store the word count for each speaker and time slot
    word_count = {}

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        sentence = entry['sentence']

        # Use the helper function to calculate the time slot
        time_slot = calculate_time_slot(start_time, end_time, chunk_size)

        # Split the sentence into words
        words = sentence.split()

        # Get or create the dictionary for the current speaker
        if speaker not in word_count:
            word_count[speaker] = {}

        # Update the word count for the speaker in the given time slot
        if time_slot in word_count[speaker]:
            word_count[speaker][time_slot] += Decimal(len(words))
        else:
            word_count[speaker][time_slot] = Decimal(len(words))

    return word_count

def calculate_speaker_time(nlp_data):
    # Create a dictionary to store the total speaking time for each speaker
    speaker_time = {}

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        
        # Calculate the duration for the current entry
        duration = end_time - start_time

        # Add the duration to the speaker's total time
        if speaker in speaker_time:
            speaker_time[speaker] += Decimal(str(duration))
        else:
            speaker_time[speaker] = Decimal(str(duration))

    return speaker_time

def calculate_speaker_rate_in_chunks(nlp_data, chunk_size=30):
    # Create a dictionary to store the word rates for each speaker in each time slot
    word_rates = {}

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        sentence = entry['sentence']
        
        # Split the sentence into words
        words = sentence.split()
        num_words = len(words)
        
        # Process the dialogue in 30-second chunks
        current_time = start_time
        while current_time < end_time:
            next_chunk_end = min(current_time + chunk_size, end_time)
            duration = next_chunk_end - current_time
            
            if duration > 0:  # Avoid division by zero
                # Calculate the proportional number of words for the current chunk
                chunk_word_count = int(num_words * (duration / (end_time - start_time)))
                rate = chunk_word_count / duration  # Words per second for the chunk
            else:
                rate = 0

            # Use the helper function to calculate the time slot
            time_slot = calculate_time_slot(current_time, next_chunk_end, chunk_size)

            # Initialize or update the word rates for this speaker and time slot
            if speaker not in word_rates:
                word_rates[speaker] = {}
            if time_slot in word_rates[speaker]:
                word_rates[speaker][time_slot] += Decimal(str(rate))
            else:
                word_rates[speaker][time_slot] = Decimal(str(rate))

            # Move to the next chunk
            current_time += chunk_size

    return word_rates
