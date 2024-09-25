import math
from typing import List
from services.nlp import softmax_weights_output  # Assuming this is an existing service.

# Sigmoid function to scale values
def sigmoid(x):
    return 1 / (1 + math.exp(-10 * (x - 0.5)))

# Compute the average of a list of scores
def compute_ave_score(scores: List[float]) -> float:
    return sum(scores) / len(scores)

# Main function to calculate body, behavior, and total scores
def get_scores(rppg_sync_data: List[List[float]], v_sync_data: List[List[float]], a_sync_data: List[List[float]]) -> dict:
    try:
        print("Processing rPPG data...")
        # Count valid body score entries and calculate the body score
        body_score_cnt = sum(1 for d in rppg_sync_data if not math.isnan(d[1]))
        print(f"Body score count: {body_score_cnt}")
        body_score = sum((d[1] / 2 + 0.5) for d in rppg_sync_data if not math.isnan(d[1])) / body_score_cnt
        print(f"Body score before sigmoid: {body_score}")
        body_score = sigmoid(body_score) * 100
        print(f"Body score after sigmoid: {body_score}")
        
    except Exception as e:
        print(f"Error processing rPPG data: {e}")
        return {}

    try:
        print("Processing radar chart data...")
        # Placeholder for radar chart data processing
        radar_chart_array = [0.0] * 5  # This could be updated with actual radar data
        radar_score = softmax_weights_output(radar_chart_array)
        print(f"Radar score calculated: {radar_score}")

    except Exception as e:
        print(f"Error processing radar chart data: {e}")
    
    try:
        print("Processing v_sync data...")
        # Process v_sync data directly as floats
        v_sync_array = [d[1] for d in v_sync_data if not math.isnan(d[1])]
        print(f"v_sync data parsed: {v_sync_array}")

        # Count valid v_sync entries and calculate v_score
        v_score_cnt = len(v_sync_array)
        v_score = sum((d / 2 + 0.5) for d in v_sync_array) / v_score_cnt
        print(f"v_score before sigmoid: {v_score}")

    except Exception as e:
        print(f"Error processing v_sync data: {e}")
        return {}

    try:
        print("Processing a_sync data...")
        # Process a_sync data directly as floats
        a_sync_array = [d[1] for d in a_sync_data if not math.isnan(d[1])]
        print(f"a_sync data parsed: {a_sync_array}")

        # Count valid a_sync entries and calculate a_score
        a_score_cnt = len(a_sync_array)
        a_score = sum((d / 2 + 0.5) for d in a_sync_array) / a_score_cnt
        print(f"a_score before sigmoid: {a_score}")

    except Exception as e:
        print(f"Error processing a_sync data: {e}")
        return {}

    try:
        # Calculate behavior score using v_score and a_score
        behavior_scores = [v_score, a_score]
        behavior_score = sigmoid(compute_ave_score(behavior_scores)) * 100
        print(f"Behavior score after sigmoid: {behavior_score}")

        # Calculate total score as a combination of body and behavior scores
        total_scores = [body_score, behavior_score]
        total_score = sigmoid(compute_ave_score(total_scores) / 100) * 100
        print(f"Total score after sigmoid: {total_score}")

        # Adjust scores to scale
        body_score = sigmoid(body_score / 100) * 100
        behavior_score = sigmoid(behavior_score / 100) * 100

    except Exception as e:
        print(f"Error calculating behavior or total scores: {e}")
        return {}

    # Return the final scores as a dictionary
    
    total = round(total_score)
    body = round(body_score)
    behavior = round(behavior_score)
    
    return total, body, behavior
