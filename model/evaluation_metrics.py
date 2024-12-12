from jiwer import wer
from Levenshtein import distance as levenshtein_distance

def evaluate(predicted, ground_truth):
    """
    Calculates Character Error Rate (CER) using Levenshtein distance and Word Error Rate (WER) using jiwer.
    Args:
        predicted (list of str): List of predicted sequences.
        ground_truth (list of str): List of ground truth sequences.
    Returns:
        cer_value (float): Average CER over the batch.
        wer_value (float): Average WER over the batch.
    """
    total_chars = 0
    total_words = 0
    total_cer_distance = 0
    total_wer_distance = 0

    for pred, gt in zip(predicted, ground_truth):
        pred = pred.replace("_", " ").strip()
        gt = gt.replace("_", " ").strip()

        total_chars += len(gt)
        total_words += len(gt.split())
        total_cer_distance += levenshtein_distance(pred, gt)
        total_wer_distance += wer(gt, pred) * len(gt.split())
        
    cer_value = total_cer_distance / total_chars if total_chars > 0 else 0.0
    wer_value = total_wer_distance / total_words if total_words > 0 else 0.0
    return cer_value, wer_value
