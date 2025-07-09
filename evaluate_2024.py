#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from helper_code import *
import psutil
import json


# Function to calculate Net Benefit
def calculate_net_benefit(y_true, y_pred, threshold):
    """
    Calculate the net benefit of predictions.
    
    Parameters:
    - y_true (array-like): Ground truth binary labels (0 or 1).
    - y_pred (array-like): Predicted probabilities.
    - threshold (float): Threshold probability for classification (default is 0.5).
    
    Returns:
    - float: Net benefit score.
    """
    tp = np.sum((y_true == 1) & (y_pred >= threshold))
    fp = np.sum((y_true == 0) & (y_pred >= threshold))
    n = len(y_true)
    
    # Calculate Net Benefit
    net_benefit = (tp / n) - ((threshold / (1 - threshold)) * (fp / n))
    return net_benefit

# Function to calculate ECE (Estimated Calibration Error)
def calculate_ece(probs, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += bin_size * np.abs(bin_acc - bin_conf) / len(probs)
    return ece

# Function to read inference time from the output folder
import os

def read_inference_time(output_folder):

    inference_time_file = os.path.join(output_folder, 'inference_time.txt')
    if not os.path.exists(inference_time_file):
        raise FileNotFoundError(f"Inference time file not found in {output_folder}")
    
    metrics = {}
    with open(inference_time_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Inference time:"):
                metrics["inference_time"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Number of patients:"):
                metrics["num_patients"] = int(line.split(":")[1].strip())
            elif line.startswith("Average time per patient:"):
                metrics["average_time_per_patient"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Additional Memory Usage:"):
                metrics["additional_memory_usage"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Additional CPU Time:"):
                metrics["additional_cpu_time"] = float(line.split(":")[1].strip().split()[0])
    
    if "inference_time" not in metrics:
        raise ValueError("Inference time not found in the file.")
    
    return metrics

# Example usage:
# Assuming output_folder is the directory where 'inference_time.txt' was saved.
# metrics = read_inference_time(output_folder)
# print(metrics)


# Helper functions
def compute_confusion_matrix(labels, predictions):
    labels = np.array(labels).astype(int)
    predictions = np.array(predictions).astype(int)
    cm = np.zeros((2, 2))
    for i in range(len(labels)):
        cm[labels[i]][predictions[i]] += 1
    return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

def compute_accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)

def compute_f1(tn, fp, fn, tp):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator != 0 else 0.0

# Function to evaluate model
def evaluate_model(label_folder, output_folder, inference_time_file, threshold_file):
    # Load labels and model outputs.
    _, _, label, _ = load_challenge_data(label_folder)
    patient_ids, prediction_probability, prediction_binary = load_challenge_predictions(output_folder)
    
    # Read or compute threshold.
    threshold = read_or_compute_threshold(threshold_file, prediction_probability, prediction_binary)
    
    with open(inference_time_file, 'r') as f:
        lines = f.readlines()
        inference_time = None
        additional_memory_usage = None
        additional_cpu_time = None
        
        for line in lines:
            if line.startswith("Average time per patient:"):
                inference_time = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Additional Memory Usage:"):
                additional_memory_usage = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Additional CPU Time:"):
                additional_cpu_time = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Parsimony Score:"):
                parsimony_score = float(line.split(":")[1].strip())

        # Optionally, combine compute metrics into one variable (e.g., as a dict)
        compute = {
            'additional_memory_usage': additional_memory_usage,
            'additional_cpu_time': additional_cpu_time
        }

    num_predictions = len(prediction_binary)

    # Compute confusion matrix and metrics
    tn, fp, fn, tp = compute_confusion_matrix(label, prediction_binary)
    accuracy = compute_accuracy(tn, fp, fn, tp)
    F1 = compute_f1(tn, fp, fn, tp)

    # Additional metrics
    auc_score = roc_auc_score(label, prediction_probability)
    precision, recall, _ = precision_recall_curve(label, prediction_probability)
    auprc = auc(recall, precision)
    net_benefit = calculate_net_benefit(label, prediction_probability, threshold)
    ece = calculate_ece(prediction_probability, label)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    

    # Scores 
    return {
        'AUC': auc_score,
        'AUPRC': auprc,
        'Net Benefit': net_benefit,
        'ECE': ece,
        'tp':tp,
        'fp':fp,
        'fn':fn,
        'tn':tn,
        'F1':F1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Parsimony Score': parsimony_score,
        'Inference Time': inference_time,
        'Compute': compute
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument("label_folder", type=str, help="Folder containing ground truth labels.")
    parser.add_argument("output_folder", type=str, help="Folder containing model predictions.")
    parser.add_argument("inference_time_file", type=str, help="File containing inference time information.")
    parser.add_argument("threshold_file", type=str, nargs="?", help="Threshold file for classification (default: 0.5)")
    parser.add_argument("output_file", type=str, nargs="?", help="File to save the evaluation results.")
   

    args = parser.parse_args()

    # Perform evaluation
    metrics = evaluate_model(args.label_folder, args.output_folder, args.inference_time_file, args.threshold_file)

    # Create score dictionary
    submission_result = {
        'score': {
            'AUC': metrics['AUC'],
            'AUPRC': metrics['AUPRC'],
            'Net Benefit': metrics['Net Benefit'],
            'ECE': metrics['ECE'],
            'F1': metrics['F1'],
            'Sensitivity': metrics['Sensitivity'],
            'Specificity': metrics['Specificity'],
            'Parsimony Score': metrics['Parsimony Score'],
            'Inference Time': metrics['Inference Time'],
            'Compute': metrics['Compute'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'tn': metrics['tn']
        },
        'completion_time': time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    # Print or save the results
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(submission_result)) 
    else:
        print(json.dumps(submission_result))