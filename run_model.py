#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running models for the Challenge. You can run it as follows:
#
#   python run_model.py models data outputs
#
# where 'models' is a folder containing the your trained models, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your models' outputs.

import numpy as np, scipy as sp, os, sys
from helper_code import *
from team_code import load_challenge_model, run_challenge_model
import time
import psutil

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load model(s).
    if verbose >= 1:
        print('Loading the Challenge models...')

    # You can use this function to perform tasks, such as loading your models, that you only need to perform once.
    model = load_challenge_model(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    
    # Read selected variables using our helper function.
    selected_variables = read_selected_variables(model, model_folder)

    
    patient_ids, data, features = load_challenge_testdata(data_folder, selected_variables)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')
    
    # Compute parsimony score.
    # Compute parsimony score: (# selected variables) / total predictor available( i.e. 136, after excluding study id and inhospitals mortality columns)
    # If no selected variables are stored, assume all features were used.
    if selected_variables is None:
        selected_count = 136
    else:
        selected_count = len(selected_variables)
    parsimony_score = selected_count / 136



    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running the Challenge model on the Challenge data...')
    
    # Initialize output variables to handle failures gracefully.
    prediction_binary = []
    prediction_probability = []

    # Allow or disallow the models to fail on parts of the data; this can be helpful for debugging.
    # Measure inference time
    start_memory, start_cpu = compute_resource()
    start_time = time.time()
    try:
        patient_ids, prediction_binary, prediction_probability = run_challenge_model(model, data_folder, verbose) ### Teams: Implement this function!!!
    except:
        if allow_failures:
            if verbose >= 2:
                print('... failed.')
                prediction_binary, prediction_probability = float('nan'), float('nan')
        else:
            raise
    end_time = time.time()
    inference_time = end_time - start_time
    end_memory, end_cpu = compute_resource()

    # Compute the differences
    memory_used = end_memory - start_memory
    cpu_time_used = end_cpu - start_cpu     

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)                          
    
    # Save inference time
    inference_time_file = os.path.join(output_folder, 'inference_time.txt')
    with open(inference_time_file, 'w') as f:
        f.write(f"Inference time: {inference_time:.6f} seconds\n")
        f.write(f"Number of patients: {num_patients}\n")
        f.write(f"Average time per patient: {inference_time / num_patients:.6f} seconds\n")
        f.write(f"Additional Memory Usage: {memory_used:.2f} MB\n")
        f.write(f"Additional CPU Time: {cpu_time_used:.2f} seconds\n")
        f.write(f"Parsimony Score: {parsimony_score:.4f}\n")
        
        
    # Save Challenge outputs.
    output_file = os.path.join(output_folder, 'outputs' + '.txt')                     
    save_challenge_outputs(output_file, patient_ids, prediction_binary, prediction_probability)    

    if verbose >= 1:
        print('Done!')
  

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')
        
    # Define the model, data, and output folders.
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    allow_failures = False

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv)==5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)