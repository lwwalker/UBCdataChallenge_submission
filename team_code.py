#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np
import os 
import sys
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.impute import SimpleImputer                  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, X, y, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    #Load lists of columns of different types
    var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var = getColLists()
    X = processData(X, var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var)

    columns = X.columns
    # Save the column names for later use during inference
    with open(os.path.join(model_folder, 'columns.txt'), 'w') as f:
        f.write("\n".join(columns))
    
    # Impute any missing features; use the mean value by default.    
    imputer = SimpleImputer().fit(X)
    X = imputer.transform(X)
    
    # Define parameters for XGboost
    subsample = 0.8
    scale_pos_weight = 25.0
    reg_lambda = 0.001
    reg_alpha = 1
    min_child_weight = 2
    max_depth = 100
    max_delta_step = 3
    grow_policy = 'depthwise'
    gamma = 0.1
    eta = 0.2
    colsample_bytree = 0.9
    booster = 'dart'
    
    #Fit the XGboost model
    mod = xgb.XGBClassifier(random_state = 619,     
        subsample = subsample, scale_pos_weight = scale_pos_weight, reg_lambda = reg_lambda, 
        reg_alpha = reg_alpha, min_child_weight = min_child_weight, max_depth = max_depth, 
        max_delta_step = max_delta_step, grow_policy = grow_policy, gamma = gamma, eta = eta, 
        colsample_bytree = colsample_bytree, booster = booster).fit(X,y)

    # Save the models.
    save_challenge_model(model_folder, imputer, mod)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')

    # Load the saved column names
    with open(os.path.join(model_folder, 'columns.txt'), 'r') as f:
        columns = f.read().splitlines()
    
    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    model['columns'] = columns
    return model

def run_challenge_model(model, data_folder, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']
    columns = model['columns']

    # Load data.
    patient_ids, X, y, features = load_challenge_data(data_folder)
    
    #Process the data 
    var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var = getColLists()
    X = processData(X, var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var)
    
    # Align test data with training columns, filling missing columns with 0
    X = X.reindex(columns=columns, fill_value=0)
    
    # Impute missing data.
    X = imputer.transform(X)

    # Apply model to data.
    prediction_binary = prediction_model.predict(X)[:]
    prediction_probability = prediction_model.predict_proba(X)[:, 1]

    return patient_ids, prediction_binary, prediction_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, prediction_model):
    d = {'imputer': imputer, 'prediction_model': prediction_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

#Define which columns should be transformed in what way (just to keep the main code tidy)
def getColLists():
    #Define how each value will be encoded into our data frame
    #These variables will be dropped either because we've been asked not to use them, or there is almost no data in the column
    var_to_drop = [
               'SpO2_3', 'admitabx_adm___1',  'admitabx_adm___2',  'admitabx_adm___3',  
               'admitabx_adm___4', 'admitabx_adm___5', 'admitabx_adm___6', 'admitabx_adm___7', 
               'admitabx_adm___8', 'admitabx_adm___9', 'admitabx_adm___10', 'admitabx_adm___11', 
               'admitabx_adm___12', 'admitabx_adm___13', 'admitabx_adm___14', 'admitabx_adm___15', 
               'admitabx_adm___16', 'admitabx_adm___17', 'admitabx_adm___18', 'admitabx_adm___19', 
               'admitabx_adm___20', 'admitabx_adm___21', 'lactate2_mmolpl_adm', 'lengthadm'
    ]

    #These are numeric continuous variables - initially, will be passed through unchanged (not centered or normalized)
    continuous_var = [
               'agecalc_adm','height_cm_adm', 'weight_kg_adm', 'muac_mm_adm', 'hr_bpm_adm', 'rr_brpm_app_adm', 
               'sysbp_mmhg_adm', 'diasbp_mmhg_adm', 'temp_c_adm', 'spo2site1_pc_oxi_adm', 'spo2site2_pc_oxi_adm', 
               'spo2other_adm' , 'momage_adm', 'momagefirstpreg_adm', 'householdsize_adm', 
               'alivechildren_adm', 'deadchildren_adm', 'hematocrit_gpdl_adm', 'lactate_mmolpl_adm', 
               'glucose_mmolpl_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm'
    ]

    #These variables contain text and will be re-encoded as non-ordinal binary variables
    categorical_var = [
               'oxygenavail_adm', 'deliveryloc_adm', 'birthattend_adm', 'travelmethod_adm', 'caregivermarried_adm',
               'watersource_adm', 'cookloc_adm', 'lightfuel_adm', 'nonexclbreastfed_adm','caregiver_adm_new'
    ]

    #These variables are binary, but are not reliably encoded as 1/0 or True/False.  List is expressed as a 
    #Pyton dictionary with each entry in the form of variableName: response to be encoded as True or 1
    binary_var = {
              'sex_adm': 'Female', 'respdistress_adm': 'Yes', 'caprefill_adm': 'Yes', 'bcgscar_adm': 'Yes', 
              'vaccmeasles_adm': 'Yes', 'vaccmeaslessource_adm': 'Card', 'vaccpneumocsource_adm': 'Card', 'vaccdptsource_adm': 'Card', 
              'priorweekabx_adm': 'Yes', 'priorweekantimal_adm': 'Yes', 'symptoms_adm___1': 'Checked',
              'symptoms_adm___2': 'Checked', 'symptoms_adm___3': 'Checked', 'symptoms_adm___4': 'Checked',
              'symptoms_adm___5': 'Checked', 'symptoms_adm___6': 'Checked', 'symptoms_adm___7': 'Checked',
              'symptoms_adm___8': 'Checked', 'symptoms_adm___9': 'Checked', 'symptoms_adm___10': 'Checked',
              'symptoms_adm___11': 'Checked', 'symptoms_adm___12': 'Checked', 'symptoms_adm___13': 'Checked',
              'symptoms_adm___14': 'Checked', 'symptoms_adm___15': 'Checked', 'symptoms_adm___16': 'Checked', 
              'comorbidity_adm___1': 'Checked', 'comorbidity_adm___2': 'Checked', 'comorbidity_adm___3': 'Checked',
              'comorbidity_adm___4': 'Checked', 'comorbidity_adm___5': 'Checked', 'comorbidity_adm___6': 'Checked',
              'comorbidity_adm___7': 'Checked', 'comorbidity_adm___8': 'Checked', 'comorbidity_adm___9': 'Checked',
              'comorbidity_adm___10': 'Checked', 'comorbidity_adm___11': 'Checked', 'comorbidity_adm___12': 'Checked',
              'prioryearwheeze_adm': 'Yes', 'prioryearcough_adm': 'Yes', 'diarrheaoften_adm': 'Checked', 
              'tbcontact_adm': 'Checked', 'duedateknown_adm': 'Yes', 'birthdetail_adm___1': 'Checked', 'birthdetail_adm___2': 'Checked',
              'birthdetail_adm___3': 'Checked',  'birthdetail_adm___4': 'Checked', 'birthdetail_adm___5': 'Checked',
              'birthdetail_adm___6': 'Checked', 'momalive_adm': 'Yes', 'momageknown_adm': 'Yes', 'momagefirstpregknown_adm': 'Yes',
              'momhiv_adm': 'Positive', 'waterpure_adm': 'Yes', 'cookfuel_adm___1': 'Checked', 'cookfuel_adm___2': 'Checked',
              'cookfuel_adm___3': 'Checked', 'cookfuel_adm___4': 'Checked', 'cookfuel_adm___5': 'Checked', 'cookfuel_adm___6': 'Checked',
              'cookfuel_adm___7': 'Checked', 'hctpretransfusion_adm': 'Yes', 'hivstatus_adm': 'HIV positive', 
              'malariastatuspos_adm': 'Yes'
    }

    #These are ordinal categorical variables - the list is a Python dictonary of dictionaries with each entry
    #in the form of variableName: {value: numericCode, value: numericCode, ...}
    #In general, values that amount to missing/unknown are encoded as zero - other approaches can be considered
    ordinal_var = {
        'bcseye_adm': {'Fails to watch or follow': 1, 'Watches or follows': 2, np.nan: 0},
        'bcsmotor_adm': {'No response or inappropriate response': 1,'Withdraws limb from painful stimulus': 2,  'Localizes painful stimulus': 3, np.nan: 0},
        'bcsverbal_adm': {'No vocal response to pain': 1, 'Moan or abnormal cry with pain': 2, 'Cries appropriately with pain, or, if verbal, speaks': 3, np.nan: 0},
        'priorhosp_adm': {'Dont know': 0, 'Never': 1, '> 1 year': 2, '1 month - 1 year': 3,'7 days - 1 month': 4 , '< 7 days': 5, '< 3 days': 6, np.nan: 0},
        'feedingstatus_adm': {'Not feeding at all': 1,'Feeding poorly': 2,'Feeding well': 3, np.nan: 0},
        'exclbreastfed_adm': {'Unknown': 0, 'never exclusively breastfed': 1, '1 month': 2, '2 months': 3, '3 months': 4, '4 months': 5 , '5 months': 6, '6 months': 7, 'More than 6 months': 8, 'Currently exclusively breastfed': 9, np.nan: 0},
        'traveldist_adm': {'< 30 minutes': 1, '30 minutes - 1 hour': 2,'1 - 2 hours': 3, '2 hours': 4, '2 - 3 hours': 5, '3 - 4 hours': 6, '4 - 8 hours': 7, '> 8 hours': 8, np.nan: 0},
        'badhealthduration_adm': {'Unknown': 0, 'In good health prior to this illness': 1, '< 1 week': 2, '1 week - 1 month': 3, '1 month - 1 year': 4, '> 1 year': 5, np.nan: 0},
        'caregiverage_adm':  {'N/A (in care)': 0, '< = 18 years old': 1, '> 18 years old': 2, '> 50 years old': 3, np.nan: 0},
        'momedu_adm': {'Dont know': 0, 'No school': 1, '< = P3': 2, 'P4-P7': 3, 'S1-S6': 4, 'Post secondary (including post S4 technical school)': 5, np.nan: 0},
        'tobacco_adm': {'Never': 1, 'Less than monthly': 2, 'Weekly': 3, 'Monthly': 4, 'Daily': 5, np.nan: 0},
        'bednet_adm': {'Never': 0, 'Always': 1, 'Sometimes': 2 , np.nan: 0},
        'vaccpneumoc_adm': {'Unknown': 0, '0 doses': 1, '1 dose': 2, '2 doses': 3, '3 doses': 4, np.nan: 0},
        'vaccdpt_adm': {'Unknown': 0, '0 doses': 1, '1 dose': 2, '2 doses': 3, '3 doses': 4, np.nan: 0}
    }

    return (var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var)
    
#Process the columns as described below
def processData(X, var_to_drop, continuous_var, categorical_var, binary_var, ordinal_var):
    #Continuous variables pass through without processing
    X_raw = X
    X = X_raw[continuous_var]

    #Binary variables are transformed from their text forms to 0/1
    X = pd.concat([X,pd.DataFrame({col: X_raw[col] == target for col, target in binary_var.items()})], axis = 1)

    #Categorical variables use one-hot encoding to make 0/1
    X = pd.concat([X, pd.get_dummies(X_raw[categorical_var], drop_first = True).astype(int)], axis = 1)

    #Ordinal variables are mapped to integer values
    X = pd.concat([X, pd.DataFrame({col: [myHash[val] for val in X_raw[col]] for col, myHash in ordinal_var.items()}, index = X.index)], axis = 1)

    #Take the spaces out of column names for compatibility with various modules
    X.columns = X.columns.str.replace(' ', '')
    return X