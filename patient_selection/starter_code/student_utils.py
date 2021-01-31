import pandas as pd
import numpy as np
import os
import tensorflow as tf
from random import sample
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    ndc_df = ndc_df[["NDC_Code", "Non-proprietary Name"]]
    ndc_df.columns = ["ndc_code", "generic_drug_name"]
    df = pd.merge(df, ndc_df, left_on = "ndc_code", right_on = "ndc_code")
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df = df.sort_values(["patient_nbr", "encounter_id"], ascending = [True, True])
    first_encounter_df = df.groupby("patient_nbr").agg('first').reset_index()
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    df = df.iloc[np.random.permutation(len(df))]
    
    ## obtenemos la lista de pacientes unica
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    
    ### calculate sample size
    sample_size_train = round(total_values* 0.6 )
    print(sample_size_train)
    patient_train = sample(list(unique_values), sample_size_train)
    unique_values_dif = list(df.loc[~df[patient_key].isin(patient_train), patient_key].unique())
    sample_size_test = round(len(unique_values_dif) * 0.5)
    patient_val = sample(unique_values_dif, sample_size_test)
    patient_test = df.loc[~df[patient_key].isin(patient_train + patient_val), patient_key].unique()


    ### extraemos los datos de train en base a ala cantidad de pacientes unicos
    train = df[df[patient_key].isin(patient_train)].reset_index(drop=True)
    validation =  df[df[patient_key].isin(patient_val)].reset_index(drop=True)
    test = df[df[patient_key].isin(patient_test)].reset_index(drop=True)
    
    print("Total number of unique patients in train = ", len(train[patient_key].unique()))
    print("Total number of unique patients in validation = ", len(validation[patient_key].unique()))
    print("Total number of unique patients in test = ", len(test[patient_key].unique()))
    print("Training partition has a shape = ", train.shape) 
    print("validation partition has a shape = ", validation.shape) 
    print("Test partition has a shape = ", test.shape)   

    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''

        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''

    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature =  tf.feature_column.numeric_column(
    key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = np.where(df[col]>5,1, 0)

    return student_binary_prediction
