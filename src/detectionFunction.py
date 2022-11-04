# modules
import venn
import os
from turtle import delay
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras.backend as K

# GPU memory growth true
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# R env
if len(gpus) == 0:
    R_HOME = os.path.expanduser('~') + '/anaconda3/envs/multiomics-cpu/lib/R'
else :
    R_HOME = os.path.expanduser('~') + '/anaconda3/envs/multiomics-gpu/lib/R'

import time
import json
import subprocess
import zlib
import urllib.request
import codecs
import csv

from pypdb import *
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
from collections import defaultdict    
import re
import pickle
import datetime
from requests import get
from pathlib import Path
import random
from retry import retry

import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted
import gc

# sklearn
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K

# rpy2
os.environ['R_HOME'] = R_HOME # env R invoke
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# db connection
import pymysql
from sqlalchemy import create_engine

aws_mariadb_url = 'mysql+pymysql://root:sempre813!@192.168.0.91:3306/Textmining'
engine_mariadb = create_engine(aws_mariadb_url)  

# function
def cancer_select(cols, cancer_type, raw_path):
    # phenotype
    phe1 = pd.read_csv(raw_path + "GDC-PANCAN.basic_phenotype.tsv", sep="\t")
    phe1 = phe1.loc[phe1.program == "TCGA", :].loc[:, ['sample', 'sample_type', 'project_id']].drop_duplicates(['sample'])
    phe1['sample'] =  phe1.apply(lambda x : x['sample'][:-1], axis=1)
    phe2 = pd.read_csv(raw_path + "TCGA_phenotype_denseDataOnlyDownload.tsv", sep="\t")
    ph_join = pd.merge(left = phe2 , right = phe1, how = "left", on = "sample").dropna(subset=['project_id'])
    
    if cancer_type == "PAN" or cancer_type == "PANCAN":
        filterd = ph_join.loc[ph_join['sample_type_y'] == "Primary Tumor", :]
        sample_barcode = filterd["sample"].tolist()
    else:
        filterd = ph_join.loc[(ph_join['sample_type_y'] == "Primary Tumor") & (ph_join['project_id'] == "TCGA-" + cancer_type) , :]
        sample_barcode = filterd["sample"].tolist()
        
    intersect_ = list(set(cols).intersection(sample_barcode))
    
    return intersect_

def mc3Mut_load(MAF_PATH):
    r = ro.r
    r['source']('src/r-function.R')
    mut_load = ro.globalenv['mut_load']
    
    mut = mut_load(MAF_PATH)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        mut = ro.conversion.rpy2py(mut)
    
    return mut

def non_zero_column(DF):
    sample_cnt = int(len(DF.columns) * 0.2)
    zero_row = dict(DF.isin([0]).sum(axis=1))
    non_remove_feature = list()

    for key, value in zero_row.items():
        if value < sample_cnt:
            non_remove_feature.append(key)
    
    return non_remove_feature

def load_tcga_dataset_version1(pkl_path:str, raw_path:str, cancer_type:str, norm:bool, minmax:bool=None) -> pd.DataFrame :    

    """
    Multi-omics load
    ===========
    
    Args:
         pkl_path (str) : 각 Omics의 pickle이 저장된 Path (없어도 됨, 첫 실행시 생성)
         raw_path (str) : 각 Omics의 TCGA Pan-Cancer(PANCAN) RAW 파일 (*.gz), https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
         cancer_type (str) : TCGA abbreviation, e.g., COAD
         norm (bool) : MinMaxScaler, StandardScaler
         minmax (bool) : MinxmaxScaler 사용 유무
    
    Return Pandas DataFrame

    Note :
         - 특정 Primary tumor의 mRNA, miRNA, Mehtylatio 이 통합된 Dataframe. Row : Sample X Column : Feature (omics)
         - mRNA : Tcga_RSEM_Hugo_norm_count.xena.gz
         - miRNA : Batch_effects_normalized_miRNA.xena.gz
         - Methylation : Methylation450K.xena.gz

    """

    if os.path.isfile(pkl_path + cancer_type + "_omics.pkl"):
        omics = pd.read_pickle(pkl_path  + cancer_type + "_omics.pkl")

        # sep
        rna = pd.read_pickle(pkl_path  + cancer_type + "_rna.pkl")
        mirna = pd.read_pickle(pkl_path + cancer_type + "_mirna.pkl")
        mt = pd.read_pickle(pkl_path  + cancer_type + "_mt.pkl")
        
        # intersect
        venn3_unweighted([set(rna.index), set(mirna.index), set(mt.index)], ('RNA', 'miRNA', 'Methylation'))
        # plt.show()
        plt.ioff()
        plt.savefig(raw_path + cancer_type + "_subset.png")
        
    else :
        # create dir
        Path(pkl_path).mkdir(parents=True, exist_ok=True)
        Path(raw_path).mkdir(parents=True, exist_ok=True)
        
        # file name
        mrna_f = 'tcga_RSEM_Hugo_norm_count.xena.gz'
        mirna_f = 'Batch_effects_normalized_miRNA.xena.gz'
        mt_f = 'Methylation450K.xena.gz'
        mt_probe_f = 'Methylation450K.hg19.GPL16304'

        # file url
        mrna_url = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_Hugo_norm_count.gz"
        mirna_url = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz"
        mt_url = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz"
        mt_probe_url = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy"

         # to make GET request
        def download(url, file_name):
            with open(file_name, "wb") as file:   # open in binary mode
                response = get(url)               # get request
                file.write(response.content)      # write to file

        # mrna
        if os.path.isfile(raw_path + mrna_f) == False:
            download(mrna_url, raw_path + mrna_f)

        # mirna
        if os.path.isfile(raw_path + mirna_f) == False:
            download(mirna_url, raw_path + mirna_f)

        # methylation
        if os.path.isfile(raw_path + mt_f) == False:
            download(mt_url, raw_path + mt_f)
            download(mt_probe_url, raw_path + mt_probe_f)
        
        
        # RNA gene expression
        if os.path.isfile(pkl_path + cancer_type + "_rna.pkl") == False:
            col = pd.read_csv(raw_path + mrna_f,
                         sep = "\t", index_col=0, nrows=0).columns.to_list()
            use_col = ['sample'] + cancer_select(cols=col, cancer_type=cancer_type, raw_path=raw_path)
            df_chunk = pd.read_csv(raw_path + mrna_f,
                         sep = "\t", index_col=0, iterator=True, chunksize=50000, usecols=use_col)
            rna = pd.concat([chunk for chunk in df_chunk])
            rna = rna[rna.index.isin(non_zero_column(rna))].T

            rna.to_pickle(pkl_path + cancer_type + "_rna.pkl")
        else : 
            rna = pd.read_pickle(pkl_path  + cancer_type + "_rna.pkl")
            
        # miRNA expression
        if os.path.isfile(pkl_path + cancer_type + "_mirna.pkl") == False:            
            col = pd.read_csv(raw_path + mirna_f, sep = "\t", index_col=0, nrows=0).columns.to_list()
            use_col = ['sample'] + cancer_select(cols=col, cancer_type=cancer_type, raw_path=raw_path)

            df_chunk = pd.read_csv(raw_path + mirna_f, sep = "\t", index_col=0, iterator=True, chunksize=50000, usecols=use_col)
            mirna = pd.concat([chunk for chunk in df_chunk])
            mirna = mirna[mirna.index.isin(non_zero_column(mirna))].T

            mirna.to_pickle(pkl_path + cancer_type + "_mirna.pkl")
        else :
            mirna = pd.read_pickle(pkl_path + cancer_type + "_mirna.pkl")            
            
        # methylation
        if os.path.isfile(pkl_path + cancer_type + "_mt.pkl") == False: 
            col = pd.read_csv(raw_path + mt_f, sep = "\t", index_col=0, nrows=0).columns.to_list()
            use_col = ['sample'] + cancer_select(cols=col, cancer_type=cancer_type, raw_path=raw_path)

            df_chunk = pd.read_csv(raw_path + mt_f, sep = "\t", index_col=0, iterator=True, chunksize=50000, usecols=use_col)
            mt = pd.concat([chunk for chunk in df_chunk])

            mt_map = pd.read_csv(raw_path + mt_probe_f, sep="\t")

            mt_join = pd.merge(mt, mt_map, how = "left", left_on = "sample", right_on = "#id")\
                     .drop(['chrom', 'chromStart', 'chromEnd', 'strand', '#id'], axis=1)
            mt_join = mt_join[mt_join.gene != "."]
            mt_join.dropna(subset = ["gene"], inplace=True)

            # gene mean 
            mt_join_gene_filter = mt_join.groupby(['gene']).mean()
            mt_join_gene_filter = mt_join_gene_filter[mt_join_gene_filter.index.isin(non_zero_column(mt_join_gene_filter))].T

            mt_join_gene_filter.to_pickle(pkl_path + cancer_type + "_mt.pkl")
        else :
            mt_join_gene_filter = pd.read_pickle(pkl_path + cancer_type + "_mt.pkl")            
            
        
        # intersect
        venn3_unweighted([set(rna.index), set(mirna.index), set(mt_join_gene_filter.index)], ('RNA', 'miRNA', 'Methylation'))
        # plt.show()
        plt.ioff()
        plt.savefig(raw_path + cancer_type + "_subset.png")
        
        # set same column for merge
        rna['sample'] = rna.index
        mirna['sample'] = mirna.index
        mt_join_gene_filter['sample'] = mt_join_gene_filter.index

        # data join
        merge_list = [rna, mirna, mt_join_gene_filter]
        omics = reduce(lambda left, right : pd.merge(left, right, on = "sample"), merge_list)
        omics.set_index('sample', inplace=True)

        # pickle save
        omics.to_pickle(pkl_path + "/" + cancer_type + "_omics.pkl")
    
    # set index
    omics_index = omics.index.to_list()
    
    # normalization
    if norm:
        if minmax:
            scalerX = MinMaxScaler()
            omics_scale = scalerX.fit_transform(omics)
        else :
            scalerX = StandardScaler()      
            omics_scale = scalerX.fit_transform(omics)
    
    # missing impute
    imputer = KNNImputer(n_neighbors=10)
    omics_impute = imputer.fit_transform(omics_scale)

    omics = pd.DataFrame(omics_impute, columns=omics.columns)
    omics.index = omics_index

    return omics

# autoencoder
def run_ae(X_train : pd.DataFrame, X_test : pd.DataFrame):

    """
    Stacked Auto-Encoer (vanilla)
    ===========
    
    Args:
         X_train (pd.DataFrame) : Omics DataFrame
         X_test (pd.DataFrame) : Omics DataFrame
    
    Return Encoder model
    """

    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    # encoder - decoder
    inputs_dim = X_train.shape[1]
    encoder = Input(shape = (inputs_dim, ))
    e = Dense(1000, activation = "tanh")(encoder)
    e = Dense(500, activation = "tanh")(e)

    ## bottleneck layer
    n_bottleneck = 100

    ## defining it with a name to extract it later
    bottleneck_layer = "bottleneck_layer"

    # can also be defined with an activation function, relu for instance
    bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)

    ## define the decoder (in reverse)
    decoder = Dense(500, activation = "tanh")(bottleneck)
    # decoder = Dense(256, activation = "relu")(decoder)
    decoder = Dense(1000, activation = "tanh")(decoder)

    ## output layer
    output = Dense(inputs_dim)(decoder)

    ## end-to-end model
    model = Model(inputs = encoder, outputs = output)

    # encdoer mdoel
    encoder = Model(inputs = model.input, outputs = bottleneck)

    # compile & fit
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")
    history = model.fit(
        X_train,
        X_train,
        batch_size = 128,
        epochs = 30,
        verbose = 0,
        validation_data = (X_test, X_test)
    ) 
    
    return encoder

# autoencoder
def run_ae_denoisy(X_train : pd.DataFrame, X_test : pd.DataFrame):

    """
    Denoise Auto-Encoer
    ===========
    
    Args:
         X_train (pd.DataFrame) : Omics DataFrame
         X_test (pd.DataFrame) : Omics DataFrame
    
    Return Encoder model
    """

    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    # noise
    noise_factor = 0.5
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=0.3, size=X_train.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=0.3, size=X_test.shape)

    # encoder - decoder
    inputs_dim = X_train.shape[1]
    encoder = Input(shape = (inputs_dim, ))
    e = Dense(1000, activation = "tanh")(encoder)
    e = Dense(500, activation = "tanh")(e)

    ## bottleneck layer
    n_bottleneck = 100

    ## defining it with a name to extract it later
    bottleneck_layer = "bottleneck_layer"

    # can also be defined with an activation function, relu for instance
    bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)

    ## define the decoder (in reverse)
    decoder = Dense(500, activation = "tanh")(bottleneck)
    # decoder = Dense(256, activation = "relu")(decoder)
    decoder = Dense(1000, activation = "tanh")(decoder)

    ## output layer
    output = Dense(inputs_dim)(decoder)

    ## end-to-end model
    model = Model(inputs = encoder, outputs = output)

    # encdoer mdoel
    encoder = Model(inputs = model.input, outputs = bottleneck)


    # compile & fit
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")
    history = model.fit(
        X_train_noisy,
        X_train_noisy,
        batch_size = 128,
        epochs = 30,
        verbose = 0,
        validation_data = (X_test, X_test)
    ) 
    
    return encoder

def run_ae_sparse(X_train : pd.DataFrame, X_test : pd.DataFrame):

    """
    Sparse Auto-Encoer
    ===========
    
    Args:
         X_train (pd.DataFrame) : Omics DataFrame
         X_test (pd.DataFrame) : Omics DataFrame
    
    Return Encoder model
    """

    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    
    # encoder - decoder
    inputs_dim = X_train.shape[1]
    encoder = Input(shape = (inputs_dim, ))
    e = Dense(1000, activation = 'tanh',activity_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001,l2=0.00001))(encoder)
    e = Dense(1000, activation = "tanh")(e)

    ## define the decoder (in reverse)
    decoder = Dense(1000, activation = 'tanh',activity_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001,l2=0.00001))(e)
    output = Dense(inputs_dim)(decoder)

    ## end-to-end model
    model = Model(inputs = encoder, outputs = output)

    # encdoer mdoel
    encoder = Model(inputs = model.input, outputs = e)

    # compile & fit
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")

    history = model.fit(
        X_train,
        X_train,
        batch_size = 128,
        epochs = 30,
        verbose = 0,
        validation_data = (X_test, X_test)
    ) 
    
    return encoder

def best_ae_model(model_list:list, o:pd.DataFrame, group_path:str, model_path:str, cancer_type:str, file_name:str, raw_path:str,
                  model_names:list = ['encoder_vanilla', 'encoder_sparse', 'encoder_denoisy']):

    """
    Best Auto encoder selection
    ===========
    
    Args:
         model_list (list) : run_ae, run_ae_sparse, run_ae_denoisy의 tensorflow model
         o (pd.DataFrame) : Omics DataFrame

    Return 
        K-mean clustering sampling 후 classifiaction 된 Pandas DataFrame
        Silhouette Coefficient

    Note :
        - 3개의 Autoencoder 중 Silhouette Coefficient이 가장 좋은 Autoencoder model 선택 
        - K-means clustring 및 Silhouette Coefficient 추출
        - https://nicola-ml.tistory.com/6
    """
    
    def model_prediction(model):

        """
        Encoder Prediction
        ===========
        
        Args:
            model (tensorflow model) : encoder
            
        Return 
            K-mean clustering sampling 후 classifiaction 된 Pandas DataFrame
            Silhouette Coefficient
            - nb_result_list[?][0] - silhoaute score
        """    

        if model[1] == 'encoder_denoisy':
            noise_factor = 0.5
            o2 = o + noise_factor + np.random.normal(loc=0.0, scale=0.3, size=o.shape)
        else :
            o2 = o
        
        omic_encoded = pd.DataFrame(model[0].predict(o2))
        column_name = ["Feature" + str(index) for index in range(1, len(omic_encoded.columns) + 1)]
        omic_encoded.columns = column_name

        omic_encoded['sample'] = o2.index.to_list()
        omic_encoded.set_index('sample', inplace=True)
      
        # pheno = pd.read_csv("https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp", 
        #             sep = "\t", usecols=['sample', 'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'])
        pheno = pd.read_csv(raw_path + "Survival_SupplementalTable_S1_20171025_xena_sp",
                            sep = "\t", usecols=['sample', 'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'])

        # encoded pheno
        omic_encoded_pheno = pd.merge(left=omic_encoded, right=pheno, how="inner", on="sample")
        omic_encoded_pheno.set_index('sample', inplace=True)
        
        # log rank test
        omic_encoded_fc = omic_encoded[log_test(omic_encoded_pheno)]
        nb_result = nb_cluster(omic_encoded_fc)
        
        return nb_result.iloc[:, 0].to_list()[0], omic_encoded_fc
    
    # best model selection
    nb_result_list = list(map(model_prediction, list(zip(model_list, model_names))))
    zipbObj = zip(model_names, list(zip(nb_result_list, model_list)))
    model_sil = dict(zipbObj)
    model_sil_sort = sorted(model_sil.items(), key = lambda item : item[1][0], reverse=True) 
    best_model_n, best_model, s_score, encoded = model_sil_sort[0][0], model_sil_sort[0][1][1], model_sil_sort[0][1][0][0],   model_sil_sort[0][1][0][1]
    
    # model save
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(model_path + cancer_type).mkdir(parents=True, exist_ok=True)
    best_model.save(model_path + cancer_type + "/" + "AE_" + best_model_n + "_"+ cancer_type + "_" + file_name)
    
    pr = 'Best AE : {0}'.format(best_model_n)
    print(pr)
    
    # k-mean clustering
    clusterer = KMeans(n_clusters = 2, random_state = 31, max_iter = 1000)
    kmeans = clusterer.fit_predict(encoded)
    
    ae_groups = pd.DataFrame(kmeans, columns = ['group'])
    ae_groups['sample'] = encoded.index.to_list()
    ae_groups.set_index('sample', inplace=True)
    
    # dir check
    Path(group_path).mkdir(parents=True, exist_ok=True)
    Path(group_path + cancer_type).mkdir(parents=True, exist_ok=True)
    ae_groups.to_csv(group_path + cancer_type + "/" + cancer_type + "_GROUP_" + file_name + ".txt", sep="\t")
    
    return ae_groups, s_score

# invoke r
def log_test(df:pd.DataFrame) -> list:
    
    """
    Stacked Auto-Encoer (vanilla)
    ===========
    
    Args:
         X_train (pd.DataFrame) : Omics DataFrame
         X_test (pd.DataFrame) : Omics DataFrame
    
    Return Encoder model
    """

    # pandas DF to R DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(df)

    # R UDF invoke
    r = ro.r
    r['source']('src/r-function.R')
    log_rank_test_r = ro.globalenv['log_rank_test']
    log_rank_test_feature = log_rank_test_r(r_from_pd_df)

    # R DF to pandas DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        log_rank_test = ro.conversion.rpy2py(log_rank_test_feature)

    feature_log = log_rank_test['Features'].to_list()
    
    return feature_log

# pandas DF to R DF
def group_convert(sample_group, raw_path):           
    r = ro.r
    r['source']('src/r-function.R')
    survFit_r = ro.globalenv['survFit']
    survFit_result = survFit_r(sample_group, raw_path)
    
    # R DF to pandas DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        survFit_result = ro.conversion.rpy2py(survFit_result)
    
    # group 0 - long survival group (wild type), control
    # group 1 - short survival group (mu type), case
    if survFit_result.iloc[0, 4] > survFit_result.iloc[1, 4]:
        return False
    else:
        return True

def log_rank_test_py(df, png_path, cancer_type, file_name):
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.py2rpy(df)

    r = ro.r
    r['source']('src/r-function.R')
    survDiff_r = ro.globalenv['log_rank_test_group']
    # survDiff_pvalue = survDiff_r(df, png_path, cancer_type, file_name)
    survDiff_pvalue = survDiff_r(df)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        survDiff_pvalue = ro.conversion.rpy2py(survDiff_pvalue)

    return survDiff_pvalue[0]

# invoke r
def nb_cluster(df):
    # pandas DF to R DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        omic_encoded_fc_r = ro.conversion.py2rpy(df)

    r = ro.r
    r['source']('src/r-function.R')
    nb_cluster_test = ro.globalenv['nb_cluster_test']
    nb_cluster_test_feature = nb_cluster_test(omic_encoded_fc_r)

    # R DF to pandas DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        omic_encoded_fc_r = ro.conversion.rpy2py(nb_cluster_test_feature)
        
    return omic_encoded_fc_r


        
    
## For Target
def load_preprocess_tcga_dataset(pkl_path, raw_path, group, norm, cancer_type):
    
    if os.path.isfile(pkl_path + "/" + cancer_type + "_omics.pkl"):
        # sep
#         omics = pd.read_pickle(pkl_path + "/" + cancer_type + "_omics.pkl")
        rna = pd.read_pickle(pkl_path + "/" + cancer_type + "_rna.pkl")
        mirna = pd.read_pickle(pkl_path + "/" + cancer_type + "_mirna.pkl")
        mt_join_gene_filter = pd.read_pickle(pkl_path + "/" + cancer_type + "_mt.pkl")
        
    else :
        raise Exception("omics's pkl not exist!")
        
    # set column for unique
#     omics.columns = list(map(lambda col : col + "_OMICS", omics.columns.to_list()))
    rna.columns = list(map(lambda col : col + "_RNA", rna.columns.to_list()))
    mirna.columns = list(map(lambda col : col + "_miRNA", mirna.columns.to_list()))
    mt_join_gene_filter.columns = list(map(lambda col : col + "_Methylation", mt_join_gene_filter.columns.to_list()))
        
    # set index
#     omics_index = omics.index.to_list()
    rna_index = rna.index.to_list()
    mirna_index = mirna.index.to_list()
    mt_join_gene_filter_index = mt_join_gene_filter.index.to_list()
    
    # normalization
    if norm:
        scalerX = StandardScaler()
#         omics_scale = scalerX.fit_transform(omics)
        rna_scale = scalerX.fit_transform(rna)
        mirna_scale = scalerX.fit_transform(mirna)
        mt_join_gene_filter_scale = scalerX.fit_transform(mt_join_gene_filter)

        # missing impute
        imputer = KNNImputer(n_neighbors=10)
#         omics_impute = imputer.fit_transform(omics_scale)        
        rna_impute = imputer.fit_transform(rna_scale)
        mirna_impute = imputer.fit_transform(mirna_scale)
        mt_join_gene_filter_impute = imputer.fit_transform(mt_join_gene_filter_scale)

        # Pandas
#         omics = pd.DataFrame(omics_impute, columns=omics.columns)
#         omics.index = omics_index       
        
        rna = pd.DataFrame(rna_impute, columns=rna.columns)
        rna.index = rna_index

        mirna = pd.DataFrame(mirna_impute, columns=mirna.columns)
        mirna.index = mirna_index

        mt = pd.DataFrame(mt_join_gene_filter_impute, columns=mt_join_gene_filter.columns)
        mt.index = mt_join_gene_filter_index
        
    else :
        # missing impute
        imputer = KNNImputer(n_neighbors=10)
#         omics_impute = imputer.fit_transform(omics)                
        rna_impute = imputer.fit_transform(rna)
        mirna_impute = imputer.fit_transform(mirna)
        mt_join_gene_filter_impute = imputer.fit_transform(mt_join_gene_filter)

        # Pandas
#         omics = pd.DataFrame(omics_impute, columns=omics.columns)
#         omics.index = rna_index       
        
        rna = pd.DataFrame(rna_impute, columns=rna.columns)
        rna.index = rna_index

        mirna = pd.DataFrame(mirna_impute, columns=mirna.columns)
        mirna.index = mirna_index

        mt = pd.DataFrame(mt_join_gene_filter_impute, columns=mt_join_gene_filter.columns)
        mt.index = mt_join_gene_filter_index
    
    # phenotype only omics 
    # pheno = pd.read_csv("https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp", 
    #                 sep = "\t", usecols=['sample', 'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'])
    pheno = pd.read_csv(raw_path + "Survival_SupplementalTable_S1_20171025_xena_sp",
                            sep = "\t", usecols=['sample', 'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'])                
    pheno.set_index('sample', inplace=True)
    
    join_list = [rna, mirna, mt]
    omics = reduce(lambda left, right : pd.merge(left = left, right = right, how = "inner", left_index = True, right_index = True), join_list)
    
    # encoded pheno
    omics = pd.merge(left=pheno, right=omics, how="inner", left_index=True, right_index=True)
    
    # list to dict
    omics_label = [omics, rna, mirna, mt]
    data_type = ["omics", "rna", "mirna", "mt"]
    omics_group = list(map(lambda df : pd.merge(left=group, right=df, how="inner", 
                                          left_index=True, right_index=True), omics_label))
    zipbObj = zip(data_type, omics_group)
    omics = dict(zipbObj)
    
    return omics

def select_top_n(X, y, cv, method_, univariate):    
    # Pipeline
    model = svm.SVC(kernel='linear')
    
    if univariate:
        # grid search
        fs = SelectKBest(score_func=method_)
        pipeline = Pipeline(steps=[('method',fs), ('lr', model)])
        grid = dict()
        
        if len(X.columns) < 1000 :
            grid['method__k'] = list(range(100, len(X.columns), 100))
        else:
            grid['method__k'] = list(range(100, 1100, 100))
    else :
        # grid search
        fs = SelectFromModel(estimator=method_)
        pipeline = Pipeline(steps=[('method',fs), ('lr', model)])
        grid = dict()
        
        if len(X.columns) < 1000 :
            grid['method__max_features'] = list(range(100, len(X.columns), 100))
        else:
            grid['method__max_features'] = list(range(100, 1100, 100))

    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring='f1_micro', n_jobs=-1, cv=cv)
    results = search.fit(X, y)
    
    return results

def select_features(X, y, N, method):   
    fs = SelectKBest(score_func=method, k= N)
    fs.fit(X, y)
    
    # transform train input data
    X_fs = fs.transform(X)
    
    return X_fs, fs

def select_features_ml(X, y, N, method):   
    fs = SelectFromModel(estimator=method, max_features=N)
    fs.fit(X, y)
    
    # transform train input data
    X_fs = fs.transform(X)
    
    return X_fs, fs

def Feature_selection(group, feature, method, univariate):
    '''
        @group - target variable
        @feautre - feature
        @method - annovar -> f_classif / mutual -> mutual_info_classif / rf -> RandomForest / xg -> XGboost / 
    '''
    if univariate:
        if method == "anova":
            method_ = f_classif
        else : # mutal
            method_ = mutual_info_classif

        # Select Top - N using cv(cross-validation)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=331)

        # grid search
        gird_result = select_top_n(feature, group, cv, method_, univariate=True)

        # Best Top-K
        N = gird_result.best_params_['method__k']
        B = gird_result.best_score_*100

        #Select N feature
        #feature selection : f_classif, mutual_info_classif
        feature_fs, fs = select_features(feature, group, N, method_)

        # result DF
        result_df = pd.DataFrame(fs.scores_, columns=[method])
        result_df['feature'] = fs.feature_names_in_
        result_df = result_df.sort_values(by = [method], axis=0, ascending=False).iloc[1:N,:]

        return result_df.reset_index(drop=True), N, B
    
    else :
        if method == "rf":
            if len(feature.columns) > 10000:
                method_ = RandomForestClassifier(n_estimators = 500)
            else:
                method_ = RandomForestClassifier()
        else : 
            method_ = XGBClassifier()

        # Select Top - N using cv(cross-validation)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=331)

        # grid search
        gird_result = select_top_n(feature, group, cv, method_, univariate=False)
        
        # Best Top-K
        N = gird_result.best_params_['method__max_features']
        B = gird_result.best_score_*100

        # Select N feature
        feature_fs, fs = select_features_ml(feature, group, N, method_)

        # result DF
        result_df = pd.DataFrame(fs.get_feature_names_out(), columns=[method])

        # feature DF, ACC, 
        return result_df.reset_index(drop=True), N, B
    
def feature_selection_svm(data_type, o):
    feature_result = dict()   
    for d_type in data_type:           
        anova = Feature_selection(group=o[d_type].loc[:, "group"], 
                                  feature=o[d_type].loc[:, o[d_type].columns != "group"],
                                  method="anova", univariate=True)
        
        rf = Feature_selection(group=o[d_type].loc[:, "group"], 
                               feature=o[d_type].loc[:, o[d_type].columns != "group"],
                               method="rf", univariate=False)

        feature_result[d_type] = [anova, rf]
        
    return feature_result

@retry(delay=3)
def deg_extract(log_fc, fdr, method, cancer_type, sample_group, deg_path, file_name, rdata_path, batch_removal, raw_path):
    r = ro.r
    r['source']('src/r-function.R')

    group_reverse = group_convert(sample_group, raw_path)
    
    # R DF to pandas DF
    Path(deg_path).mkdir(parents=True, exist_ok=True)
    Path(deg_path+cancer_type).mkdir(parents=True, exist_ok=True)
    Path(deg_path+cancer_type+"_volcano").mkdir(parents=True, exist_ok=True)
    Path(rdata_path).mkdir(parents=True, exist_ok=True)
    
    ## EdgeR
    if method == "edger" or method == "all":
        run_edgeR_r = ro.globalenv['run_edgeR']
        edger = run_edgeR_r(cancer_type, sample_group, rdata_path, group_reverse)
        with localconverter(ro.default_converter + pandas2ri.converter):
            edger = ro.conversion.rpy2py(edger)
        
        # DEG list
        edger_filter = ((edger.logFC <= -(log_fc)) | (edger.logFC >= log_fc)) & (edger.FDR < fdr)
        edger = edger.loc[edger_filter, :]
        edger.to_csv(deg_path + cancer_type + "/" + cancer_type + "_EDGER_" + file_name + ".txt", sep = "\t", index = False)
    
    if method == "deseq2" or method == "all":
    ## Deseq2
        run_deseq_r = ro.globalenv['run_deseq']
        deseq = run_deseq_r(cancer_type, sample_group, rdata_path, group_reverse, file_name, deg_path, batch_removal)
        with localconverter(ro.default_converter + pandas2ri.converter):
            deseq = ro.conversion.rpy2py(deseq)
        
        # DEG list
        # deseq_filter = ((deseq.log2FoldChange <= -(log_fc)) | (deseq.log2FoldChange >= log_fc)) & (deseq.padj < fdr)
        # deseq = deseq.loc[deseq_filter, :]
        deseq.to_csv(deg_path + cancer_type + "/" + cancer_type + "_DESEQ2_" + file_name + ".txt", sep = "\t", index = False)
    
    if method == "deseq2":
        return deseq
    elif method == "edger":
        return edger
    else:
        return [edger, deseq]
    
def deg_extract_normal(log_fc, pvalue, cancer_type, rdata_path, deg_path, batch_removal):
    r = ro.r
    r['source']('src/r-function.R')
    run_deseq2_normal_r = ro.globalenv['run_deseq_normal']
    
    # R DF to pandas DF
    Path(rdata_path).mkdir(parents=True, exist_ok=True)
    
    deseq = run_deseq2_normal_r(cancer_type, rdata_path, deg_path, batch_removal)
    with localconverter(ro.default_converter + pandas2ri.converter):
        deseq = ro.conversion.rpy2py(deseq)

    # DEG list
    deseq_filter = ((deseq.log2FoldChange <= -(log_fc)) | (deseq.log2FoldChange >= log_fc)) & (deseq.pvalue < pvalue)
    deseq = deseq.loc[deseq_filter, :]
    
    return deseq

# Analysis
def deseq2_edger_combine(df):
    e = df[0]
    d = df[1]
    
    e_col = ["gene", "EdgeR-logFC"]
    d_col = ["gene", "Deseq2-logFC"]
    
    e = e[["mRNA", "logFC"]]
    d = d[["row", "log2FoldChange"]]
    # rename column
    e.columns = e_col
    d.columns = d_col
    
    return pd.merge(e, d, left_on='gene', right_on='gene', how='outer')

def col_rename(df, num, bs):
    change_col = df.columns.to_list()[1:]
    change_col = ["gene"] + ["SubGroup-" + bs[num] + "_" + value for value in change_col]
    df.columns = change_col
    
    return df

def db_query(x):   
    q1 = pd.read_sql_table(table_name=x, con=engine_mariadb)
    q1.columns = ['gene', x + '_TYPE', x + '_SUPPORT', x + '_CONFIDENCE', x + '_LIFT', x + '_COUNT']
    return q1

def dgidb_extract(gene_list, parallel=None):
    r = ro.r
    r['source']('src/r-function.R')
    if parallel:
      dgidb_r = ro.globalenv['dgidb_interaction_parallel']
    else:
      dgidb_r = ro.globalenv['dgidb_interaction']

    
    dgidb_result = dgidb_r(gene_list)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        dgidb_result = ro.conversion.rpy2py(dgidb_result)
        
    return dgidb_result

def proteinAtlas_extract(df):
    # pandas DF to R DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        sybmol_df = ro.conversion.py2rpy(df)

    r = ro.r
    r['source']('src/r-function.R')
    symbol_mapping = ro.globalenv['protein_atlas']
    symbol_df = symbol_mapping(sybmol_df)

    # R DF to pandas DF
    with localconverter(ro.default_converter + pandas2ri.converter):
        symbol_df = ro.conversion.rpy2py(symbol_df)
        
    return symbol_df

@retry(delay=1)
def textmining_extract(query_types):
  tm_result = reduce(lambda q1, q2 : pd.merge(left = q1, right = q2, on="gene", how='outer'), map(db_query, query_types))
  return tm_result

def tmb_t_test(group, raw_path):
  r = ro.r
  r['source']('src/r-function.R')
  tmb_calc = ro.globalenv["tmb_calculation"]

  # pd to r df
  with localconverter(ro.default_converter + pandas2ri.converter):
    group = ro.conversion.py2rpy(group.reset_index(level=0))

  t_test_result = tmb_calc(group, raw_path)

  return t_test_result[0]

# Uniprot, PDB ID Search
POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"
tsv_loader = "?compressed=true&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Corganism_name&format=tsv"

retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, taxid, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "taxId": taxid, "ids": ",".join(ids)},
    )
    check_response(request)
    return request.json()["jobId"]


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(request["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text


def get_xml_namespace(element):
    m = re.match(r"\{(.*)\}", element.tag)
    return m.groups()[0] if m else ""


def merge_xml_results(xml_results):
    merged_root = ElementTree.fromstring(xml_results[0])
    for result in xml_results[1:]:
        root = ElementTree.fromstring(result)
        for child in root.findall("{http://uniprot.org/uniprot}entry"):
            merged_root.insert(-1, child)
    ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
    return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")

def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")
    request = session.get(url)
    check_response(request)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)

def mergeDictionary(dict_):
    dict_3 = defaultdict(list)
    for d in dict_: # you can list as many input dicts as you want here
        for key, value in d.items():
            dict_3[key].append(value)
            # list(itertools.chain(*list_of_lists))
    return dict_3

@retry(delay=6)
def get_pdb_structure_id(gene_list):
    # Gene to Uniprot
    job_id = submit_id_mapping(
        from_db="Gene_Name", to_db="UniProtKB", taxid=9606, ids=gene_list
    )
    
    print(job_id)
    
    link = get_id_mapping_results_link(job_id)
    mapping_result = get_id_mapping_results_stream(link + tsv_loader)
    mapping_result = list(map(lambda x : x.split("\t"), mapping_result))
    mapping_result_df = pd.DataFrame(mapping_result[1:], columns=mapping_result[0])
    m_v = mapping_result_df[mapping_result_df['Reviewed'] == "reviewed"]

    # Uniprot to PDB
    m_v['pdb'] = m_v.apply(lambda x : Query(x['Entry']).search(), axis=1)
    # m_v['pdbIDCount'] = m_v.apply(lambda x : 0 if x['pdb'] is None else int(len(x['pdb'])), axis=1)
    m_v['pdbID'] = m_v.apply(lambda x : None if x['pdb'] is None else ';'.join(x['pdb']), axis=1)
    m_v = m_v[['From', 'Entry', 'Reviewed', 'pdbID']]
    m_v.columns = ['GENE_NAME', 'UniprotKB', 'Reviewed', 'pdbID']
    
    # post processing
    pdbID_collapse = m_v.groupby('GENE_NAME').apply(lambda x : ';'.join(filter(None, x['pdbID'])))
    pdbID_collapse.name = "pdbID"

    UniprotKB_collapse = m_v.groupby('GENE_NAME').apply(lambda x : ';'.join(filter(None, x['UniprotKB'])))
    UniprotKB_collapse.name = "UniprotKB"

    pdb_collapse = pd.merge(UniprotKB_collapse, pdbID_collapse, right_index = True,
                   left_index = True)

    pdb_collapse['pdbID'] = pdb_collapse.apply(lambda x : x['pdbID'].split(';'), axis=1)

    pdb_collapse['pdbCount'] = pdb_collapse.apply(lambda x : len(set(x['pdbID'])) - 1, axis=1)

    pdb_collapse['pdbID'] = pdb_collapse.apply(lambda x : ';'.join(set(x['pdbID'])),axis = 1)
    
    return pdb_collapse

def query_mariadb(db, query):
    # Connect to MariaDB (mariadb.connect 대신 pymysql.connect 로 해도 된다)
    dbconn = pymysql.connect(
        user="root",
        password="sempre813!",
        host="192.168.0.91",
        port=3306,
        database=db
    )
 
    # dbconn = mydb.cursor()  # 이 명령어는 불필요.
    # mariaDB Query to Pandas DataFrame
    query_result= pd.read_sql(query,dbconn)
    dbconn.close()
 
    return query_result

# mygene api
def symbol2pdb(gene_list):
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    params = 'q='+ ','.join(gene_list)+' &scopes=symbol&fields=taxid,pdb'
    res = requests.post('http://mygene.info/v3/query', data=params, headers=headers)
    out = pd.DataFrame(json.loads(codecs.decode(bytes(res.text, 'utf-8'), 'utf-8-sig')))
    out = out[out.taxid == 9606]
    out = out[out.pdb.notna()]
    out = out.loc[:, ['query', 'pdb']]

    out['pdbCount'] = out.apply(lambda x : len(x['pdb']) if isinstance(x['pdb'], list) else 1, axis=1)
    out['pdb'] = out.apply(lambda x : ';'.join(x['pdb']) if isinstance(x['pdb'], list) else x['pdb'],axis = 1)
    
    return out

# OncoKB curated gene
def oncokb_allcuratedGenes():

    ONCOKB_TOKEN = query_mariadb(db="TOKEN", query="SELECT * FROM ONCOKB").TOKEN.to_list()[0]

    proc = subprocess.run(["curl",  "-X", "GET",  
                           'https://www.oncokb.org/api/v1/utils/allCuratedGenes.txt?includeEvidence=true',
                           '-H', 'accept: application/json',
                           '-H', 'Authorization: Bearer '+ ONCOKB_TOKEN[0]

                          ],
                       stdout=subprocess.PIPE, encoding='utf-8')

    oncokb_curated = proc.stdout
    oncokb_curated_df = pd.DataFrame(list(map(lambda x: x.split("\t"), oncokb_curated.split('\n'))))
    oncokb_curated_df.columns = oncokb_curated_df.iloc[0]
    oncokb_curated_df.drop(oncokb_curated_df.index[0], inplace=True)
    oncokb_curated_df = oncokb_curated_df.loc[:, ['Hugo Symbol', 'Is Oncogene', 'Is Tumor Suppressor Gene', 'Highest Level of Evidence(sensitivity)',
           'Highest Level of Evidence(resistance)', 'Background']]
    oncokb_curated_df.columns = ['gene'] + ["OncoKB_" + value for value in oncokb_curated_df.columns[1:]]
    
    return oncokb_curated_df

def norm_function(omics, norm, minmax):
    omics_index = omics.index.to_list()

    # normalization
    if norm:
        if minmax:
            scalerX = MinMaxScaler()
            omics_scale = scalerX.fit_transform(omics)
        else :
            scalerX = StandardScaler()      
            omics_scale = scalerX.fit_transform(omics)
    
    return omics_scale

def impute_function(omics):
    omics_index = omics.index.to_list()
    
    imputer = KNNImputer(n_neighbors=10)
    omics_impute = imputer.fit_transform(omics)

    omics = pd.DataFrame(omics_impute, columns=omics.columns)
    omics.index = omics_index
    
    return omics

if __name__ == "__main__": 
    print("not main")
    