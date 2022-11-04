"""
example : 
conda activate multiomics-cpu
python src/Multi-omics-integration-subgroup.py \
    -b /home/wmbio/WORK/gitworking/Multi-omics-intergration/ \
    -c BRCA -e 100

@author: Jinwoo Lee

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from detectionFunction import * 
import venn
import argparse

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Subgroup Detection!')
    parser.add_argument('-b', '--base', required=True, type=str, help='Root Path')
    parser.add_argument('-c', '--cancer', required=True, type=str, help='Types of cancer')
    parser.add_argument('-e', '--cycle', default=100, type=int, help='configuration of the name of output without a filename extension')
    parser.add_argument('-n', '--threads', default=16, type=int, help='parallelism threads')
    static_args = parser.parse_args()

    # file path
    os.chdir(static_args.base)
    CANCER_TYPE = static_args.cancer
    CYCLE = int(static_args.cycle)
    THREAD = int(static_args.threads)

    RAW_file_path = os.getcwd() + "/RAW_DATA/"
    PKL_file_path = os.getcwd() + "/pkl/"
    MODEL_PATH = os.getcwd() + "/models/"
    GROUP_PHTH = os.getcwd() + '/group/'
    PNG_PATH = os.getcwd() + '/png/'
    GROUP_VALIDATION_PATH = os.getcwd() + '/group_validation/'
    DEG_PATH = os.getcwd() + "/deg/"

    # Data-load
    omics = load_tcga_dataset_version1(pkl_path=PKL_file_path, raw_path=RAW_file_path, cancer_type=CANCER_TYPE, norm=True)
    X_train, X_test = train_test_split(omics, test_size = .2, random_state = 21, shuffle=True)


    for _ in range(CYCLE):
      try:
        log_pvalue_l, tmb_pvalue_l, silhouette_score_l, rna_anovar_f1, rna_rf_f1 = [], [], [], [], []
        mirna_anovar_f1, mirna_rf_f1, mt_anovar_f1, mt_rf_f1 = [], [], [], []
        file_name = []
        
        FILE_NAME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #SAMPLE_GROUP = GROUP_PHTH + CANCER_TYPE + "_GROUP_" + FILE_NAME + ".txt"
        file_name.append(FILE_NAME)
        print(FILE_NAME)
        
        ## AE(vanilla, sparse, denoisy) - Model compile & Fit
        encoder_vanilla = run_ae(X_train=X_train, X_test=X_test)
        encoder_sparse = run_ae_sparse(X_train=X_train, X_test=X_test)
        encoder_denoisy = run_ae_denoisy(X_train=X_train, X_test=X_test)

        group, silhouette_score = best_ae_model(model_list=[encoder_vanilla, encoder_sparse, encoder_denoisy], o=omics,
                    group_path=GROUP_PHTH, model_path=MODEL_PATH, cancer_type=CANCER_TYPE, raw_path=RAW_file_path, file_name=FILE_NAME)

        ## Sub-group Evalutation
        ### load preprocess data
        omics_preprocess = load_preprocess_tcga_dataset(pkl_path=PKL_file_path, raw_path=RAW_file_path, group=group, norm=True, 
                                                    cancer_type=CANCER_TYPE)

        ### Feature Selection(Anova, RandomForest) for SVM
        feature_result = feature_selection_svm(data_type=["rna", "mirna", "mt"], o=omics_preprocess)

        ### Survival Analysis - logranktest
        log_pvalue = log_rank_test_py(df=omics_preprocess["omics"].iloc[:, :3], png_path=PNG_PATH, cancer_type = CANCER_TYPE,file_name=FILE_NAME)
        tmb_pvalue = tmb_t_test(group, RAW_file_path)

        ### Score
        log_pvalue_l.append(log_pvalue)

        tmb_pvalue_l.append(tmb_pvalue)

        silhouette_score_l.append(silhouette_score)

        rna_anovar_f1.append(feature_result["rna"][0][2])
        rna_rf_f1.append(feature_result["rna"][1][2])

        mirna_anovar_f1.append(feature_result["mirna"][0][2])
        mirna_rf_f1.append(feature_result["mirna"][1][2])

        mt_anovar_f1.append(feature_result["mt"][0][2])
        mt_rf_f1.append(feature_result["mt"][1][2])
        
        # session clear
        gc.collect()
        
        # Write Score DF
        score_df = pd.DataFrame({
        'FILENAME' : file_name,
        'Log Rank Test' : log_pvalue_l,
        'TMB T-Test' : tmb_pvalue_l, 
        'Silhouette' : silhouette_score_l,
        'RNA_ANOVA_F1' : rna_anovar_f1,
        'RNA_RF_F1' : rna_rf_f1,
        'miRNA_ANOVA_F1' : mirna_anovar_f1,
        'miRNA_RF_F1' : mirna_rf_f1,
        'Methylation_ANOVA_F1' : mt_anovar_f1,
        'Methylation_RF_F1' : mt_rf_f1
        })
        
        print('Silhouette score : {0}\nCox-ph Log Rank Test : {1}\nTMB T-test : {2}'.format(silhouette_score_l, log_pvalue_l, tmb_pvalue_l))
      
        # score table
        if not os.path.exists(GROUP_VALIDATION_PATH + CANCER_TYPE + "_validation.csv"):
            Path(GROUP_VALIDATION_PATH).mkdir(parents=True, exist_ok=True)
            with open(GROUP_VALIDATION_PATH + CANCER_TYPE + "_validation.csv",'w') as file :
                write = csv.writer(file)
                write.writerow(score_df.columns.to_list())    
        score_df.to_csv(GROUP_VALIDATION_PATH + CANCER_TYPE + "_validation.csv", index=False, mode='a', header=False)  

      except Exception as e:
        print("Skip this epoke : ", e)