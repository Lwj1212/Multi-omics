"""
example : python src/Multi-omics-integration-analysis.py \
         -b /home/wmbio/WORK/gitworking/Multi-omics-intergration/ \
         -c PAAD

@author: Jinwoo Lee
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from analysisFunction import * 
import venn
import argparse

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Subgroup Analysis!')
    parser.add_argument('-b', '--base', required=True, type=str, help='Root Path')
    parser.add_argument('-c', '--cancer', required=True, type=str, help='Types of cancer')
    parser.add_argument('-a', '--cancer2', default=None, type=str, help='Types of cancer2')
    parser.add_argument('-d', '--dea', default="deseq2", type=str, help='DESeq2(deseq2) or EdgeR(edger) or ALL(all)')
    parser.add_argument('-l', '--logfc', default=1, type=float, help='log2FC Threshold')
    parser.add_argument('-f', '--fdr', default=0.05, type=float, help='False dicovery rate Threshold')
    parser.add_argument('-r', '--seed', default=331, type=int, help='Random Seed')
    static_args = parser.parse_args()

    # file path
    os.chdir(static_args.base)
    CANCER_TYPE = static_args.cancer
    CANCER_TYPE2 = static_args.cancer2
    METHOD = static_args.dea
    LOGFC = static_args.logfc
    FDR = static_args.fdr
    RANDOM_SEED = static_args.seed

    GROUP_PHTH = os.getcwd() + '/group/'
    PNG_PATH = os.getcwd() + '/png/'
    GROUP_VALIDATION_PATH = os.getcwd() + '/group_validation/'
    DEG_PATH = os.getcwd() + "/best_deg/"
    RDATA_PATH = os.getcwd() + "/RAW_DATA/GDC_PREPARE/"
    RAW_PATH = os.getcwd() + "/RAW_DATA/"

    # Load Validation score
    col=['FILENAME','Log Rank Test', 'TMB T-Test', 'Silhouette','RNA_ANOVA_F1','RNA_RF_F1',
        'miRNA_ANOVA_F1','miRNA_RF_F1','Methylation_ANOVA_F1','Methylation_RF_F1']

    group_score = pd.read_csv(GROUP_VALIDATION_PATH + CANCER_TYPE + '_validation.csv', usecols=col)
    
    # Q3 value
    SILHOUETTE = group_score.Silhouette.quantile(.5)
    RNA_ANOVA = group_score.RNA_ANOVA_F1.quantile(.7)
    RNA_RF = group_score.RNA_RF_F1.quantile(.7)
    MIRNA_ANOVAR = group_score.miRNA_ANOVA_F1.quantile(.7)
    MIRNA_RF = group_score.miRNA_RF_F1.quantile(.7)
    MT_ANOVAR = group_score.Methylation_ANOVA_F1.quantile(.7)
    MT_RF = group_score.Methylation_RF_F1.quantile(.7)

    # stdout
    print("SILHOUETTE Q2 : ", SILHOUETTE)
    print("RNA_ANOVA Q3 : ", RNA_ANOVA)
    print("RNA_RF Q3 : ", RNA_RF)
    print("MIRNA_ANOVAR Q3 : ", MIRNA_ANOVAR)
    print("MIRNA_RF Q3 : ", MIRNA_RF)
    print("MT_ANOVAR Q3 : ", MT_ANOVAR)
    print("MT_RF Q3 : ", MT_RF)
 
    # Condition for Filtering
    filter_cond = (group_score['Silhouette'] >= SILHOUETTE) & (group_score['Log Rank Test'] < 0.05) & (group_score['TMB T-Test'] < 0.05) & \
              ((group_score['RNA_ANOVA_F1'] >= RNA_ANOVA) | (group_score['RNA_RF_F1'] >= RNA_RF)) & \
              ((group_score['miRNA_ANOVA_F1'] >= MIRNA_ANOVAR) | (group_score['miRNA_RF_F1'] >= MIRNA_RF)) & \
              ((group_score['Methylation_ANOVA_F1'] >= MT_ANOVAR) | (group_score['Methylation_RF_F1'] >= MT_RF))
              
    group_score = group_score[filter_cond].sort_values(["Silhouette"], ascending = (False))
    bestSubgroup = group_score.FILENAME.to_list()

    # random 추출
    if len(bestSubgroup) >= 100:
      random.seed(RANDOM_SEED)
      bestSubgroup = random.sample(bestSubgroup, k=100)
    print("SubGroup count : ", len(bestSubgroup))

    # DEA
    dea_result = list()
    for best_group in bestSubgroup:
        
        DEG_CHECK = "_".join([CANCER_TYPE, METHOD.upper(), best_group]) + ".txt"
        SAMPLE_GROUP = GROUP_PHTH + CANCER_TYPE + "/" + CANCER_TYPE + "_GROUP_" + best_group + ".txt"
        
        if os.path.isfile(DEG_PATH + CANCER_TYPE + "/" + DEG_CHECK):
            deg_list = pd.read_csv(DEG_PATH + CANCER_TYPE + "/" + DEG_CHECK, sep = "\t")

            # cut-off
            deseq_filter = ((deg_list.log2FoldChange <= -(LOGFC)) | (deg_list.log2FoldChange >= LOGFC)) & (deg_list.padj < FDR)
            deg_list = deg_list.loc[deseq_filter, :]
        else :
            # DEG Extraction
            deg_list = deg_extract(log_fc=LOGFC, fdr=FDR,
                          cancer_type=CANCER_TYPE, 
                          sample_group=SAMPLE_GROUP, deg_path=DEG_PATH, 
                          file_name=best_group,
                          rdata_path=RDATA_PATH,
                          method=METHOD,
                          batch_removal=True,
                          raw_path=RAW_PATH)
            # cut-off
            deseq_filter = ((deg_list.log2FoldChange <= -(LOGFC)) | (deg_list.log2FoldChange >= LOGFC)) & (deg_list.padj < FDR)
            deg_list = deg_list.loc[deseq_filter, :]

        dea_result.append(deg_list)
        gc.collect()

    # Filter DEA
    # combine result
    if METHOD == 'all':
        dea_combine = list(map(deseq2_edger_combine, dea_result))
        dea_combine = [col_rename(dea_combine[index], index, bestSubgroup) for index in range(len(dea_combine))]
        dea_combine = reduce(lambda left, right : pd.merge(left, right, on='gene', how = 'outer'), dea_combine)
    elif METHOD == 'deseq2' :
        dea_combine = list(map(lambda d : d[["row", "log2FoldChange", "padj"]], dea_result))
        dea_combine = [col_rename(dea_combine[index], index, bestSubgroup) for index in range(len(dea_combine))]
        dea_combine = reduce(lambda left, right : pd.merge(left, right, on='gene', how = 'outer'), dea_combine)
    print("Multiple DEA Combine Finished!")

    # blank row calculation
    blank_row = dea_combine.loc[:, dea_combine.columns.str.contains("[0-9]_log2FoldChange")].isnull().sum(axis=1) # serise
    dea_combine['1-blank_ratio'] = blank_row.apply(lambda x : ((1 - (x / len(bestSubgroup))) * 100))
    print("Blank ratio calculation Combine Finished!")

    # log2FCmedian & mean
    dea_combine["SubGroup-log2FC_median"] = dea_combine.loc[:, dea_combine.columns.to_series().str.endswith("_log2FoldChange")].median(axis=1)
    dea_combine["SubGroup-log2FC_mean"] = dea_combine.loc[:,  dea_combine.columns.to_series().str.endswith("_log2FoldChange")].mean(axis=1)

    # NT vs TP DEA
    # log fc, FDR value 수정
    nt_tp_deseq2 = deg_extract_normal(log_fc=0, pvalue=0.1, cancer_type=CANCER_TYPE, 
                                  rdata_path=RDATA_PATH, deg_path=DEG_PATH, batch_removal=True)

    nt_tp_deseq2_col = nt_tp_deseq2[['row', 'log2FoldChange', 'pvalue']]
    nt_tp_deseq2_col.columns = ['gene', 'NT-TP_Log2FC', 'NT-TP.Padj']

    result_combine = pd.merge(left=dea_combine, right=nt_tp_deseq2_col, on='gene', how = 'left')   
    print("NT-TP DEA Combine Finished!")

    # Textmining  
    if CANCER_TYPE2 is not None:
      query_types = [CANCER_TYPE] + [CANCER_TYPE2]
    else:
      query_types = [CANCER_TYPE]

    # tm_df = reduce(lambda q1, q2 : pd.merge(left = q1, right = q2, on="gene", how='outer'), map(db_query, query_types))
    tm_df = textmining_extract(query_types=query_types)
    tm_df.columns = ['gene', 'TM.type', 'TM.Support', 'TM.Confidence', 'TM.Lift', 'TM.Count']
    result_combine_tm = pd.merge(left=result_combine, right=tm_df, on="gene", how='left')
    print("Textmining Combine Finished!")
    
    # ELMER
    elmer_rs = elmer(df=result_combine_tm, cancer_type=CANCER_TYPE, base_dir=os.getcwd())
    result_combine_elmer = pd.merge(left=result_combine_tm, right=elmer_rs, left_on="gene", right_on="gene", how='left')
    print("Methylation analysis Finished!!")

    # DGIdb
    gene_list = result_combine_elmer.loc[:, 'gene'].to_list()
    dgidb_df = dgidb_extract(gene_list, True)
    dgidb_df.columns = ['gene', 'DGI.Gene Category', 'DGI.DrugName;Score;Type', 'DGI.Count']
    result_combine_dgidb = pd.merge(left=result_combine_elmer, right=dgidb_df, on='gene', how='left')
    print("DGIdb Combine Finished!")

    # PDBid
    gene_list = result_combine_dgidb.loc[:, 'gene'].to_list()
    pdb_df = symbol2pdb(gene_list)
    pdb_df.columns = ['gene', 'PDB.Id', 'PDB.Count']
    result_combine_pdbid = pd.merge(left=result_combine_dgidb, right=pdb_df, on='gene', how='left')
    print("PDBid Combine Finished!")

    # OncoKB
    oncokb_curated_df = oncokb_allcuratedGenes()
    oncokb_curated_df.columns = ['gene', 'OncoKB.Is Oncogene', 'OncoKB.Is TSG', 
                      'OncoKB.Highest Level of Evidence(sensitivity)',
                      'OncoKB.Highest Level of Evidence(resistance)',
                      'OncoKB.Background']
    result_combine_oncokb = pd.merge(left=result_combine_pdbid, right=oncokb_curated_df, on='gene', how = 'left')
    print("OncoKB curated Gene Combine finished!")

    # Protein Atlas
    gene_list = pd.DataFrame(result_combine_pdbid.loc[:, 'gene'])
    result_pa = proteinAtlas_extract(gene_list)
    result_combine_proteinatlas = pd.merge(left=result_combine_oncokb, right=result_pa, on='gene', how = 'left')
    print("Protein Atlas Combine finished!")

    # Result write
    Path(os.getcwd() + "/RESULT").mkdir(parents=True, exist_ok=True)
    Path(os.getcwd() + "/RESULT/" + CANCER_TYPE).mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # sort
    result_combine_proteinatlas.sort_values(by = ['gene'], axis = 0).to_csv(os.getcwd() + "/RESULT/" + CANCER_TYPE + "/" + CANCER_TYPE + '-RandomSeed-'+ str(RANDOM_SEED) + '-' + time_stamp + '.csv', index = False)
