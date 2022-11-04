# library load
suppressPackageStartupMessages({
  library(survival)
  library(ranger)
  library(NbClust)
  library(survminer)
  library(tidyverse)
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(DESeq2)
  library(BiocParallel)
  library(EnhancedVolcano)
  library(sva)
  library(httr)
  library(parallel)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(ELMER)
  library(ELMER.data)
})

survFit <- function(sample_group_path, raw_path){
  suppressMessages({
    sample_group <- read_delim(file = sample_group_path, delim = "\t", show_col_types = FALSE, progress = FALSE)
    
    # surv data
    pheno <- read_delim(paste0(raw_path, "/Survival_SupplementalTable_S1_20171025_xena_sp"), 
                        col_select = c('sample', 'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'),
                        delim = "\t", show_col_types = FALSE, progress = FALSE)
    sample_group_surv <- left_join(x = sample_group, y = pheno, by = "sample")
    
    fit <- survfit(Surv(time = OS.time, event = OS) ~ group, data = sample_group_surv)
  })
    summary(fit)$table %>% as.data.frame() %>%
      return()
  
}  

# log-rank test
log_rank_test <- function(df){   
  suppressMessages({
    # column name extraction
    df_name <- colnames(df)
    df_name_filter <- df_name[str_detect(string = df_name, pattern = "Feature")]
    
    # log_rank test
    df_log_rank <- lapply(X = df_name_filter, function(col_name){
      cox <- coxph(
        formula = as.formula(paste0("Surv(time = OS.time, event = OS) ~", col_name)), 
        data = df)
      cox_result <- summary(cox)
      tibble(Features = col_name, log_p_value = cox_result$logtest["pvalue"]) %>% return()
    }) %>% bind_rows()
  })
    # p-value 0.05 cuttoff
    df_log_rank %>% 
      filter(log_p_value < 0.05) %>% 
      return()
  
}

# log rank test
log_rank_test_group <- function(group_sample){

  suppressMessages({
    group_sample <- as_tibble(group_sample)
    survdiff_result <- survdiff(Surv(group_sample$OS.time, group_sample$OS) ~ group_sample$group)
    surv_result <- survfit(Surv(group_sample$OS.time, group_sample$OS) ~ group_sample$group)
    # print(group_sample)
    # t <- ggsurvplot(surv_result, 
    #                 data = group_sample, 
    #                 conf.int = TRUE, 
    #                 risk.table.col = 'strata',
    #                 ggtheme = theme_bw(),
    #                 palette = c("#E7B800", "#2E9FDF"),
    #                 pval = TRUE)
    # ggsave(t, file = paste0(png_path, "/", cancer_type, "/", file_name, "_logrank.png"))
    log_rank_test_p <- pchisq(survdiff_result$chisq, df = 1, lower.tail = F)
  })

  return(log_rank_test_p)
}


# random_forest test
nb_cluster_test <- function(df){ 
  suppressMessages({
    nc <- NbClust(df,min.nc=2,max.nc=9,method="kmeans", index = c("silhouette","cindex"))
    nc$All.index %>% as_tibble() %>% 
    dplyr::select("Silhouette") %>% return()
  })
}

# DGIdb
dgidb_interaction_parallel <- function(gene_name){
  base_url <- "https://dgidb.org"
  request_url <- paste0(base_url, "/api/v2/interactions.json?")
  result_list <- list()
  
  # chunk id
  id_chunk <- split(gene_name, ceiling(seq_along(gene_name)/200))
  
  mclapply(X = 1:length(id_chunk), FUN = function(index){
    # print(index)
    payload <-  list(genes = paste0(id_chunk[[index]], collapse = ","),
                     fda_approved_drug="true")
    
    # output
    dgidb_result <- POST(request_url, body = payload, encode = "form", config = httr::config(connecttimeout = 60)) %>%  
      httr::content(encoding = "UTF-8") 
    
    lapply(X = dgidb_result$matchedTerms, FUN = function(dgidb_element){
      gene_category <- dgidb_element$geneCategories %>% 
        sapply(X = ., FUN = function(value) {value$name}) %>% 
        paste0(collapse = ",")
      
      interaction <- dgidb_element$interactions %>% 
        sapply(X = ., FUN = function(value){
          drug_name <- value$drugName
          score <- value$score
          types <- value$interactionTypes %>% unlist() %>% paste0(collapse = "&")
          
          paste0(c(drug_name, score, types), collapse = ";") %>% 
            as_tibble() %>% 
            return()
          
          # return(drug_name)  
        }) %>% unlist() %>% 
        paste0(., collapse = "&")
      
      tibble(
        gene = dgidb_element$geneName,
        DGI_GENE_CATEGORY = gene_category, 
        `DGI(DRUG_NAME;SCORE;TYPE)` = interaction,
        DGI_COUNT = length(dgidb_element$interactions)
      )  %>% return()
      
    }) %>% return()
  }, mc.cores = 3) %>% bind_rows() %>% return()
}

dgidb_interaction <- function(gene_name){
  base_url <- "https://dgidb.org"
  request_url <- paste0(base_url, "/api/v2/interactions.json?")
  result_list <- list()
  
  # chunk id
  id_chunk <- split(gene_name, ceiling(seq_along(gene_name)/200))
  
  print(paste0("total chunk length : ", length(id_chunk)))
  for(index in 1:length(id_chunk)){
    print(index)
    # print(index)
    payload <-  list(genes = paste0(id_chunk[[index]], collapse = ","),
                     fda_approved_drug="true")
    
    # output
    dgidb_result <- POST(request_url, body = payload, encode = "form", config = httr::config(connecttimeout = 100)) %>%  
      httr::content(encoding = "UTF-8") 
  
    result_list[[index]] <- lapply(X = dgidb_result$matchedTerms, FUN = function(dgidb_element){
      gene_category <- dgidb_element$geneCategories %>% 
        sapply(X = ., FUN = function(value) {value$name}) %>% 
        paste0(collapse = ",")
      
      interaction <- dgidb_element$interactions %>% 
        sapply(X = ., FUN = function(value){
          drug_name <- value$drugName
          score <- value$score
          types <- value$interactionTypes %>% unlist() %>% paste0(collapse = "&")
          
          paste0(c(drug_name, score, types), collapse = ";") %>% 
            as_tibble() %>% 
            return()
          
          # return(drug_name)  
        }) %>% unlist() %>% 
        paste0(., collapse = "&")
      
      tibble(
        gene = dgidb_element$geneName,
        DGI_GENE_CATEGORY = gene_category, 
        `DGI(DRUG_NAME;SCORE;TYPE)` = interaction,
        DGI_COUNT = length(dgidb_element$interactions)
      )  %>% return()
      
    }) %>% bind_rows() 
  }
  
  result_list %>% bind_rows() %>% return()
}

protein_atlas <- function(DF){
  suppressMessages({
    protein_atlas_url <- "https://www.proteinatlas.org/"
    gene_list <- DF %>% 
      dplyr::pull(gene)
    gene_list_mapping <- mapIds(org.Hs.eg.db,
                                keys=gene_list, 
                                column="ENSEMBL",
                                keytype="SYMBOL",
                                multiVals="first") %>% 
      tibble(gene = names(.), ENSEMBL = .) %>% 
      mutate(PROTEIN_ATLAS = ifelse(!is.na(ENSEMBL), paste0(protein_atlas_url, ENSEMBL, "-", gene), "")) %>% 
      dplyr::select(gene, PROTEIN_ATLAS)
  })
  return(gene_list_mapping)
}

# tmb
tmb_calculation <- function(group, raw_path){
  suppressMessages({
    # maf_tcga <- maftools::read.maf(paste0(raw_path, "mc3.v0.2.8.PUBLIC.maf.gz"), verbose = 0)
    load(paste0(raw_path, "mc3.v0.2.8.PUBLIC.RData"))
    pancan_tmb <- maftools::tmb(maf = maf_tcga, captureSize = 38) %>%  
      separate(Tumor_Sample_Barcode, into = LETTERS[1:7], sep = "-") %>% 
      dplyr::filter(D == "01A") %>% 
      dplyr::select(-E, -F, -G) %>% 
      unite(col = sample, A,B,C,D, sep = "-") %>% 
      mutate(sample = substring(sample,1, nchar(sample)-1),
            total_perMB_log = log((total_perMB + 1), 2)) %>% 
      as_tibble()
      
    tmp <- left_join(x = group, y = pancan_tmb, by = "sample")
    t_test_result <- t.test(total_perMB_log ~ group, tmp) %>% .$p.value
  })
  return(t_test_result)
}

# function
# run_edgeR <- function(pr_name, sample_group_path, group_reverse){
  
#   suppressMessages({
#     sample_group <- read_delim(file = sample_group_path, delim = "\t", show_col_types = FALSE)
    
#     # group convert
#     if(group_reverse){sample_group <- sample_group %>% mutate(group = ifelse(group == 0, 1, 0))}
#     if((!file.exists(paste0(rdata_path, pr_name, ".RData"))) | 
#        (!file.exists(paste0(rdata_path, pr_name, "_RnaseqSE.RData")))){
#         query <- GDCquery(project = paste0("TCGA-", pr_name), 
#                           data.category = "Gene expression",
#                           data.type = "Gene expression quantification",
#                           experimental.strategy = "RNA-Seq",
#                           platform = "Illumina HiSeq",
#                           file.type = "results",
#                           barcode = sample_group %>% pull(1), 
#                           legacy = TRUE)

#         GDCdownload(query)
#         RnaseqSE <- GDCprepare(query)
        
#         save(RnaseqSE, file = paste0(rdata_path, pr_name, "_RnaseqSE.RData"))
        
#         Rnaseq_CorOutliers <- assay(RnaseqSE) # to matrix

#         # normalization of genes, # quantile filter of genes
#         dataNorm <- TCGAanalyze_Normalization(tabDF = Rnaseq_CorOutliers, geneInfo =  geneInfo)
#         dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
#                                           method = "quantile", 
#                                           qnt.cut =  0.25)
        
#         save(dataFilt, file = paste0(rdata_path, pr_name, ".RData"))
#     } else {
#         load(paste0(rdata_path, pr_name, "_RnaseqSE.RData"))
#         load(paste0(rdata_path, pr_name, ".RData"))
#     }
    
#     # subgroup names
#     sub0 <- sample_group %>% filter(group == 0) %>% pull(1)
#     sub1 <- sample_group %>% filter(group == 1) %>% pull(1)
    
#     sample_subgroup <- Rnaseq_CorOutliers %>% colnames() %>% lapply(X = ., FUN = function(value){
      
#       value_trans <- str_extract_all(value, pattern = "TCGA-[:alnum:]+-[:alnum:]+-[:digit:]+") %>%  unlist()
#       subgroup_df <- tibble(sample_barcode = NA, subgroup = NA, .rows = 0)
      
      
#       if(value_trans %in% sub0){
#         subgroup_df <- tibble(sample_barcode = value, subgroup = 0)
#       } else {
#         subgroup_df <- tibble(sample_barcode = value, subgroup = 1)
#       }
      
#       return(subgroup_df)
      
#     }) %>% bind_rows()
    
#     sub0_name <- sample_subgroup %>% filter(subgroup == 0) %>% pull(sample_barcode)
#     sub1_name <- sample_subgroup %>% filter(subgroup == 1) %>% pull(sample_barcode)
    
#     #Diff.expr.analysis (DEA)
#     dataDEGs <- TCGAanalyze_DEA(mat1 = dataFilt[,sub0_name],
#                                 mat2 = dataFilt[,sub1_name],
#                                 Cond1type = "Sub-0",
#                                 Cond2type = "Sub-1",
#                                 #                               fdr.cut = 0.01 ,
#                                 #                               logFC.cut = 1,
#                                 method = "glmLRT",
#                                 paired = F)
    
#     # DEGs table with expression values in normal and tumor samples
#     dataDEGsFiltLevel <- TCGAanalyze_LevelTab(dataDEGs,"Sub-0","Sub-1",
#                                               dataFilt[,sub0_name],dataFilt[,sub1_name])
#     #   write_delim(dataDEGsFiltLevel, file = paste0(dea_result_path, "_", pr_name, "_EDGER_", file_name, ".txt"), delim = "\t")   
#   })  
  
#   return(dataDEGsFiltLevel)
# }
# run_edgeR_pancan <- function(sample_group_path, involve_brca, group_reverse){
#   sample_group <- read_delim(file = sample_group_path, delim = "\t", show_col_types = FALSE)
    
#   # group convert
#   if(group_reverse){sample_group <- sample_group %>% mutate(group = ifelse(group == 0, 1, 0))}
    
#   study_list <- tcga_available()$Study_Abbreviation[-34] %>% 
#     lapply(X = ., FUN = function(value){
#       paste0("TCGA-", value) %>% return()
#     }) %>% as.character()
  
#   if(!involve_brca){
#     study_list <- study_list[study_list != 'TCGA-BRCA']
#   }
  
#   suppressMessages({
#     tcga_mrna <- mclapply(X = study_list, FUN = function(type){
      
#       error_occur <- FALSE
#       tryCatch(
#         expr = {
#           query <- GDCquery(project = type, 
#                             data.category = "Gene expression",
#                             data.type = "Gene expression quantification",
#                             experimental.strategy = "RNA-Seq",
#                             platform = "Illumina HiSeq",
#                             file.type = "results",
#                             barcode = sample_group %>% pull(1), 
#                             legacy = TRUE)
#           # Download a list of barcodes with platform IlluminaHiSeq_RNASeqV2
#           GDCdownload(query)
          
#           # rsem.genes.results as values
#           RnaseqSE <- GDCprepare(query)
          
#         },
#         error = function(e) { 
#           error_occur <<- TRUE
#         }
#       )
      
#       if(error_occur){
#         return(NULL)
#       } else {
#         RnaseqSE %>%
#           assay() %>% 
#           as.data.frame() %>% 
#           rownames_to_column(var = "gene") %>% return()
#       }
      
#     }, mc.cores = 20)
    
#     # inner join
#     reduce_join <- function(left, right){
#       if(is.null(right)){
#         return(left)
#       } else{
#         inner_join(left, right, by = "gene") %>% return()
#       }
#     }
#     tcga_mrna_join <- purrr::reduce(tcga_mrna, reduce_join) %>% 
#       column_to_rownames(var = "gene") %>% as.matrix()
    
#     # normalization of genes # ~ 10 min
#     dataNorm <- TCGAanalyze_Normalization(tabDF = tcga_mrna_join, geneInfo =  geneInfo)
    
#     # subgroup names
#     sub0 <- sample_group %>% filter(km_tsne1 == 0) %>% pull(1)
#     sub1 <- sample_group %>% filter(km_tsne1 == 1) %>% pull(1)
    
#     sample_subgroup <- tcga_mrna_join %>% colnames() %>% lapply(X = ., FUN = function(value){
      
#       value_trans <- str_extract_all(value, pattern = "TCGA-[:alnum:]+-[:alnum:]+-[:digit:]+") %>%  unlist()
#       subgroup_df <- tibble(sample_barcode = NA, subgroup = NA, .rows = 0)
      
      
#       if(value_trans %in% sub0){
#         subgroup_df <- tibble(sample_barcode = value, subgroup = 0)
#       } else {
#         subgroup_df <- tibble(sample_barcode = value, subgroup = 1)
#       }
      
#       return(subgroup_df)
      
#     }) %>% bind_rows()
    
#     sub0_name <- sample_subgroup %>% filter(subgroup == 0) %>% pull(sample_barcode)
#     sub1_name <- sample_subgroup %>% filter(subgroup == 1) %>% pull(sample_barcode)
    
#     # quantile filter of genes
#     dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
#                                       method = "quantile", 
#                                       qnt.cut =  0.25)
    
#     #Diff.expr.analysis (DEA) # ~ 20min
#     dataDEGs <- TCGAanalyze_DEA(mat1 = dataFilt[,sub0_name],
#                                 mat2 = dataFilt[,sub1_name],
#                                 Cond1type = "Sub-0",
#                                 Cond2type = "Sub-1",
#                                 fdr.cut = 0.01 ,
#                                 logFC.cut = 1,
#                                 method = "glmLRT",
#                                 paired = F)
    
#     # DEGs table with expression values in normal and tumor samples
#     dataDEGsFiltLevel <- TCGAanalyze_LevelTab(dataDEGs,"Sub-0","Sub-1",
#                                               dataFilt[,sub0_name],dataFilt[,sub1_name])    
#   })  
  
  
  
#   return(dataDEGsFiltLevel)
# }

run_deseq_normal <- function(pr_name, rdata_path, deg_path, batch_removal){
  register(MulticoreParam(20))
  suppressMessages({
    if((!file.exists(paste0(rdata_path, pr_name, "_normal.RData"))) | 
       (!file.exists(paste0(rdata_path, pr_name, "_RnaseqSE_normal.RData")))){
      query <- GDCquery(project = paste0("TCGA-", pr_name), 
                        data.category = "Gene expression",
                        data.type = "Gene expression quantification",
                        experimental.strategy = "RNA-Seq",
                        platform = "Illumina HiSeq",
                        file.type = "results",
                        sample.type = c("Primary Tumor", "Solid Tissue Normal"), 
                        legacy = TRUE)

      GDCdownload(query)
      RnaseqSE <- GDCprepare(query)

      save(RnaseqSE, file = paste0(rdata_path, pr_name, "_RnaseqSE_normal.RData"))

      Rnaseq_CorOutliers <- assay(RnaseqSE, "raw_count") # to matrix

      # normalization of genes, # quantile filter of genes
      dataNorm <- TCGAanalyze_Normalization(tabDF = Rnaseq_CorOutliers, geneInfo =  geneInfo)
      dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
                                        method = "quantile", 
                                        qnt.cut =  0.25)

      save(dataFilt, file = paste0(rdata_path, pr_name, "_normal.RData"))
    } else {
      load(paste0(rdata_path, pr_name, "_RnaseqSE_normal.RData"))
      load(paste0(rdata_path, pr_name, "_normal.RData"))
    }


    # selection of normal samples "NT"
    samplesNT <- TCGAquery_SampleTypes(barcode = colnames(dataFilt), typesample = c("NT")) %>% 
      as_tibble() %>% 
      mutate(group = 0) %>% 
      dplyr::rename(sample = value)

    # selection of tumor samples "TP"
    samplesTP <- TCGAquery_SampleTypes(barcode = colnames(dataFilt), typesample = c("TP")) %>% 
      as_tibble() %>% 
      mutate(group = 1) %>% 
      dplyr::rename(sample = value)

    metadata <- bind_rows(samplesNT, samplesTP) %>% 
      mutate(group = ifelse(group == 0, "NT", "TP"))
    metadata$group <- factor(metadata$group, levels = c("NT", "TP"))

    tcga_se <- DESeqDataSetFromMatrix(countData = dataFilt, colData = metadata, design = ~ group)
    tcga_se$group <- relevel(tcga_se$group, ref = "NT")
    tcga_deseq <- DESeq(tcga_se, parallel = TRUE)

    # Hidden Batch effect remove
    if(batch_removal){
        # hidden batch removal
        dat <- counts(tcga_deseq, normalized=TRUE)
        mod <- model.matrix(~ group, colData(tcga_deseq))
        mod0 <- model.matrix(~ 1, colData(tcga_deseq))
        
        # run sva
        svseq <- svaseq(dat, mod, mod0, n.sv=2)
        tcga_se_batch <-tcga_se

        tcga_se_batch$SV1 <- svseq$sv[,1]
        tcga_se_batch$SV2 <- svseq$sv[,2]
        design(tcga_se_batch) <- ~ SV1 + SV2 + group
        tcga_deseq <- DESeq(tcga_se_batch, parallel = TRUE)
    }
      
    tcga_deseq_result <- results(tcga_deseq, 
                                 alpha = 0.9999)
    tcga_deseq_result_tidy <- results(tcga_deseq, 
                                      alpha = 0.9999,
                                      tidy = TRUE)
      
    # volcano plot
    p <- EnhancedVolcano(tcga_deseq_result,
                    lab = rownames(tcga_deseq_result),
                    x = 'log2FoldChange',
                    y = 'padj',
                    title = 'Primary Tumor versus Solid Tissue Normal',
                    pCutoff = 0.05,
                    FCcutoff = 1.5,
                    pointSize = 3.0,
                    labSize = 6.0)

    ggsave(plot = p, filename = paste0(deg_path, pr_name, "_volcano/", pr_name, "_DESEQ2_normal_volcano.png"), height = 8, width = 12, dpi = 70)    
  })  
  
  return(tcga_deseq_result_tidy)
}

run_deseq <- function(pr_name, sample_group_path, rdata_path, group_reverse, file_name, deg_path, batch_removal){
  register(MulticoreParam(20))
  suppressMessages({
    sample_group <- read_delim(file = sample_group_path, delim = "\t", show_col_types = FALSE, progress = FALSE)
      
    # group convert
    if(group_reverse){sample_group <- sample_group %>% mutate(group = ifelse(group == 0, 1, 0))}  
    
    if((!file.exists(paste0(rdata_path, pr_name, ".RData"))) | 
       (!file.exists(paste0(rdata_path, pr_name, "_RnaseqSE.RData")))){
        query <- GDCquery(project = paste0("TCGA-", pr_name), 
                          data.category = "Gene expression",
                          data.type = "Gene expression quantification",
                          experimental.strategy = "RNA-Seq",
                          platform = "Illumina HiSeq",
                          file.type = "results",
                          barcode = sample_group %>% dplyr::pull(1), 
                          legacy = TRUE)

        GDCdownload(query)
        RnaseqSE <- GDCprepare(query)
        
        save(RnaseqSE, file = paste0(rdata_path, pr_name, "_RnaseqSE.RData"))
        
        Rnaseq_CorOutliers <- assay(RnaseqSE, "raw_count") # to matrix

        # normalization of genes, # quantile filter of genes
        dataNorm <- TCGAanalyze_Normalization(tabDF = Rnaseq_CorOutliers, geneInfo =  geneInfo)
        dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
                                          method = "quantile", 
                                          qnt.cut =  0.25)
        colnames(dataFilt) <- dataFilt %>% 
            colnames() %>% lapply(X = ., FUN = function(col_name){
            str_extract_all(col_name, pattern = "TCGA-[:alnum:]+-[:alnum:]+-[:digit:]+") %>% 
                unlist()}) %>% unlist()

        
        
        save(dataFilt, file = paste0(rdata_path, pr_name, ".RData"))
    } else {
        load(paste0(rdata_path, pr_name, "_RnaseqSE.RData"))
        load(paste0(rdata_path, pr_name, ".RData"))
    }


    # metadata
    metadata <- tibble(sample = RnaseqSE$sample_submitter_id, barcode = RnaseqSE$barcode) %>% 
      mutate(sample = str_extract_all(sample, pattern = "TCGA-[:alnum:]+-[:alnum:]+-[:digit:]+") %>% unlist()) %>% 
      left_join(x = ., y = sample_group, by = "sample") %>% 
      mutate(group = ifelse(group == 0, "Sub0", "Sub1"))
    metadata$group <- factor(metadata$group, levels = c("Sub0", "Sub1"))
    
    # order check
    if(!identical(metadata$sample, colnames(dataFilt))){
        stop("DESeq2 order none-identical")  
    }
      
    tcga_se <- DESeqDataSetFromMatrix(countData = dataFilt, colData = metadata, design = ~ group)
    tcga_se$group <- relevel(tcga_se$group, ref = "Sub0")
    tcga_deseq <- DESeq(tcga_se, parallel = TRUE)
    
    # Hidden Batch effect remove
    if(batch_removal){
        # hidden batch removal
        dat <- counts(tcga_deseq, normalized=TRUE)
        mod <- model.matrix(~ group, colData(tcga_deseq))
        mod0 <- model.matrix(~ 1, colData(tcga_deseq))
        
        # run sva
        svseq <- svaseq(dat, mod, mod0, n.sv=2)
        tcga_se_batch <-tcga_se

        tcga_se_batch$SV1 <- svseq$sv[,1]
        tcga_se_batch$SV2 <- svseq$sv[,2]
        design(tcga_se_batch) <- ~ SV1 + SV2 + group
        tcga_deseq <- DESeq(tcga_se_batch,                                                    
                            parallel = TRUE)
    }  
    
    tcga_deseq_result <- results(tcga_deseq, 
                                 alpha = 0.9999)
    tcga_deseq_result_tidy <- results(tcga_deseq, 
                                      alpha = 0.9999,
                                      tidy = TRUE)
    
    # volcano plot
    p <- EnhancedVolcano(tcga_deseq_result,
                    lab = rownames(tcga_deseq_result),
                    x = 'log2FoldChange',
                    y = 'padj',
                    title = 'SubGroup-0 versus SubGroup-1',
                    pCutoff = 0.05,
                    FCcutoff = 1.5,
                    pointSize = 3.0,
                    labSize = 6.0)

    ggsave(plot = p, filename = paste0(deg_path, pr_name, "_volcano/", pr_name, "_DESEQ2_", file_name, "_volcano.png"), height = 8, width = 12, dpi = 70)    
  })  
  
  return(tcga_deseq_result_tidy)
}
# run_deseq_pancan <- function(sample_group_path, involve_brca, group_reverse, batch_removal){
#   register(MulticoreParam(5))
#   sample_group <- read_delim(file = sample_group_path, delim = "\t", show_col_types = FALSE)
    
#   # group convert
#   if(group_reverse){sample_group <- sample_group %>% mutate(group = ifelse(group == 0, 1, 0))}
    
#   study_list <- tcga_available()$Study_Abbreviation[-34] %>% 
#     lapply(X = ., FUN = function(value){
#       paste0("TCGA-", value) %>% return()
#     }) %>% as.character()
  
#   if(!involve_brca){
#     study_list <- study_list[study_list != 'TCGA-BRCA']
#   }
  
#   suppressMessages({
#     tcga_mrna <- mclapply(X = study_list, FUN = function(type){
      
#       error_occur <- FALSE
#       tryCatch(
#         expr = {
#           query <- GDCquery(project = type, 
#                             data.category = "Gene expression",
#                             data.type = "Gene expression quantification",
#                             experimental.strategy = "RNA-Seq",
#                             platform = "Illumina HiSeq",
#                             file.type = "results",
#                             barcode = sample_group %>% pull(1), 
#                             legacy = TRUE)
#           # Download a list of barcodes with platform IlluminaHiSeq_RNASeqV2
#           GDCdownload(query)
          
#           # rsem.genes.results as values
#           RnaseqSE <- GDCprepare(query)
          
#         },
#         error = function(e) { 
#           error_occur <<- TRUE
#         }
#       )
      
#       if(error_occur){
#         return(NULL)
#       } else {
#         RnaseqSE %>%
#           assay() %>% 
#           as.data.frame() %>% 
#           rownames_to_column(var = "gene") %>% return()
#       }
      
#     }, mc.cores = 20)
    
#     # inner join
#     reduce_join <- function(left, right){
#       if(is.null(right)){
#         return(left)
#       } else{
#         inner_join(left, right, by = "gene") %>% return()
#       }
#     }
#     tcga_mrna_join <- purrr::reduce(tcga_mrna, reduce_join) %>% 
#       column_to_rownames(var = "gene") %>% as.matrix()
    
#     # normalization of genes, # quantile filter of genes
#     dataNorm <- TCGAanalyze_Normalization(tabDF = tcga_mrna_join, geneInfo =  geneInfo)
#     dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
#                                       method = "quantile", 
#                                       qnt.cut =  0.25)
    
#     # metadata
#     metadata <- tibble(sample = RnaseqSE$sample_submitter_id, barcode = RnaseqSE$barcode) %>% 
#       mutate(sample = str_extract_all(sample, pattern = "TCGA-[:alnum:]+-[:alnum:]+-[:digit:]+") %>% unlist()) %>% 
#       left_join(x = ., y = sample_group, by = "sample") %>% 
#       mutate(group = ifelse(group == 0, "Sub0", "Sub1"))
#     metadata$group <- factor(metadata$group, levels = c("Sub0", "Sub1"))
    
#     tcga_se <- DESeqDataSetFromMatrix(countData = dataFilt, colData = metadata, design = ~ group)
#     tcga_deseq <- DESeq(tcga_se)
    
#     # Hidden Batch effect remove
#     if(batch_removal){
#         # hidden batch removal
#         dat <- counts(tcga_deseq, normalized=TRUE)
#         mod <- model.matrix(~ group, colData(tcga_deseq))
#         mod0 <- model.matrix(~ 1, colData(tcga_deseq))
        
#         # run sva
#         svseq <- svaseq(dat, mod, mod0, n.sv=2)
#         tcga_se_batch <-tcga_se

#         tcga_se_batch$SV1 <- svseq$sv[,1]
#         tcga_se_batch$SV2 <- svseq$sv[,2]
#         design(tcga_se_batch) <- ~ SV1 + SV2 + group
#         tcga_deseq <- DESeq(tcga_se_batch, parallel = TRUE)
#     }      
    
#     tcga_deseq_result <- results(tcga_deseq, tidy = T, contrast=c("group", "Sub1", "Sub0"))
#     tcga_deseq_result_tidy <- results(tcga_deseq, tidy = TRUE, contrast=c("group", "Sub1", "Sub0"))
    
#     # volcano plot
#     p <- EnhancedVolcano(tcga_deseq_result,
#                     lab = rownames(tcga_deseq_result),
#                     x = 'log2FoldChange',
#                     y = 'padj',
#                     title = 'SubGroup-0 versus SubGroup-1',
#                     pCutoff = 0.05,
#                     FCcutoff = 1.5,
#                     pointSize = 3.0,
#                     labSize = 6.0)

#     ggsave(plot = p, filename = paste0(deg_path, pr_name, "_DESEQ2_", file_name, "_volcano.png"), height = 8, width = 12, dpi = 70) 
    
#   })  
  
#   return(tcga_deseq_result_tidy)
# }
stand_alone_deg <- function(cancer_type, subgroup_path, deg_path){
  subgroup_file_list <- list.files(subgroup_path)
  
  for(value in subgroup_file_list){
    file_name_group <- value %>% 
      str_extract_all(string = ., pattern = "[:digit:]+-[:digit:]+")
    
    # LIHC_DESEQ2_20220127-120900.txt
    file_name_deg <- paste0(cancer_type, "_EDGER_DEG_", file_name_group, ".txt")
    
    # file check
    if(!(file_name_deg %in% list.files(deg_path))){
      print(value)
      edger <- run_edgeR(cancer_type, paste0(subgroup_path, value)) %>% 
        mutate(mRNA = mRNA[,1])
      deseq <- run_deseq(cancer_type, paste0(subgroup_path, value))
      
      # table writing
      edger %>% as_tibble() %>% 
        write_delim(file = paste0(deg_path, cancer_type, "_EDGER_", file_name_group, ".txt"), delim = "\t")
      edger %>% as_tibble() %>% 
        dplyr::select_at(1) %>% 
        write_delim(file = paste0(deg_path, cancer_type, "_EDGER_DEG_", file_name_group, ".txt"), delim = "\t")
      
      deseq %>% as_tibble() %>% 
        write_delim(file = paste0(deg_path, cancer_type, "_DESEQ2_", file_name_group, ".txt"), delim = "\t")
      deseq %>% as_tibble() %>% 
        dplyr::select_at(1) %>% 
        write_delim(file = paste0(deg_path, cancer_type, "_DESEQ2_DEG_", file_name_group, ".txt"), delim = "\t")
    } 
  }
}

# preprocessing
mut_load <- function(MAF_PATH){
  mutation_type <- c('Nonsense_Mutation', 'Missense_Mutation', 'Frame_Shift_Del', 
                     'Frame_Shift_Ins','Splice_Site' )
  
  mut <- data.table::fread(file = MAF_PATH) %>% 
    as_tibble() %>% 
    select(Tumor_Sample_Barcode, everything()) %>% 
    filter(str_detect(Tumor_Sample_Barcode, 'TCGA-[A-Z0-9]+-[A-Z0-9]+-01')) %>%
    filter(Variant_Classification %in% mutation_type) %>% 
    group_by(Tumor_Sample_Barcode, Hugo_Symbol) %>% 
    summarise(cnt = n()) %>% 
    mutate(cnt = 1) 
  
  tumor_barcode <- mut %>% 
    pull(Tumor_Sample_Barcode) %>% 
    unique()
  
  tcga_mut <- lapply(X = tumor_barcode, FUN = function(id){
    longtowide <- mut %>% 
      filter(Tumor_Sample_Barcode == id) %>%
      pivot_wider(names_from = "Tumor_Sample_Barcode", values_from = "cnt")
  }) %>% 
    purrr::reduce(.x = ., full_join, by = "Hugo_Symbol") %>% 
    replace(is.na(.), 0)
  
  tcga_mut_t <- tcga_mut %>% select(-Hugo_Symbol) %>% t()
  colnames(tcga_mut_t) <- tcga_mut %>% pull(Hugo_Symbol)
  
  # rowname
  row_name <- tcga_mut_t %>% rownames() %>% 
    lapply(X = ., FUN = function(value){
      regex_patern = 'TCGA-[A-Z0-9]+-[A-Z0-9]+-01'
      str_extract_all(string = value, pattern = regex_patern) %>% 
        .[[1]] %>% 
        return()
    }) %>% 
    unlist() %>% 
    tibble(sample = .)
  
  tcga_mut_t %>% as_tibble() %>% bind_cols(row_name, .) %>% 
    return()
}

methylation_analysis <- function(df, pr_name, method_="both", base_dir){
  
  suppressMessages({
  log_save <- paste(base_dir, "ELMER", pr_name, sep = "/")
  dir.create(paste(base_dir, "ELMER", pr_name, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  
  if(file.exists(paste0(log_save, "/", pr_name, "_methylation_filtering.RData"))){
    load(paste0(log_save, "/", pr_name, "_methylation_filtering.RData"))
    
  } else {
    
    if(method_ == "both")
      method_ <- c("hypo", "hyper")
    
    if(!file.exists(paste0(log_save, "/", pr_name, "/", pr_name, "_RNA_hg19.rda"))){
      getTCGA(disease = pr_name, 
              genome = "hg19",
              basedir = log_save)  
      
    } 
    load(paste0(log_save, "/", pr_name, "/", pr_name, "_RNA_hg19.rda"))
    load(paste0(log_save, "/", pr_name, "/", pr_name, "_meth_hg19.rda"))
    geneExp <- assay(rna)
    Meth <- assay(met)
    rm(rna, met);gc()
    
    
    distal.probes <- get.feature.probe(
      genome = "hg19", 
      met.platform = "450K", 
      # rm.chr = paste0("chr",c("Y")) # Y는 male specific
      rm.chr = paste0('chr', c(1:6, 10:22, "X","Y"))
    )
    
    mae <- createMAE(
      exp = geneExp, 
      met = Meth,
      save = TRUE,
      linearize.exp = TRUE,
      save.filename = paste0(log_save, "/", pr_name,"_mae.rda"),
      filter.probes = distal.probes,
      met.platform = "450K",
      genome = "hg19",
      TCGA = TRUE
    )
    
    sig.diff_list <- list()
    nearGenes_list <- list()
    methylation_filtering <- list()
    enriched.motif_list <- list()
    TF_list <- list()
    
    for(method in method_){
      sig.diff_list[[method]] <- get.diff.meth(data = mae, 
                                               group.col = "definition",
                                               group1 =  "Primary solid Tumor",
                                               group2 = "Solid Tissue Normal",
                                               minSubgroupFrac = 0.2, # if supervised mode set to 1
                                               sig.dif = 0.3,
                                               diff.dir = method, # Search for hypomethylated probes in group 1
                                               cores = 20, 
                                               dir.out = log_save, 
                                               pvalue = 0.05)
      
      nearGenes_list[[method]] <- GetNearGenes(data = mae, 
                                               probes = sig.diff_list[[method]]$probe, 
                                               numFlankingGenes = 30)
      
      methylation_filtering[[method]] <- get.pair(data = mae,
                                                  group.col = "definition",
                                                  group1 =  "Primary solid Tumor",
                                                  group2 = "Solid Tissue Normal",
                                                  nearGenes = nearGenes_list[[method]],
                                                  mode = "unsupervised",
                                                  permu.dir = log_save,
                                                  permu.size = 1000, # Please set to 100000 to get significant results
                                                  raw.pvalue = 0.05,   
                                                  Pe = 0.001, # Please set to 0.001 to get significant results
                                                  filter.probes = TRUE, # See preAssociationProbeFiltering function
                                                  filter.percentage = 0.05,
                                                  filter.portion = 0.3,
                                                  dir.out = log_save,
                                                  cores = 20,
                                                  label = method)
    }
    
    save(methylation_filtering, file = paste0(log_save, "/", pr_name, "_methylation_filtering.RData"))
  }

  methylation_filtering_ <- lapply(X = names(methylation_filtering), FUN = function(method){
    GENE_NAME <- methylation_filtering[[method]]$Symbol
    P.adjust <- methylation_filtering[[method]]$Pe
    
    tmp_df <- tibble(GENE_NAME = GENE_NAME, Methylated = method, P.adjust = P.adjust) %>% 
      distinct(GENE_NAME, .keep_all = TRUE)
    colnames(tmp_df) <- c("GENE_NAME", paste0('Methylated_', method),
                          paste0('P.adjust_', method))
    return(tmp_df)
  })
  
  names(methylation_filtering_) <- names(methylation_filtering)
  
  rs <- df %>% left_join(x = ., y = methylation_filtering_$hypo, by = c("gene"="GENE_NAME")) %>% 
    left_join(x = ., y = methylation_filtering_$hyper, by =c("gene"="GENE_NAME")) %>% 
    unite(Methylated, Methylated_hypo, Methylated_hyper, sep = "/", na.rm = TRUE) %>% 
    unite(Methylated.Padj, P.adjust_hypo, P.adjust_hyper, sep = "/", na.rm = TRUE)
  })

  return(rs)
}