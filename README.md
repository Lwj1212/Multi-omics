# **Multi-Omics-Integration**

## **Workflow**

![workflow](https://user-images.githubusercontent.com/42958809/167774736-f059e43b-2de6-4cae-bc5a-dc9b02e0606a.png)


## **Environment**

```
<Clone Repo>
cd Multi-omics-intergration
git clone https://github.com/Jin0331/Multi-omics-intergration.git

<GPU>
conda env create --file conda_env_gpu.yaml
conda activate multiomics

<CPU>
conda env create --file conda_env_cpu.yaml
conda activate multiomics-cpu

```

```
<Subgroup Detection>

optional arguments:
  -h, --help            show this help message and exit
  -b BASE, --base BASE  Root Path
  -c CANCER, --cancer CANCER
                        Types of cancer
  -e CYCLE, --cycle CYCLE
                        configuration of the name of output without a filename extension

example : python src/Multi-omics-integration-subgroup.py \
    -b /home/wmbio/WORK/gitworking/Multi-omics-intergration/ \
    -c COAD \
    -e 1000

<Analysis>

optional arguments:
  -h, --help            show this help message and exit
  -b BASE, --base BASE  Root Path
  -c CANCER, --cancer CANCER
                        Types of cancer
  -a CANCER2, --cancer2 CANCER2
                        Types of cancer2
  -d DEA, --dea DEA     DESeq2(deseq2) or EdgeR(edger) or ALL(all)
  -l LOGFC, --logfc LOGFC
                        log2FC Threshold
  -f FDR, --fdr FDR     False dicovery rate Threshold
  -r SEED, --seed SEED  Random Seed

example : python src/Multi-omics-integration-analysis.py \
         -b /home/wmbio/WORK/gitworking/Multi-omics-intergration/ \
         -c COAD \
         -r 331

```

```@ wmbio.co```
