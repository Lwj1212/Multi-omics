FROM tensorflow/tensorflow:2.4.3-jupyter

# multiomics dir copy
RUN git clone https://github.com/Jin0331/Multi-omics-intergration.git /root/Multi-omics-intergration
WORKDIR /root/Multi-omics-intergration

# COPY GDCdata GDCdata
# COPY RAW_DATA RAW_DATA
# COPY pkl pkl

# linux package
RUN apt-get clean all && \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
    libhdf5-dev \
    libcurl4-gnutls-dev \
    libssl-dev \
    libxml2-dev \
    libpng-dev \
    libxt-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libglpk40 \
    libgit2-28 \
    libx11-dev \
    libcairo2-dev \
    libxt-dev \
    libmysqlclient-dev \
  && apt-get clean all && \
  apt-get purge && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*




CMD ["bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]

# conda install tensorflow==2.4.1
# conda install seaborn


# pip install pypdb matplotlib-venn xgboost rpy2

# # rpackage
# conda install -c conda-forge r-base
# conda install -c conda-forge r-essentials
# conda install -c conda-forge r-tidyverse
# conda install -c conda-forge r-ranger
# conda install -c conda-forge r-nbclust
# conda install -c conda-forge r-survminer
# conda install -c conda-forge r-parallelly

# conda install -c bioconda bioconductor-tcgabiolinks
# conda install -c bioconda bioconductor-summarizedexperiment
# conda install -c bioconda bioconductor-deseq2
# conda install -c bioconda bioconductor-biocparallel
# conda install -c bioconda bioconductor-enhancedvolcano
# conda install -c bioconda bioconductor-sva
# conda install -c biobuilds r-annotationdbi
# conda install -c bioconda bioconductor-org.hs.eg.db


# ELMER
# ELMER.data
# sesameData