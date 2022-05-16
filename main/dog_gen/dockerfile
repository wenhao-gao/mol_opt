FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion unzip  vim \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh \
 && /bin/bash ~/anaconda.sh -b -p /opt/conda  \
 && rm ~/anaconda.sh  \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Copy this repo into the container -- make sure that you have cloned both the repo and its submodules (see readme)!
COPY . /synthesis-dags
RUN unzip -o /synthesis-dags/uspto.zip -d /synthesis-dags
RUN unzip -o /synthesis-dags/scripts/dataset_creation/data.zip -d /synthesis-dags/scripts/dataset_creation/

WORKDIR /synthesis-dags

RUN /opt/conda/bin/conda env create -f conda_dogae_gpu.yml \
 && /opt/conda/bin/conda env create -f conda_mtransformer_gpu.yml \
 && /opt/conda/bin/conda clean --all

# Clone down the Molecular Transformer from pschwllr's repo!:
# This is their code for their paper: https://pubs.acs.org/doi/10.1021/acscentsci.9b00576
RUN git clone https://github.com/pschwllr/MolecularTransformer.git /molecular_transformer
# Get the weights here:
RUN mkdir /molecular_transformer/saved_models \
 && wget --quiet https://ndownloader.figshare.com/files/25673870 -O /molecular_transformer/saved_models/molecular_transformer_weights.pt
COPY ./misc/mtransformer_example_server.conf.json /molecular_transformer/available_models

CMD /bin/bash docker_run_tests.sh
