FROM ubuntu:20.04 
COPY conf/sources.list /etc/apt/sources.list
RUN rm -rf /etc/apt/sources.list.d/cuda*
RUN rm -rf /etc/apt/sources.list.d/nvidia*

RUN echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections
RUN echo 'tzdata tzdata/Zones/Europe select Shanghai' | debconf-set-selections
ENV DEBIAN_FRONTEND="noninteractive"

RUN rm -rf /var/lib/apt/lists/* &&\
    apt-get clean &&\
    apt-get update  &&\
    apt-get install -y --no-install-recommends \
        wget curl ca-certificates unzip autoconf libtool automake tzdata \
        vim git build-essential cmake pkg-config tree software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y imagemagick
RUN rm /etc/ImageMagick-6/policy.xml
COPY conf/policy.xml /etc/ImageMagick-6/policy.xml
RUN apt-get install -y openssh-server
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1


ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV PYTHONIOENCODING=utf-8
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
ENV NVIDIA_VISIBLE_DEVICES=all

RUN mkdir /root/.pip
COPY conf/pip.conf /root/.pip/pip.conf

RUN echo 'Asia/Shanghai' >/etc/timezone && rm -rf /etc/localtime && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
WORKDIR /root/


RUN wget -O /root/Miniconda3-4.7.12.1-Linux-x86_64.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
     /bin/bash /root/Miniconda3-4.7.12.1-Linux-x86_64.sh -b -p /root/miniconda3 && rm /root/Miniconda3-4.7.12.1-Linux-x86_64.sh
RUN echo '\n\
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"\n\
if [ $? -eq 0 ]; then\n\
    eval "$__conda_setup"\n\
else\n\
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then\n\
        . "/root/miniconda3/etc/profile.d/conda.sh"\n\
    else\n\
        export PATH="/root/miniconda3/bin:$PATH"\n\
    fi\n\
fi\n\
unset __conda_setup\n' >> ~/.bashrc && /bin/bash -c 'source  ~/.bashrc'
RUN echo "\
channels:\n\
   - defaults\n\
show_channel_urls: true\n\
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda\n\
default_channels:\n\
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free\n\
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro\n\
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n" > ~/.condarc
RUN ln -sf /root/miniconda3/bin/conda /bin/conda
RUN conda create -n torch_new python=3.8.5 -y
RUN ln -sf /root/miniconda3/envs/torch_new/bin/pip /bin/pip
RUN ln -sf /root/miniconda3/envs/torch_new/bin/python /bin/python
RUN echo '\n\
conda activate torch_new\n'  >> ~/.bashrc && /bin/bash -c 'source  ~/.bashrc'

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# COPY torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl /root/
# COPY torchvision-0.11.2+cu102-cp38-cp38-linux_x86_64.whl /root/
# COPY torchaudio-0.10.1+rocm4.1-cp38-cp38-linux_x86_64.whl /root/
# RUN pip install /root/torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl && rm /root/torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl
# RUN pip install /root/torchvision-0.11.2+cu102-cp38-cp38-linux_x86_64.whl && rm /root/torchvision-0.11.2+cu102-cp38-cp38-linux_x86_64.whl
# RUN pip install /root/torchaudio-0.10.1+rocm4.1-cp38-cp38-linux_x86_64.whl && rm /root/torchaudio-0.10.1+rocm4.1-cp38-cp38-linux_x86_64.whl
# RUN pip install http://192.168.1.80/torch-1.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl
# RUN pip install torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

COPY conf/requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
RUN pip list && python -V

RUN mkdir /Logs
RUN mkdir /root/SensedealImgAlg

ENV PYTHONPATH /root/SensedealImgAlg
COPY . /root/SensedealImgAlg

WORKDIR /root/SensedealImgAlg/WORKFLOW/OTHER/OCR/v3

CMD ["python","sc_utils/server.py","ProdEnv"]
