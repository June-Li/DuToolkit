FROM hub.sensedeal.vip/library/ubuntu-python-base:22.04-20240612
# COPY conf/sources.list /etc/apt/sources.list
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


RUN wget -O  ImageMagick-6.9.11-62.tar.xz  https://sdai-resource-zjk.oss-cn-zhangjiakou-internal.aliyuncs.com/tools/ImageMagick-6.9.11-62.tar.xz ;\
    tar -xf  ImageMagick-6.9.11-62.tar.xz ;\
    cd ImageMagick-6.9.11-62 ;\
    /ImageMagick-6.9.11-62/configure ;\
    make  ;\
    make install ;\
    /sbin/ldconfig /usr/local/lib ;\
    mkdir /etc/ImageMagick-6
COPY conf/policy.xml /etc/ImageMagick-6/policy.xml
RUN apt-get install -y openssh-server \
libsm6  \
libxrender1  \
libgl1-mesa-glx  



