FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bck
RUN cat /etc/apt/sources.list.bck | sed 's#us\.archive#old-releases#g' > /etc/apt/sources.list
RUN apt-get update --fix-missing
RUN apt-get install -y python3 python3-pip psmisc git libturbojpeg
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools tox tox-current-env pytest jupyter matplotlib tensorboard \
	catalyst==21.9 faiss-cpu==1.7.2 jpeg4py==0.1.4 mxnet==2.0.0b1 numpy==1.23.4 optuna==3.0.3 \
	pretrainedmodels==0.7.4 scikit-image==0.19.3 scikit-learn==1.1.3 scipy==1.8.1 torch==1.12.1 \
	torchvision==0.13.1 Pillow==9.3.0 PyYAML==6.0 gitpython==3.1.29 wandb==0.13.5 pycocotools==2.0.6 \
    protobuf==3.20.1