from setuptools import setup, find_namespace_packages


setup(
    version="0.2.0",
    name="mdn_metric",
    long_description="Probabilistic approach to classification and metric learning.",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "catalyst==21.9",
        "faiss-cpu==1.7.2",  # Need for MAP@R metric computation.
        "jpeg4py==0.1.4",
        "mxnet==2.0.0b1",  # Used for RecordIO reading.
        "numpy==1.23.4",
        "optuna==3.0.3",
        "pretrainedmodels==0.7.4",
        "scikit-image==0.19.3",
        "scikit-learn==1.1.3",
        "scipy==1.8.1",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "Pillow==9.3.0",
        "PyYAML==6.0",
        "gitpython==3.1.29",
        "wandb==0.13.5",
        "pycocotools==2.0.6"
    ]
)
