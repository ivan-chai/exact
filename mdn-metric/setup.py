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
        "faiss-cpu",  # Need for MAP@R metric computation.
        "jpeg4py",
        "mxnet",  # Used for RecordIO reading.
        "numpy",
        "optuna",
        "pretrainedmodels",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "Pillow",
        "PyYAML",
        "gitpython",
        "wandb",
        "pycocotools"
    ]
)
