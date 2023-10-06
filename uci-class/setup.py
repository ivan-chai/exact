from setuptools import setup, find_namespace_packages


setup(
    version="0.0.1",
    name="uci-class",
    long_description="Data loaders for UCI classification datasets.",
    url="https://github.com/tinkoff-ai/uci-class",
    author="Ivan Karpukhin (Tinkoff)",
    author_email="i.a.karpukhin@tinkoff.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "exact-pytorch>=0.0.2",
        "openpyxl",
        "pandas",
        "numpy>=1.12.0",
        "scikit-learn>=0.24.2",
        "torch>=1.10.2",
        "tqdm",
        "xlrd"
    ],
    dependency_links=[
        "git@github.com:tinkoff-ai/exact-internal.git#egg=exact-pytorch"
    ]
)
