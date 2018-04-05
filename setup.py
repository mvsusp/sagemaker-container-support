import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker-containers',
    version='1.0',
    description='Open source library for creating containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    long_description=read('README.md'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-container-support/',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    install_requires=['Flask', 'boto3', 'six', 'numpy'],
    dependency_links=['https://github.com/aws/sagemaker-python-sdk/tree/mvs-local-mode#egg=sagemaker-1.1.2'],
    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'keras', 'pytest', 'sagemaker']
    }
)
