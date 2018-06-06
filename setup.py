from setuptools import setup

setup(
    name='houttuynia',
    version='0.0.1',
    packages=[
        'houttuynia',
        'houttuynia.transformers',
        'houttuynia.datasets',
        'houttuynia.nn',
    ],
    install_requires=[
        'tqdm',
    ],
    url='',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='PyTorch for NLP'
)
