from setuptools import setup

setup(
    name='houttuynia',
    version='0.0.1',
    packages=[
        'houttuynia',
        'houttuynia.datasets',
        'houttuynia.extensions',
        'houttuynia.nn',
        'houttuynia.nn.modules',
    ],
    install_requires=[
        'torch>=0.4.0',
        'tqdm',
        'numpy',
        'sklearn',
        'matplotlib',
        'logbook',
        'tensorboardX',
    ],
    url='',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='PyTorch for NLP'
)
