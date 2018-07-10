from setuptools import setup

setup(
    name='houttuynia',
    version=open('VERSION', 'r').read(),
    long_description=open('README.md', 'r').read(),
    packages=[
        'houttuynia',
        'houttuynia.data_loader',
        'houttuynia.extensions',
        'houttuynia.nn',
        'houttuynia.nn.modules',
        'houttuynia.nn.utils',
    ],
    install_requires=[
        'torch>=0.4.0',
        'tqdm',
        'numpy',
        'scipy',
        'gensim',
        'nltk',
        'spacy',
        'sklearn',
        'matplotlib',
        'yellowbrick',
        'logbook',
        'tensorboardX',
    ],
    url='https://github.com/speedcell4/houttuynia.git',
    license='MIT',
    author='Izen',
    author_email='speedcell4@gmail.com',
    description='PyTorch for NLP'
)
