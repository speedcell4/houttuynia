from setuptools import setup

with open('VERSION', 'r', encoding='utf-8') as fd:
    version = str(fd.read())

with open('README.md', 'r', encoding='utf-8') as fd:
    long_description = str(fd.read())

with open('requirements.txt', 'r', encoding='utf-8') as fd:
    install_requires = [str(requirement.strip()) for requirement in fd]

setup(
    name='houttuynia',
    version=version,
    description='the fear of freedom',
    long_description=long_description,
    install_requires=install_requires,
    url='https://github.com/speedcell4/houttuynia.git',
    license='MIT',
    author='Izen',
    author_email='speedcell4@gmail.com',
    packages=[
        'houttuynia',
        'houttuynia.data_loader',
        'houttuynia.schedules',
        'houttuynia.schedules.extensions',
        'houttuynia.nn',
        'houttuynia.nn.modules',
        'houttuynia.nn.utils',
    ],
)
