from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='helper',
    version='0.1.0',
    packages=find_packages(),
    requirements=requirements,
    install_requires=requirements,
    url='https://github.com/MatiasEtcheve/DRL-FlappyBird',

)
