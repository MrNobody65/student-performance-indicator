from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        while HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='MLprojects-CI-CD-AWS',
    version='0.0.1',
    author='MrNobody65',
    author_email='lexuanvu161@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)