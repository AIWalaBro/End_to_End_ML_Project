from setuptools import find_packages,setup
from typing import List

Hypen_dot_e = "-e ."

def get_requirements(filepath:str)->List[str]:
    """This function will return the list of requirements
    """

    requirements = []
    with open (filepath) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace ("\n","") for req in requirements]

        if Hypen_dot_e in requirements:
            requirements.remove(Hypen_dot_e)

    return requirements



setup(
    name = "ML_Projects",
    version = "0.0.1",
    author = "Bharatbhushan",
    author_email = "aiwalabro@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )