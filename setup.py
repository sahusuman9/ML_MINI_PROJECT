from setuptools import find_packages, setup  # Importing necessary modules from setuptools
from typing import List  # Importing List from typing module for type hinting

# Defining a constant for the '-e .' entry in requirements.txt
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the given requirements file and returns a list of required packages.
    
    Args:
        file_path (str): Path to the requirements.txt file.
    
    Returns:
        List[str]: A list of package names required for the project.
    """
    requirements = []  # Initialize an empty list to store requirements
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()  # Read all lines from the file
        requirements = [req.replace("\n", "") for req in requirements]  # Remove newline characters
        
        # Remove '-e .' if present in the requirements list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements  # Return the cleaned list of requirements

# Setting up the package
setup(
    name='mlproject',  # Name of the project/package
    version='0.0.1',  # Version of the package
    author='Suman Sahu',  # Author name
    author_email='suman.sahu.981@gmail.com',  # Author email
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=get_requirements('requirements.txt')  # Installing dependencies from requirements.txt
)
