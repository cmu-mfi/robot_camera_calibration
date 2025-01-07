from setuptools import setup, find_packages

setup(
    name='robot_canera_calib',
    version='1.0',
    packages=find_packages(),  # Automatically find all packages (e.g., 'core')
    install_requires=[         # Add dependencies from requirements.txt
        line.strip() for line in open('requirements.txt').readlines()
    ]
)
