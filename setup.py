# @File  : setup.py
# @Author: Piston.Y
# @Contact : pistonyang@gmail.com
# @Date  : 18-9-18

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pistonlib',
    version='0.0.1a',
    platforms='any',
    keywords=('deep learning', 'personal useful tools'),
    description='Piston deep learning tools.',
    license='MIT License',

    author='Piston Yang',
    author_email='pistonyang@gmail.com',
    url='https://github.com/PistonY/piston',

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
