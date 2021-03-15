from setuptools import setup, find_packages

# TODO package data 

with open('README.md', 'r') as fh:
    long_description = fh.read()

print('FIND PACKAGES OUTPUT:', find_packages())


setup(
    name='tasknet',
    version='0.0.0',
    author='Stanford University',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='none of your business',
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'networkx',
        'jupytext'
    ],
    package_data={},
    include_package_data=True
)
