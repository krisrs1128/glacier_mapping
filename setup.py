# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='glacier_mapping',
    version='0.1.0',
    description='Glacier mapping',
    author='Various',
    author_email='test@test.com',
    url='https://github.com/krisrs1128/glacier_mapping',
    install_requires=required,
    packages=find_packages(exclude=('web', 'ee_code', 'docs', 'conf', 'cluster'))
)


