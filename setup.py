import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='glacier_mapping',
    version='0.1.0',
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Various",
    author_email="NA",
    license='BSD 2-clause',
    packages=['glacier_mapping'],
    install_requires=required,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
    ],
)
