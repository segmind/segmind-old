import setuptools

import os
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('segmind.version',
                           os.path.join('segmind',
                                        'version.py')).load_module().VERSION
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='segmind',
    version=version,
    author='Saurabh Chopra',
    author_email='Saurabh.Chopra.2021@live.rhul.ac.uk',
    description='A tracking tool for deep-learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="https://github.com/pk00095/keras_jukebox/archive/0.0.3.tar.gz",
    packages=setuptools.find_packages(),
    install_requires=[
        'pycocotools', 'click', 'protobuf==3.13.0', 'lxml',
        'pascal_voc_writer', 'GPUtil', 'PYyaml', 'pandas',
        'entrypoints', 'psutil', 'boto3', 'requests'
    ],
    entry_points={'console_scripts': ['segmind=segmind.cli:cli']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux'
    ],
    python_requires='>=3.6')
