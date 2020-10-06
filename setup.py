import setuptools
import os
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('segmind_track.version',
                           os.path.join('segmind_track',
                                        'version.py')).load_module().VERSION
with open("README.md","r") as f:
    long_description = f.read()

setuptools.setup(
    name="segmind_track",
    version=version,
    author="T Pratik",
    author_email="pratik@segmind.com",
    description="A tracking tool for deep-learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pk00095/keras_jukebox/archive/0.0.3.tar.gz",
    packages=setuptools.find_packages(),
    install_requires=[
        'pycocotools', 
        'click', 
        'protobuf==3.13.0',
        'lxml',
        'pascal_voc_writer',
        'GPUtil',
        'PYyaml',
        'pandas'],
    entry_points={
      "console_scripts": [
          "segmind=segmind_track.cli:cli"]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"],
    python_requires='>=3.6')