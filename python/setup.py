#!/usr/bin/env python

import os
from distutils.core import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='rules',
      version='0.1',
      description='',
      author='',
      author_email='',
      license='',
      url='',
      packages=['rules'],
      long_description=read('README.md'),
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.8",
      ],
      install_requires=[]
      )
