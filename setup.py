#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re

# load version form _version.py
VERSIONFILE = "neutral_diffusion/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(name='neutral_diffusion',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujii@me.kyoto-u.ac.jp",
      description=("Neutral diffusion model in high temperature plasmas."),
      license="BSD 3-clause",
      keywords="plasma-fusion",
      url="http://github.com/fujii-team/neutral_diffusion",
      include_package_data=True,
      ext_modules=[],
      packages=["neutral_diffusion", ],
      package_dir={'neutral_diffusion': 'neutral_diffusion'},
      py_modules=['neutral_diffusion.__init__'],
      test_suite='tests',
      install_requires="""
        numpy>=1.11
        sparse
        """,
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics']
      )
