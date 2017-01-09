import os
import re
# import glob
from subprocess import check_output
from setuptools import setup, find_packages


def find_version(*paths):
    fname = os.path.join(*paths)
    with open(fname) as fhandler:
        version_file = fhandler.read()
        version_match = re.search(r"^__VERSION__ = ['\"]([^'\"]*)['\"]",
                                  version_file, re.M)

    if not version_match:
        raise RuntimeError("Unable to find version string in %s" % (fname,))

    version = version_match.group(1)

    try:
        command = 'git describe --tags'
        with open(os.devnull, 'w') as fnull:
            tag = check_output(
                command.split(),
                stderr=fnull).decode('utf-8').strip()

        if tag.startswith('v'):
            assert tag == 'v' + version
    except Exception:
        pass

    return version


def find_readme(*paths):
    with open(os.path.join(*paths)) as f:
        return f.read()

setup(
    name = 'tfhelper',
    version = find_version('tfhelper', '__init__.py'),
    packages = find_packages(),
    license = 'MIT License',
    description = "Tensorflow helper",
    classifiers = [
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
    ],
    install_requires = ['tflearn==0.2.2'],
    scripts = [],
)
