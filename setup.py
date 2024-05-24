"""Setup file for tfsoftadapt package installation."""

from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
      name="tfsoftadapt",
      version="0.0.1",
      author="Philip Schwedler",
      author_email="phil.schwedler@web.de",
      description=("SoftAdapt for tensorflow"),
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/philipschw/tfsoftadapt",
      download_url="https://github.com/philipschw/tfsoftadapt",
      packages=find_packages(),
      install_requires=[
                        "findiff",
                        "tqdm>=4.47.0",
                        "tensorflow>=2.13.1",
                        ],
      classifiers=[
                   "Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT Software License",
                   "Programming Language :: Python :: 3.8.10",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"
                   ],
      keywords=("Adaptive-Weighting, Multi-Task-Nerual-Networks Optimization",
                "Gradient-Descent-Weighting, Machine Learning")
      )
