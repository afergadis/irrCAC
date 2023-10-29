from setuptools import setup, find_packages

setup(
    name="irrCAC",
    version="0.4.1",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
    ],
    install_requires=["scipy", "pandas"],
    extras_require={
        # optional dependencies grouped by functionality, e.g.
        "docs": [
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme",
            "sphinxcontrib-bibtex",
            "myst-parser",
            "nbsphinx",
        ],
    },
    # other setup kwargs
)
