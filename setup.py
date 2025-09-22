from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="microeconomics-databricks",
    version="1.0.0",
    author="Mauricio Cortes",
    author_email="mauricio.cortes@example.com",
    description="Uma biblioteca abrangente de microeconomia para Python no Databricks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mauricio-cortes/from-databricks-microeconomics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords="microeconomics economics databricks python analysis modeling",
    project_urls={
        "Bug Reports": "https://github.com/mauricio-cortes/from-databricks-microeconomics/issues",
        "Source": "https://github.com/mauricio-cortes/from-databricks-microeconomics",
        "Documentation": "https://github.com/mauricio-cortes/from-databricks-microeconomics#readme",
    },
)