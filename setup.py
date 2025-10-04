from setuptools import find_packages, setup

setup(
    name="optionslab",
    version="0.1.0",
    description="Quantitative options pricing and risk toolkit",
    author="Diego",
    author_email="tu-email@example.com",  # pon tu email aquí
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.10",
        "pytest>=7.0",
        "xgboost>=1.7",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "streamlit>=1.22",
        # añade cualquier otra dependencia que uses
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
