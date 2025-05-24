from setuptools import setup, find_packages

setup(name="pictollms",
    version="0.1.0",
    description="MT for french pictogram-to-text generation",
    author="Robin Kokot",
    author_email="robin.edu.hr@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.5.0",
        "transformers>=4.35.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "requests>=2.25.0",
        "lmdb>=1.4.0",
        "tqdm>=4.60.0",
        "pillow>=8.3.0",
    ],python_requires="==3.10.17",
)