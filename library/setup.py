import setuptools

setuptools.setup(
    name="library",
    version="0.1.1",
    author="Alexander Piehler & Jann Goschenhofer",
    author_email="moritzwagner95@hotmail.de",
    description="package for representation learning",
    url="https://github.com/MoritzWag/Representation-Learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    zip_safe=False
)