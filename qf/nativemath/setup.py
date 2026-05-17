from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Set compiler flag for C++20 based on OS
compile_args = ["-std=c++23"]
if os.name == 'nt':  # Windows/MSVC
    compile_args = ["/std:c++20"]

# Define the path to the nativemath directory
native_dir = "qf/nativemath"

ext = Extension(
    name="qf.nativemath.impl",  # This determines how you import it in Python
    sources=[
        f"{native_dir}/impl.pyx", 
        f"{native_dir}/src/indicators.cpp", 
        f"{native_dir}/src/probabilities.cpp", 
        f"{native_dir}/src/angles.cpp", 
        f"{native_dir}/src/prices.cpp", 
        f"{native_dir}/src/fracdiff.cpp",
        f"{native_dir}/src/stats.cpp",
        f"{native_dir}/src/entries.cpp",
        f"{native_dir}/src/sizing.cpp",
        f"{native_dir}/src/ohlc.cpp"
    ],
    language="c++",
    extra_compile_args=compile_args,
    include_dirs=[np.get_include(), f"{native_dir}/src"]
)

setup(
    name="qf",
    packages=["qf", "qf.nativemath"],
    ext_modules=cythonize(ext, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()]
)