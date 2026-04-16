from setuptools import setup, Extension
import pybind11

ext = Extension(
    "pamcon._core",
    sources=[
        "pamcon/_core.cpp",
        "NIST/mmio.c",       # required by COO.h -> ReadMM
    ],
    include_dirs=[
        ".",                  # repo root: CSC.h, COO.h, consensus.h, utils.h, defs.h
        pybind11.get_include(),
    ],
    extra_compile_args=[
        "-std=c++11",
        "-O3",
        "-fopenmp",
        "-ffast-math",
        "-fpermissive",
    ],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    ext_modules=[ext],
)
