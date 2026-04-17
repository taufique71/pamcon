import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pybind11


class build_ext(_build_ext):
    """Strip C++-only flags when compiling plain C sources."""
    _cpp_only = {"-std=c++11", "-fpermissive"}

    def build_extension(self, ext):
        original = self.compiler._compile

        def patched(obj, src, ext_name, cc_args, extra_postargs, pp_opts):
            if src.endswith(".c"):
                extra_postargs = [a for a in extra_postargs if a not in self._cpp_only]
            original(obj, src, ext_name, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = patched
        try:
            super().build_extension(ext)
        finally:
            self.compiler._compile = original

def get_openmp_flags():
    """Return (compile_args, link_args, include_dirs) for OpenMP on current platform."""
    if sys.platform == "darwin":
        # macOS: Apple clang does not include OpenMP.
        # Requires: brew install libomp
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            # Fallback to common Homebrew paths (Intel and Apple Silicon)
            import os
            for path in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
                if os.path.isdir(path):
                    prefix = path
                    break
            else:
                raise RuntimeError(
                    "OpenMP (libomp) not found on macOS.\n"
                    "Install it with: brew install libomp"
                )
        compile_args = ["-Xpreprocessor", "-fopenmp"]
        link_args    = [f"-L{prefix}/lib", "-lomp"]
        include_dirs = [f"{prefix}/include"]
        return compile_args, link_args, include_dirs

    elif sys.platform.startswith("linux"):
        return ["-fopenmp"], ["-fopenmp"], []

    else:
        raise RuntimeError(
            f"Platform '{sys.platform}' is not supported. "
            "pamcon requires Linux or macOS."
        )


openmp_compile, openmp_link, openmp_includes = get_openmp_flags()

ext = Extension(
    "pamcon._core",
    sources=[
        "pamcon/_core.cpp",
        "NIST/mmio.c",       # required by COO.h -> ReadMM
    ],
    include_dirs=[
        ".",                  # repo root: CSC.h, COO.h, consensus.h, utils.h, defs.h
        pybind11.get_include(),
    ] + openmp_includes,
    extra_compile_args=[
        "-std=c++11",
        "-O3",
        "-ffast-math",
        "-fpermissive",
    ] + openmp_compile,
    extra_link_args=openmp_link,
    language="c++",
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
