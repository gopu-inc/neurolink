import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = os.path.dirname(os.path.abspath(__file__))


class BuildWithC(build_py):
    """Compile csrc/neurolite.c en neurolite/libneurolite.so avant de copier le paquet."""

    def run(self):
        src = os.path.join(ROOT, "csrc", "neurolite.c")
        out = os.path.join(ROOT, "neurolite", "libneurolite.so")
        cc = os.environ.get("CC", "gcc")
        ops = os.path.join(ROOT, "csrc", "nlops.c")
        subprocess.check_call([cc, "-O3", "-Wall", "-Wextra", "-shared", "-fPIC",
                               src, ops, "-o", out, "-lm"])
        super().run()


setup(
    name="neurolite",
    version="0.2.0",
    description="Réseau de neurones 100% C avec interface Python et serveur style Ollama",
    packages=["neurolite"],
    package_data={"neurolite": ["libneurolite.so"]},
    python_requires=">=3.9",
    entry_points={"console_scripts": ["neurolink-server=neurolite.server:main"]},
    cmdclass={"build_py": BuildWithC},
)
