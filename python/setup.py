from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            "danma_torch._C",
            ["csrc/danma_privateuse1.cpp"],
            extra_compile_args={"cxx": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
