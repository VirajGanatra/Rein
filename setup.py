from setuptools import setup, Extension
import pybind11
import os

# Define the extension module
replay_buffer_module = Extension(
    'replay_buffer',
    sources=[
        'src/cpp/src/bindings.cpp',
        'src/cpp/src/replay_buffer.cpp'
    ],
    include_dirs=[
        'src/cpp/include',
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    language='c++',
    extra_compile_args=['-std=c++14']
)

# Setup script
setup(
    name='Rein',
    version='0.1',
    description='Reinforcement Learning with Replay Buffer',
    ext_modules=[replay_buffer_module],
    zip_safe=False,
)