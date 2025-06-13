from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import sysconfig

class CUDABuildExt(build_ext):
    def build_extension(self, ext):
        # Check if nvcc is available
        try:
            subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "CUDA compiler (nvcc) not found. This package requires CUDA to be installed. "
                "Please install CUDA toolkit or use a pre-built wheel if available."
            )
        
        # Only handle extensions with .cu files
        cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
        if not cuda_sources:
            # If no CUDA sources, use default build
            super().build_extension(ext)
            return
            
        # Compile each .cu file to object files
        objects = []
        for source in cuda_sources:
            # Remove the source from the Extension's sources list
            ext.sources.remove(source)
            
            # Determine the output path
            output_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
            os.makedirs(output_dir, exist_ok=True)
            
            object_file = os.path.join(
                self.build_temp, 
                os.path.splitext(os.path.basename(source))[0] + '.o'
            )
            os.makedirs(os.path.dirname(object_file), exist_ok=True)
            
            # Build nvcc command - updated to include c++ standard libs
            include_dirs = ext.include_dirs if hasattr(ext, 'include_dirs') else []
            include_args = [f'-I{d}' for d in include_dirs]
            
            # Add the directory containing the source file to include paths
            source_dir = os.path.dirname(os.path.abspath(source))
            include_args.append(f'-I{source_dir}')
            
            nvcc_cmd = [
                'nvcc', '-c', source, '-o', object_file, 
                '--compiler-options', '-fPIC', '--std=c++14',
                '-Wno-deprecated-gpu-targets'  # Suppress deprecated GPU targets warning
            ] + include_args
            
            print(f"Compiling {source} with command: {' '.join(nvcc_cmd)}")
            try:
                result = subprocess.check_output(nvcc_cmd, stderr=subprocess.STDOUT)
                print(f"NVCC output: {result.decode()}")
            except subprocess.CalledProcessError as e:
                print(f"NVCC compilation failed with return code {e.returncode}")
                print(f"Command: {' '.join(nvcc_cmd)}")
                print(f"Output: {e.output.decode() if e.output else 'No output'}")
                raise
            objects.append(object_file)
        
        # Add the object files to the Extension
        ext.extra_objects = objects + (ext.extra_objects or [])
        
        # Add CUDA runtime libraries and C++ runtime
        cuda_lib_dirs = ['/usr/local/cuda/lib64']
        ext.library_dirs = (ext.library_dirs or []) + cuda_lib_dirs
        ext.libraries = (ext.libraries or []) + ['cudart', 'stdc++']
        ext.runtime_library_dirs = cuda_lib_dirs
        
        # Now build the extension with the object files
        super().build_extension(ext)

python_include = sysconfig.get_path('include')
python_lib = sysconfig.get_config_var('LIBDIR')
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
cuda_include = os.path.join(cuda_home, 'include')
cuda_lib = os.path.join(cuda_home, 'lib64')

setup(
    ext_modules=[
        Extension(
            'plavchan',
            sources=['./plavchan_gpu/plavchan.cu'],
            include_dirs=[python_include, cuda_include],
            library_dirs=[python_lib],
            libraries=[python_version, 'stdc++'],
        )
    ],
    cmdclass={
        'build_ext': CUDABuildExt,
    },
    # Detailed package info now comes from pyproject.toml
)