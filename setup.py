from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import sysconfig

class CUDABuildExt(build_ext):
    def build_extension(self, ext):
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
            
            nvcc_cmd = [
                'nvcc', '-c', source, '-o', object_file, 
                '--compiler-options', '-fPIC', '--std=c++14'
            ] + include_args
            
            print(f"Compiling {source} with command: {' '.join(nvcc_cmd)}")
            subprocess.check_call(nvcc_cmd)
            objects.append(object_file)
        
        # Add the object files to the Extension
        ext.extra_objects = objects + (ext.extra_objects or [])
        
        # Add CUDA runtime libraries and C++ runtime
        cuda_lib_dirs = ['/usr/local/cuda/lib64']
        ext.library_dirs = (ext.library_dirs or []) + cuda_lib_dirs
        ext.libraries = (ext.libraries or []) + ['cudart', 'stdc++']  # Added stdc++
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
    name="plavchan_gpu",
    version='1.0',
    ext_modules=[
        Extension(
            'plavchan',
            sources=['plavchan.cu'],
            include_dirs=[python_include, cuda_include],
            library_dirs=[python_lib],
            libraries=[python_version, 'stdc++'],
        )
    ],
    cmdclass={
        'build_ext': CUDABuildExt,
    }
)