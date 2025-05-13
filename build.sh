rm -r build
rm -r plavchan_gpu.egg-info
rm plavchan_gpu.cpython-*.so
python -m setup build_ext --inplace
pip uninstall plavchan_gpu -y
pip install .