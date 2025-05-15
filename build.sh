rm -r build
rm -r plavchan_gpu.egg-info
rm -r dist
rm plavchan_gpu.cpython-*.so
pip uninstall plavchan_gpu -y
python -m setup build_ext install