rm -r build
rm -r plavchan_gpu.egg-info
rm -r dist
rm plavchan_gpu.cpython-*.so
pip uninstall plavchan_gpu -y
python -m setup build_ext install
python -m build --wheel
WHEEL_FILE=$(ls dist/*.whl)
auditwheel repair $WHEEL_FILE --plat manylinux_2_17_x86_64 -w dist/fixed