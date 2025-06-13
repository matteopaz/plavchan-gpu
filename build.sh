rm -r build
rm -r plavchan_gpu.egg-info
rm -r dist
rm plavchan_gpu.cpython-*.so
pip uninstall plavchan_gpu -y

python -m build --wheel --outdir dist
python -m build --sdist --outdir dist

auditwheel repair dist/*.whl --plat manylinux_2_17_x86_64 -w dist