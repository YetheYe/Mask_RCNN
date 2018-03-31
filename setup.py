from setuptools import setup, find_packages

setup(
    name='Mask_RCNN',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'h5py',
        'keras',
        'numpy',
        'opencv-contrib-python',
        'scikit-image',
        'tensorflow'
    ]
)
