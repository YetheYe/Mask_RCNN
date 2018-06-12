from setuptools import setup, find_packages

setup(
    name='Mask_RCNN',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'h5py==2.8.0rc1',
        'keras==2.1.6',
        'numpy',
        'opencv-contrib-python',
        'scikit-image',
    ]
)
