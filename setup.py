from setuptools import setup

setup(
    name='keras_sentiment_analysis',
    packages=['keras_sentiment_analysis'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn',
        'nltk',
        'numpy',
        'h5py'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)