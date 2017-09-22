from setuptools import setup

setup(
    name='keras_sentiment',
    packages=['keras_sentiment'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)