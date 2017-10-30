from setuptools import setup

setup(
    name='keras_sentiment_analysis_web',
    packages=['keras_sentiment_analysis_web'],
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