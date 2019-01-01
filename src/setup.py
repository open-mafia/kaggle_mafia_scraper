
from setuptools import setup, find_packages

setup(
    name="bay12_scraper",
    description="Scraper for Bay12's forums (for mafia use)",
    version="0.1",
    author="Anatoly Makarevich",
    author_email="anatoly_mak@yahoo.com",  
    packages=find_packages(),
    install_requires=[
        'beautifulsoup4', 'requests', 
        'numpy', 'pandas', 
        'prompt_toolkit>=2', 
    ]
    # TODO: Add requirements and such
)
