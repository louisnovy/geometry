from setuptools import setup, find_packages

__version__ = "0.0.0"

def main():
    setup(
        name="geometry",
        version=__version__,
        author="Louis Novy",
        author_email="novylouis@gmail.com",
        install_requires=["numpy", "scipy", "xxhash"],
        extras_require={"dev": ["pytest"]},
    )

if __name__ == "__main__":
    main()