import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "mocet",
    version = "0.0.3",
    author = "Jiwoong Park",
    author_email = "jiwoongpark@skku.edu",
    description = "Python package for correcting head motion in eyetracking",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/jwparks/mocet",
    project_urls = {
        "Bug Tracker": "https://github.com/jwparks/mocet",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)