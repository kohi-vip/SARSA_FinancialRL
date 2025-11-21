from setuptools import setup, find_packages

setup(
    name="data_processor",  # Tên package
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "TA-Lib",
    ],
    author="Koh1gh",
    description="SARSA Financial RL Project",
)

setup(
    name="data_provider",  # Tên package
    version="0.1.0",
    packages=find_packages(where="."),  # Tìm packages trong thư mục này
    package_dir={"": "."},  # Root package là thư mục hiện tại
    install_requires=[
        "pandas",
        "numpy",
        "TA-Lib",  # Dependencies cần thiết cho data_processor
    ],
    author="Your Name",
    description="Data processing module for SARSA Financial RL",
)