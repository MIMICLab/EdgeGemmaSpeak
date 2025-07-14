from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentvox",
    version="0.2.0",
    author="MIMIC Lab",
    author_email="",
    description="Edge-based voice assistant using Gemma LLM with STT and TTS capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentvox",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "SpeechRecognition",
        "faster-whisper",
        "llama-cpp-python",
        "edge-tts",
        "pygame",
        "sounddevice",
        "soundfile",
        "gradio",
        "flask",
        "pyaudio",
    ],
    entry_points={
        "console_scripts": [
            "agentvox=agentvox.cli:main",
        ],
    },
    package_data={
        "agentvox": ["*.gguf"],
    },
    include_package_data=True,
)