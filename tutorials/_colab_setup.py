# tutorials/_colab_setup.py
import os, sys, subprocess

def setup():
    """Install Epydemix and its dependencies if running on Colab."""
    if "google.colab" not in sys.modules and not os.environ.get("COLAB_RELEASE_TAG"):
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", "https://raw.githubusercontent.com/epistorm/epydemix/main/requirements.txt"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "epydemix"])
