"""Start the FastAPI server"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.serve:app",
        "--port", "8000",
        "--reload"
    ])
