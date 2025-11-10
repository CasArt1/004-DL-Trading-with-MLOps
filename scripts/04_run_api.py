"""Script to run the FastAPI server."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uvicorn
from configs.config import API_HOST, API_PORT


def main():
    """Run the FastAPI server."""
    print("Starting DL Trading API server...")
    print(f"Server will be available at: http://{API_HOST}:{API_PORT}")
    print(f"API documentation: http://{API_HOST}:{API_PORT}/docs")
    print("Press CTRL+C to stop the server")
    
    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )


if __name__ == "__main__":
    main()
