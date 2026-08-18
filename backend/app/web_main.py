"""Run MADA as a web server."""
import uvicorn
from app.web_api import create_web_app

if __name__ == "__main__":
    app = create_web_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
