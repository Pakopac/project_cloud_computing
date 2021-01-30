import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import subprocess

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/algo_start", response_class=HTMLResponse)
async def model_page(request: Request):
    return templates.TemplateResponse("algo_start.html", {"request": request})

@app.post("/algo", response_class=HTMLResponse)
async def model(request: Request, model: str = Form(...), split: float = Form(...)):
    bashCommand = "python /home/lilian/project_cloud_computing/ml/model.py --model {} -split {}".format(model, split)
    output = subprocess.check_output(bashCommand, shell=True)
    return templates.TemplateResponse("algo_result.html", {"request": request, "output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
