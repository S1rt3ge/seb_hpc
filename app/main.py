from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="SEB SME Cash Management")

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/SEB Login_files", StaticFiles(directory="../templates/SEB Login_files"), name="seb_login_files")

# Templates
templates = Jinja2Templates(directory="../templates")

# Sample customer data (will be replaced with DB later)
CUSTOMERS = [
    {"id": "SME-12345", "name": "Tech Solutions Ltd"},
    {"id": "SME-67890", "name": "Baltic Trade Co"},
    {"id": "SME-24680", "name": "Green Energy SIA"},
    {"id": "SME-13579", "name": "Digital Services OÜ"},
]


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "SEB Login.html",
        {"request": request, "customers": CUSTOMERS}
    )


@app.post("/login")
async def login(customer_id: str = Form(...)):
    return RedirectResponse(
        url=f"/dashboard?customer_id={customer_id}",
        status_code=303
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, customer_id: str):
    # Find customer name
    customer = next(
        (c for c in CUSTOMERS if c["id"] == customer_id),
        {"id": customer_id, "name": "Unknown"}
    )
    return templates.TemplateResponse(
        "SME Cash Management - SEB.html",
        {"request": request, "customer": customer}
    )


@app.get("/api/customers")
async def get_customers():
    return CUSTOMERS


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
