from fastapi import FastAPI, Request, Response
import httpx

app = FastAPI()

BACKENDS = {
    "figure1":          "http://figure1:8053",
    "path-integration": "http://path-integration:8051",
    "temporal":         "http://temporal:8052",
    "polar":            "http://polar:8056",
    "trace-plot":       "http://trace-plot:8055",
}

@app.get("/")
async def export_img(request: Request):
    app_name = request.query_params.get("app")
    if not app_name:
        names = ", ".join(BACKENDS)
        return Response(f"Require &app=<name>. Available: {names}", status_code=400,
                        media_type="text/plain")
    backend = BACKENDS.get(app_name)
    if backend is None:
        names = ", ".join(BACKENDS)
        return Response(f'Unknown app "{app_name}". Available: {names}', status_code=400,
                        media_type="text/plain")

    # Strip &app= before forwarding — the Dash route doesn't know it
    qs = "&".join(p for p in request.url.query.split("&") if not p.startswith("app="))
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{backend}/?{qs}", timeout=120.0)
            media = r.headers.get("content-type", "application/octet-stream")
            return Response(content=r.content, status_code=r.status_code, media_type=media)
        except Exception as e:
            return Response(f"Connection error: {e}", status_code=502,
                            media_type="text/plain")
