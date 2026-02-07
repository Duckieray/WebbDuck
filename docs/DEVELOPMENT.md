# WebbDuck Development Guide 🛠️

This guide covers the core backend architecture and how to extend WebbDuck.

## 🏗️ Backend Architecture (Python)

The backend is built with **FastAPI** (`server/`) and a custom PyTorch-based inference engine (`core/`).

### Folder Structure

```
webbduck/
├── server/
│   ├── app.py          # FastAPI application & Endpoint definitions
│   ├── events.py       # WebSocket broadcasting
│   ├── state.py        # Shared runtime state (VRAM, Progress)
│   └── storage.py      # File system helpers
├── core/
│   ├── pipeline.py     # Diffusers pipeline manager (Load/Unload models)
│   ├── worker.py       # Async GPU job processor
│   ├── generation.py   # Core generation logic
│   └── schedulers.py   # Scheduler definitions
├── modes/              # Generation Logic (Strategy Pattern)
│   ├── text2img.py     # Txt2Img implementation
│   ├── img2img.py      # Img2Img implementation
│   └── inpaint.py      # Inpainting implementation
└── ui/                 # Frontend (See ui/README.md)
    ├── core/           # Utilities & State
    ├── modules/        # Feature blocks (Gallery, Lightbox, etc.)
    └── styles/         # CSS Architecture

## 🛠️ Development Tips

### Auto-Reload
The server is configured with `reload=True` in `run.py`. Any changes to python files in `webbduck/` will automatically restart the server.

### Frontend Modules
The UI uses ES6 modules. No build step is required!
- Edit `ui/modules/MyManager.js`
- Refresh browser
- Done!
```

---

## ⚡ How to Add a New API Endpoint

### 1. Define the Endpoint in `server/app.py`

WebbDuck uses FastAPI forms for compatibility with simple frontend requests.

```python
# server/app.py

@app.post("/my_feature")
async def my_feature_endpoint(
    image: UploadFile = File(...),
    intensity: float = Form(0.5),
):
    """Docstrings are good!"""
    try:
        # 1. Validate inputs
        if intensity > 1.0:
             return JSONResponse(status_code=400, content={"error": "Too intense!"})

        # 2. Perform logic (CPU bound)
        result = do_cpu_work(image, intensity)
        
        return {"status": "success", "data": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
```

### 2. Handling GPU Tasks (The Queue)

If your endpoint needs the GPU (Inference, VRAM usage), **DO NOT** run it directly in the endpoint function. You must use the `generation_queue` to avoid conflicts.

```python
# server/app.py

@app.post("/gpu_task")
async def gpu_task_endpoint(prompt: str = Form(...)):
    # 1. Create a Future to await the result
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # 2. Construct the Job
    job = {
        "type": "my_custom_job_type",
        "settings": { "prompt": prompt },
        "future": future
    }

    # 3. Enqueue
    # This sends it to core/worker.py -> gpu_worker()
    await generation_queue.put(job)

    # 4. Wait for result
    return await future
```

**Note**: You will need to modify `core/worker.py` to handle `"my_custom_job_type"`.

### 3. Modifying the Worker (`core/worker.py`)

Open `core/worker.py` and find the `gpu_worker` loop. Add your handler:

```python
# core/worker.py

async def gpu_worker(queue):
    while True:
        job = await queue.get()
        try:
            if job["type"] == "batch":
                # ... existing logic
            
            elif job["type"] == "my_custom_job_type":
                # CALL YOUR CORE LOGIC HERE
                result = run_my_gpu_task(job["settings"])
                job["future"].set_result(result)

        except Exception as e:
             job["future"].set_exception(e)
```

---

## 🧩 Adding a New Generation Mode

Generation modes (Txt2Img, Inpaint) reside in `modes/`. To add a new one:

1.  Create `modes/my_mode.py`.
2.  Implement a function that accepts `(pipe, settings)` and returns images.
3.  Register it or call it from `core/generation.py`.

---

## 🌐 Connecting the Frontend

1.  **Update API Client**: Add a wrapper in `ui/core/api.js`.
    ```javascript
    export async function runMyFeature(formData) {
        return postForm('/my_feature', formData);
    }
    ```

2.  **Add UI Handler**: See `ui/README.md` for details on adding buttons and handlers.

---

## 🧪 Testing

WebbDuck uses **pytest**. Run tests from the root directory:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_server.py
```
