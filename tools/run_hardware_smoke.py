#!/usr/bin/env python3
"""Run WebbDuck's image hardware smoke matrix against a live server.

This is intentionally API-level validation: it exercises the same public model
catalog, capability checks, queue, storage and backend routing used by Studio.
It never downloads models and does nothing expensive unless --execute is set.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid


PROMPT = "A small brass clockwork duck on a wooden workbench, studio lighting, highly detailed"
NEGATIVE = "blurry, malformed, duplicate"


def _request_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[-1200:]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach {url}: {exc}") from exc
    return json.loads(body) if body else {}


def _multipart(fields: dict[str, object], files: dict[str, Path] | None = None) -> tuple[bytes, str]:
    boundary = f"----webbduck-smoke-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    for key, path in (files or {}).items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"; filename="{path.name}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _post_form(base_url: str, path: str, fields: dict[str, object], files: dict[str, Path] | None = None, timeout: float = 7200.0):
    body, boundary = _multipart(fields, files)
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {raw[-2000:]}") from exc
    return json.loads(raw) if raw else {}


def _make_fixture_images(root: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    source = root / "source.png"
    mask = root / "mask.png"
    img = Image.new("RGB", (1024, 1024), (42, 54, 68))
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 220, 844, 804), fill=(122, 86, 48))
    draw.ellipse((362, 330, 662, 630), fill=(220, 175, 62))
    draw.text((390, 700), "WebbDuck smoke", fill=(245, 245, 245))
    img.save(source)

    m = Image.new("L", (1024, 1024), 0)
    md = ImageDraw.Draw(m)
    md.rectangle((360, 340, 680, 660), fill=255)
    m.save(mask)
    return source, mask


def _find_model(items: list[dict], explicit: str | None, patterns: tuple[str, ...]) -> dict | None:
    if explicit:
        for item in items:
            if str(item.get("name")) == explicit:
                return item
        return None
    for item in items:
        text = str(item.get("name") or "").lower()
        if all(pattern in text for pattern in patterns):
            return item
    return None


def _runtime_summary(item: dict | None) -> dict:
    if not item:
        return {"ready": False, "reason": "model not discovered"}
    runtime = dict(item.get("runtime") or {})
    return {
        "ready": bool(item.get("ready")),
        "reason": runtime.get("reason"),
        "python": runtime.get("python"),
        "torch": runtime.get("torch"),
        "diffusers": runtime.get("diffusers"),
        "cuda_available": runtime.get("cuda_available"),
        "gpu_name": runtime.get("gpu_name"),
        "compute_capability": runtime.get("compute_capability"),
        "vram_gb": runtime.get("vram_gb"),
        "source_state": item.get("source_state"),
    }


def _output_refs(result: dict) -> list[str]:
    refs: list[str] = []
    images = result.get("images")
    if isinstance(images, list):
        refs.extend(str(value) for value in images)
    if result.get("image"):
        refs.append(str(result["image"]))
    nested = result.get("result")
    if isinstance(nested, dict):
        refs.extend(_output_refs(nested))
    return refs


def _wait_for_job(base_url: str, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        queue = _request_json(f"{base_url}/queue", timeout=20)
        candidates = list(queue.get("jobs") or []) + list(queue.get("recent_completed") or [])
        for row in candidates:
            if str(row.get("job_id")) != job_id:
                continue
            status = str(row.get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                if status != "completed":
                    raise RuntimeError(str(row.get("error") or f"job ended as {status}"))
                return row
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for WebbDuck job {job_id}")


def _run_generation(base_url: str, fields: dict[str, object], files: dict[str, Path] | None, timeout: float) -> tuple[dict, float]:
    started = time.monotonic()
    result = _post_form(base_url, "/generate", fields, files, timeout=timeout)
    if result.get("status") == "queued" and result.get("job_id"):
        result = _wait_for_job(base_url, str(result["job_id"]), timeout)
    elapsed = time.monotonic() - started
    refs = _output_refs(result)
    if not refs:
        raise RuntimeError(f"Generation returned no output artifact reference: {result}")
    return result, elapsed


def _base_fields(model_name: str, *, width: int, height: int, steps: int, cfg: float, seed: int = 0) -> dict[str, object]:
    return {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "base_model": model_name,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "num_images": 1,
        "scheduler": "UniPC",
        "loras": "[]",
        "embeddings": "[]",
    }


def _rows(args, readiness_items: list[dict], source: Path, mask: Path) -> list[dict]:
    sdxl = _find_model(readiness_items, args.sdxl_model, tuple()) if args.sdxl_model else None
    flux = _find_model(readiness_items, args.flux_model, ("flux.2", "klein"))
    krea = _find_model(readiness_items, args.krea_model, ("krea-2", "turbo"))
    dark = _find_model(readiness_items, args.dark_beast_model, ("dark", "beast"))
    qwen = _find_model(readiness_items, args.qwen_model, ("qwen-image-2512",))

    def row(name, model, fields, files=None, optional=False):
        return {"row": name, "model": model, "fields": fields, "files": files or {}, "optional": optional}

    rows: list[dict] = []
    if sdxl:
        name = str(sdxl["name"])
        rows.extend(
            [
                row("sdxl-t2i", sdxl, _base_fields(name, width=1024, height=1024, steps=20, cfg=7.0)),
                row("sdxl-i2i", sdxl, {**_base_fields(name, width=1024, height=1024, steps=20, cfg=7.0), "strength": 0.6}, {"image": source}),
                row("sdxl-inpaint", sdxl, {**_base_fields(name, width=1024, height=1024, steps=20, cfg=7.0), "strength": 0.8, "inpainting_fill": "replace", "mask_blur": 8}, {"image": source, "mask": mask}),
            ]
        )
        asset_fields = _base_fields(name, width=1024, height=1024, steps=20, cfg=7.0)
        if args.lora:
            asset_fields["loras"] = json.dumps([{"name": args.lora, "weight": 1.0}])
        if args.refiner:
            asset_fields.update({"second_pass_model": args.refiner, "second_pass_steps": 12, "second_pass_blend": 0.8})
        rows.append(row("sdxl-assets", sdxl, asset_fields, optional=not (args.lora or args.refiner)))
    else:
        rows.append(row("sdxl", None, {}, optional=True))

    if flux:
        name = str(flux["name"])
        rows.append(row("flux-t2i", flux, _base_fields(name, width=1024, height=1024, steps=4, cfg=1.0)))
        rows.append(row("flux-edit", flux, _base_fields(name, width=1024, height=1024, steps=4, cfg=1.0), {"image": source}))
    else:
        rows.append(row("flux", None, {}))

    if krea:
        rows.append(row("krea-turbo", krea, _base_fields(str(krea["name"]), width=1024, height=1024, steps=8, cfg=0.0)))
    else:
        rows.append(row("krea-turbo", None, {}))

    if dark:
        defaults = dark.get("defaults") or {}
        rows.append(row("dark-beast-fp8", dark, _base_fields(str(dark["name"]), width=int(defaults.get("width") or 1024), height=int(defaults.get("height") or 1024), steps=int(defaults.get("steps") or 28), cfg=float(defaults.get("cfg", 4.5)))))
    else:
        rows.append(row("dark-beast-fp8", None, {}, optional=True))

    if qwen:
        defaults = qwen.get("defaults") or {}
        rows.append(row("qwen-image-2512", qwen, _base_fields(str(qwen["name"]), width=int(defaults.get("width") or 1328), height=int(defaults.get("height") or 1328), steps=int(defaults.get("steps") or 50), cfg=float(defaults.get("cfg", 4.0)))))
    else:
        rows.append(row("qwen-image-2512", None, {}))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WebbDuck's hardware smoke matrix against a live server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--execute", action="store_true", help="Actually submit generation jobs. Without this flag only preflight is performed.")
    parser.add_argument("--only", action="append", default=[], help="Run only named row(s); may be repeated.")
    parser.add_argument("--sdxl-model", help="Exact public model name for the known-good SDXL checkpoint.")
    parser.add_argument("--flux-model", help="Override auto-detected FLUX model name.")
    parser.add_argument("--krea-model", help="Override auto-detected Krea model name.")
    parser.add_argument("--dark-beast-model", help="Exact public name for the local Dark Beast 3 FP8 checkpoint.")
    parser.add_argument("--qwen-model", help="Override auto-detected Qwen Image model name.")
    parser.add_argument("--lora", help="Known-good SDXL LoRA name for the optional asset row.")
    parser.add_argument("--refiner", help="Known-good SDXL second-pass/refiner model name for the optional asset row.")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--report-dir", type=Path, default=Path("smoke_reports"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    readiness = _request_json(f"{base_url}/runtime-readiness", timeout=60)
    items = list(readiness.get("items") or [])
    if not items:
        raise SystemExit("No models were returned by /runtime-readiness.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"webbduck_{stamp}.json"
    report = {
        "kind": "webbduck-hardware-smoke",
        "started_at": time.time(),
        "base_url": base_url,
        "execute": args.execute,
        "rows": [],
    }

    with tempfile.TemporaryDirectory(prefix="webbduck_smoke_") as tmp:
        source, mask = _make_fixture_images(Path(tmp))
        rows = _rows(args, items, source, mask)
        if args.only:
            selected = set(args.only)
            rows = [row for row in rows if row["row"] in selected]

        for spec in rows:
            model = spec.get("model")
            entry = {
                "row": spec["row"],
                "model": model.get("name") if model else None,
                "runtime": _runtime_summary(model),
                "status": "pending",
            }
            report["rows"].append(entry)
            if model is None:
                entry["status"] = "skipped" if spec.get("optional") else "blocked"
                entry["reason"] = "required smoke target is not discovered"
                print(f"{entry['status'].upper():7} {spec['row']}: {entry['reason']}")
                continue
            if not model.get("ready"):
                entry["status"] = "blocked"
                entry["reason"] = str((model.get("runtime") or {}).get("reason") or "runtime/weights preflight is not ready")
                print(f"BLOCKED {spec['row']}: {entry['reason']}")
                continue
            if spec.get("optional") and not (args.lora or args.refiner):
                entry["status"] = "skipped"
                entry["reason"] = "no --lora or --refiner supplied"
                print(f"SKIPPED {spec['row']}: {entry['reason']}")
                continue
            if not args.execute:
                entry["status"] = "ready"
                print(f"READY   {spec['row']}: {entry['model']}")
                continue

            try:
                result, seconds = _run_generation(base_url, spec["fields"], spec["files"], args.timeout)
                entry.update(
                    {
                        "status": "passed",
                        "elapsed_seconds": round(seconds, 3),
                        "outputs": _output_refs(result),
                        "result": result,
                    }
                )
                print(f"PASSED  {spec['row']}: {seconds:.1f}s")
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                print(f"FAILED  {spec['row']}: {exc}")
            finally:
                try:
                    _post_form(base_url, "/models/unload_all", {}, timeout=120)
                except Exception as exc:
                    entry.setdefault("warnings", []).append(f"unload failed: {exc}")
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["finished_at"] = time.time()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")
    failed = any(row["status"] in {"failed", "blocked"} for row in report["rows"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
