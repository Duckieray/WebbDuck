# WebbDuck Setup Guide (For Complete Beginners)

This guide walks you through installing and running WebbDuck, step by step, even if
you have never used a terminal, have no GitHub account, and have never installed
Python before. Every command is copy-and-paste-ready.

---

## What you will need

- A computer running **Windows 10 or 11**, or **Linux** (WSL on Windows also works — see Part B).
- An internet connection.
- About **15–30 GB of free disk space** (the program, your model files, and generated images).
- **Optional but strongly recommended:** an NVIDIA graphics card (GPU). WebbDuck can run
  without one, but image generation will be much slower on a computer with no NVIDIA GPU.

> You do **not** need a GitHub account. You do **not** need any paid software.

---

## How to read the commands in this guide

Throughout this guide you will see blocks of text like this:

```
python --version
```

1. Click anywhere inside the box so it is highlighted.
2. Press **Ctrl + C** (Windows) or **Ctrl + Shift + C** (Linux terminal) to copy it.
3. Go to your terminal window.
4. Click in the terminal, then paste with **right-click → Paste**, or press **Ctrl + V**
   (Windows PowerShell) / **Ctrl + Shift + V** (Linux).
5. Press **Enter**.

Do **not** type the `PS C:\>` or `$` prompt symbols you might see in examples — only type
(rather, paste) the commands shown in the boxes.

---

## Part A — Windows (native PowerShell)

### Step 1: Download WebbDuck (no GitHub account needed)

1. Open your browser and go to: **https://github.com/Duckieray/WebbDuck**
2. Click the green **Code** button near the top right of the page.
3. In the menu that drops down, click **Download ZIP**.
4. Wait for `WebbDuck-main.zip` to finish downloading (it goes to your **Downloads** folder).
5. Open your Downloads folder. **Right-click** `WebbDuck-main.zip` and choose **Extract All...**,
   then click **Extract**.
6. A new folder named `WebbDuck-main` appears. **Right-click** it and choose **Rename**, then
   rename it to `webbduck` (this keeps the commands below simple).
7. Move the `webbduck` folder somewhere easy to remember, for example `C:\webbduck`
   (move it into `C:\` by dragging, then you will know exactly where it is).
   Keeping it in Downloads is fine too — just remember where it is.

### Step 2: Open PowerShell

1. Press the **Windows key** on your keyboard.
2. Type `PowerShell`.
3. Click **Windows PowerShell** (the regular one is fine — no need for Administrator).

A blue window opens. This is your terminal — pretend it is your computer's "driver's seat".

### Step 3: Go to the WebbDuck folder

Type (or paste) this, replacing `C:\webbduck` with your folder's location if you put it elsewhere:

```
cd C:\webbduck
```

> **Beginner tip:** Instead of typing the path, you can drag the `webbduck` folder from File
> Explorer straight into the PowerShell window and press **Enter** — Windows fills in the path
> for you.

Check that you are in the right place:

```
dir
```

You should see files named `run.py`, `README.md`, `startup.ps1`, and folders like `server/` and `ui/`.
If you see that, you are in the right folder.

### Step 4: Check that Python is installed

```
python --version
```

- If you see `Python 3.10.x`, `Python 3.11.x`, or `Python 3.12.x` — great, skip to **Step 5**.
- If Windows opens the **Microsoft Store** or says *"Python was not found"*:

  1. Go to **https://www.python.org/downloads/** and download the latest Python **3.12** installer.
  2. Run the installer.
  3. **Important:** tick the box that says **"Add python.exe to PATH"** at the bottom of the first screen.
  4. Click **Install Now**, wait for it to finish, then **close and reopen PowerShell**.
  5. Run `python --version` again — it should now print a version number.

### Step 5: Create a "virtual environment" and activate it

A virtual environment keeps WebbDuck and its libraries in their own private space so they
never break other programs. Copy-paste these two lines one at a time:

```
python -m venv .venv
```

```
.\.venv\Scripts\Activate.ps1
```

After the second line, your prompt should start with `(.venv)`, like `(.venv) PS C:\webbduck>`.

> If you get a red error that says **"running scripts is disabled on this system"**, paste this
> line, press **Enter** (press **Y** if asked), then run the `Activate.ps1` line again:
>
> ```
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### Step 6: Install PyTorch (the big AI library)

This downloads a few gigabytes, so it can take a while depending on your internet speed.
Paste the line that matches your situation:

**If you have an NVIDIA graphics card** (most common):

```
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

**If you do not have an NVIDIA card** (or want a safe fallback, e.g. on a very new GPU):

```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Wait for it to finish. A long stream of text is completely normal.

### Step 7: Install the rest of WebbDuck

```
pip install -r requirements.windows.txt
```

This installs everything else WebbDuck needs and can take 5–15 minutes. Do not close the
window while it runs. When it finishes, you will see your prompt again.

### Step 8: Create the folders WebbDuck expects

```
mkdir checkpoint\sdxl, lora, embeddings, outputs, weights
```

### Step 9: Install WebbDuck's generation engine

WebbDuck runs each AI "engine" (SDXL, FLUX, Krea 2, Qwen) inside its own private mini-environment,
so one engine can never break the others. Even your first SDXL image needs this step.

```
python tools\prepare_model_runtimes.py sdxl
```

- It **auto-detects your graphics card** and installs the matching AI libraries. First run downloads
  a few GB, so it can take a while — do not close the window.
- When it finishes it **remembers where the engine lives** (it writes a small file, `webbduck_runtime.env`,
  in the WebbDuck runtimes folder). The launcher below reads that file automatically, so you never have
  to type `export` lines yourself.

> **Only want SDXL for now?** Then you are done here. Later, to enable FLUX, Krea 2, and Qwen model
> engines too (big downloads), run the same command with `all` instead of `sdxl`:
> `python tools\prepare_model_runtimes.py all`.

### Step 10: Start WebbDuck

**Easy way — the one-call launcher:**

```
.\startup.ps1
```

Wait until you see something like `Uvicorn running on http://0.0.0.0:8010`. Do not close this
window — WebbDuck runs inside it.

**Or start it directly** (same command as you would have typed by hand):

```
python .\run.py --output .\outputs\ --port 8010
```

### Step 11: Open WebbDuck in your browser

1. Open **Chrome**, **Edge**, or **Firefox**.
2. Go to: **http://localhost:8010**

WebbDuck's Studio should load. You're in. 🎉

> Keep the PowerShell window open while you use the app. To stop WebbDuck later, click the window
> and press **Ctrl + C** (or just close the window).

---

## Part B — Linux (and WSL on Windows)

These instructions work for Ubuntu/Debian-style Linux and for **WSL** running on Windows
(WSL already gives you a Linux terminal window you can open).

### Step 1: Download and unzip WebbDuck (no GitHub account needed)

**Option 1 — browser (easiest):**

1. Go to **https://github.com/Duckieray/WebbDuck**
2. Click the green **Code** button → **Download ZIP**.
3. The file `WebbDuck-main.zip` lands in your `Downloads` folder.
4. Open your Terminal and paste:

```
cd ~/Downloads
unzip WebbDuck-main.zip
mv WebbDuck-main webbduck
```

**Option 2 — all in the terminal:**

```
cd ~
wget https://github.com/Duckieray/WebbDuck/archive/refs/heads/main.zip
unzip main.zip
mv WebbDuck-main webbduck
```

If `unzip` is not installed, install it first (Ubuntu/Debian):

```
sudo apt install unzip
```

### Step 2: Open a terminal

Most Linux desktops: press **Ctrl + Alt + T**. In WSL: open the **Ubuntu** (or your distro) app.

### Step 3: Go to the WebbDuck folder

```
cd ~/webbduck
```

Verify you are in the right place:

```
ls
```

You should see files named `run.py`, `README.md`, `startup.sh`, and folders like `server/` and `ui/`.

### Step 4: Check that Python is installed

```
python3 --version
```

- `Python 3.10`, `3.11`, or `3.12` — great, continue to **Step 5**.
- If it says `command not found` or the version is older, install Python (Ubuntu/Debian):

```
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### Step 5: Create and activate a virtual environment

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

> If the first line fails with a message about `ensurepip` or `venv`, run
> `sudo apt install python3-venv` and try again.

### Step 6: Install PyTorch

**If you have an NVIDIA graphics card:**

```
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```

**If you do not have an NVIDIA card:**

```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

This download is large; give it time.

### Step 7: Install the rest of WebbDuck

```
pip install -r requirements.txt
```

This can take 5–15 minutes. When it finishes, your prompt returns.

### Step 8: Create the folders WebbDuck expects

```
mkdir -p checkpoint/sdxl lora embeddings outputs weights
```

### Step 9: Install WebbDuck's generation engine

WebbDuck runs each AI "engine" (SDXL, FLUX, Krea 2, Qwen) inside its own private mini-environment,
so one engine can never break the others. Even your first SDXL image needs this step.

```
python tools/prepare_model_runtimes.py sdxl
```

- It **auto-detects your graphics card** and installs the matching AI libraries. First run
  downloads a few GB, so it can take a while — do not close the terminal.
- When it finishes it **remembers where the engine lives** (it writes a small file, `webbduck_runtime.env`,
  in the WebbDuck runtimes folder). The launcher below reads that file automatically, so you never have
  to type `export` lines yourself.

> **Only want SDXL for now?** Then you are done here. Later, to enable FLUX, Krea 2, and Qwen model
> engines too (big downloads), run the same command with `all` instead of `sdxl`:
> `python tools/prepare_model_runtimes.py all`.

### Step 10: Start WebbDuck

**Easy way — the one-call launcher:**

```
./startup.sh
```

Wait until you see `Uvicorn running on http://0.0.0.0:8010`. Keep this terminal window open.

**Or start it directly** (same command as you would have typed by hand):

```
python run.py --output ./outputs --port 8010
```

### Step 11: Open WebbDuck in your browser

Open your browser and go to: **http://localhost:8010**

> To stop WebbDuck later, press **Ctrl + C** in the terminal window (or close it).

---

## After it runs: your first image needs a model

WebbDuck is the "paint shop"; it also needs a **model** (the brain that draws) — you installed the
engine in Step 9, but models are the individual "artists" that ship separately. Until you add one,
the model list will be empty.

1. Get an **SDXL** model file (a `.safetensors` file, several GB).
   - You can download a free one from sites like Hugging Face (`civitai.com` and
     `huggingface.co` are common sources).
2. Put that file inside the `checkpoint/sdxl/` folder in your WebbDuck folder.
   - (`C:\webbduck\checkpoint\sdxl\` on Windows, or `~/webbduck/checkpoint/sdxl/` on Linux)
3. Back in the WebbDuck browser tab, **refresh the model list** (restart WebbDuck, or use the
   refresh/reload control in the Studio) and the model should appear.

For a full tour of the Studio, prompts, and tools, read `docs/USER_GUIDE.md` and
`docs/SIMPLE_GUIDE.md` (the short in-app help).

---

## Restarting and updating WebbDuck

**Restart after you closed the window:**

- Windows: open PowerShell, then:
  ```
  cd C:\webbduck
  .\startup.ps1
  ```
  (or the by-hand version: `.\.venv\Scripts\Activate.ps1` then `python .\run.py --output .\outputs\ --port 8010`)
- Linux: open a terminal, then:
  ```
  cd ~/webbduck
  ./startup.sh
  ```
  (or the by-hand version: `source .venv/bin/activate` then `python run.py --output ./outputs --port 8010`)

**Update to a newer version:**

WebbDuck updates are released as new zip downloads (following git isn't required).

1. Download the latest ZIP and extract it into a new folder (for example, keep the old folder too).
2. Copy your **personal folders** from the old one into the new one — these hold YOUR stuff:
   - `checkpoint/` (your models)
   - `lora/` and `embeddings/` (your add-ons)
   - `outputs/` (your generated images)
   - `weights/` (upscaler weights)
3. Install and run the new folder following steps 5–11 of Part A or Part B. Your images and
   models carry over; you do not lose anything.

---

## Quick guide: common problems

| If you see this... | It means... | Do this |
| --- | --- | --- |
| `python is not recognized` / `Python was not found` | Python isn't on your PATH | Reinstall Python and tick **"Add python.exe to PATH"**, then reopen your terminal (Windows Step 4) |
| `running scripts is disabled on this system` | Windows blocks the activation script | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again (Windows Step 5) |
| `ModuleNotFoundError: webbduck` | You ran WebbDuck from the wrong folder | Run the `cd` step first, then start it from that same folder (Parts A/B step 3) |
| `no kernel image is available for execution on the device` | PyTorch build doesn't match your GPU | Reinstall torch using the CPU wheel line, or update your GPU driver (Windows Step 6) |
| `Address already in use` / port 8010 busy | Another program uses that port | Use a different port: `python run.py --port 8020`, then go to `http://localhost:8020` |
| The model list is empty | No model file is in `checkpoint/sdxl/` yet | Add an SDXL `.safetensors` file there and refresh (see "After it runs") |
| Generation fails with a message about a missing runtime engine (hint: `prepare_model_runtimes.py`) | The engine was never installed | Run Step 9 (`python tools\prepare_model_runtimes.py sdxl` on Windows, or `python tools/prepare_model_runtimes.py sdxl` on Linux), then restart WebbDuck |
| Startup says there is no runtime env file | You skipped Step 9 (the engine installer) | Run Step 9 once so the engine and its `webbduck_runtime.env` file are created, then launch again |
| Generation is very slow | No NVIDIA GPU, or the app fell back to CPU | Nothing is broken; CPU generation is just slow. A dedicated NVIDIA GPU makes it fast |
| A long Hugging Face symlink warning on Windows | Windows-safe file-linking notice | Often non-fatal; if generation works, ignore it |

If a command fails, the exact red text it prints is the most useful thing to search for or
share with someone helping you.

---

## Want to skip ZIP downloads next time? (optional)

If you ever create a free GitHub account and want the `git` command way instead, replace the
download step with:

```
git clone https://github.com/Duckieray/WebbDuck.git
cd WebbDuck
```

Everything else in this guide stays the same.

---

## More help

- `docs/USER_GUIDE.md` — the end-user features tour.
- `docs/SIMPLE_GUIDE.md` — the short in-app help text.
- `README.md` — product overview and the full documentation list.