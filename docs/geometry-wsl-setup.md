# Running the geometry service on Windows with WSL2

This sets up the geometry service on the Windows PC that already runs ComfyUI (RTX 5090,
`192.168.86.67`). The client keeps running on the Mac and reaches the service through an SSH
tunnel. Background on why WSL is needed and how the two share the GPU: section 6 of
[exploration-notes.md](exploration-notes.md).

Commands marked **PowerShell (admin)** run on Windows in an elevated PowerShell. Commands marked
**WSL** run in the Ubuntu shell. Commands marked **Mac** run on the client machine.

## 1. Windows prerequisites

1. **NVIDIA driver.** Use a current Windows Game Ready or Studio driver. Do not install a Linux
   NVIDIA driver inside WSL: the Windows driver provides the GPU to WSL.
2. **Sysmem fallback off.** In NVIDIA Control Panel → *Manage 3D settings* → *Global Settings*,
   set **CUDA – Sysmem Fallback Policy** to **Prefer No Sysmem Fallback**. Without this, a job
   that runs out of VRAM spills into system RAM and slows to a crawl instead of failing with an
   error the service can retry. ComfyUI's `--lowvram` mode does its own offloading, so it should
   be unaffected. If ComfyUI starts reporting out-of-memory errors, this setting is why.
3. **Install WSL with Ubuntu 24.04** (it ships Python 3.12, which the service requires).

   **PowerShell (admin)**
   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   ```
   Reboot if asked, then open *Ubuntu 24.04* from the Start menu and create a user.

4. **WSL settings.** Create `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored   # WSL shares Windows' network, so 127.0.0.1 is the same on both
   memory=64GB               # default is half of RAM; reconstruction offloads frames to RAM
   swap=16GB
   ```
   Then restart WSL so the settings apply:

   **PowerShell (admin)**
   ```powershell
   wsl --shutdown
   ```

5. **Check the GPU is visible from WSL.**

   **WSL**
   ```bash
   nvidia-smi
   ```
   You should see the RTX 5090. The memory figures are for the whole card, including what
   ComfyUI holds. The service relies on this to decide when a job can start.

## 2. System packages

**WSL**
```bash
sudo apt update
sudo apt install -y git build-essential python3.12-venv python3-dev ffmpeg tmux
```

## 3. Get the code

Keep everything on the Linux filesystem (`~/...`), not under `/mnt/c`. Access to the Windows
drive from WSL is many times slower.

**WSL**
```bash
cd ~
git clone https://github.com/stevemayne/ATAS.git
cd ATAS/geometry
```

## 4. Python environment

The tested environment is pinned in `geometry/environment.snapshot.txt` (PyTorch 2.10 built for
CUDA 12.8, which supports the RTX 5090).

**WSL**
```bash
python3.12 -m venv ~/envs/atas-gpu
source ~/envs/atas-gpu/bin/activate
pip install -U pip wheel

# PyTorch first, from the CUDA 12.8 index
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Everything else in the snapshot except flash_attn (see the note below)
grep -v '^flash_attn' environment.snapshot.txt > /tmp/geometry-reqs.txt
pip install -r /tmp/geometry-reqs.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Check: torch sees the card and supports its architecture (expect sm_120 in the list)
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

**flash_attn.** Neither model the service runs needs it. Pi3X falls back to xFormers or PyTorch
attention, and SAM3's FlashAttention path is off by default. There is no prebuilt wheel for the
RTX 5090, and compiling it takes a long time. Skip it unless a job fails with an import error
that names it. In that case:

```bash
pip install ninja
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash_attn==2.8.3 --no-build-isolation
```

If another snapshot package fails to install, it is most likely a dependency of the wider
Vista4D project rather than of the service. Remove that line from `/tmp/geometry-reqs.txt` and
run the install again. If a job later fails to import it, install it on its own.

## 5. Model weights

The service needs Pi3X (reconstruction) and SAM3 (moving-object masks). It expects them as:

```
~/models/
├── pi3x/            # Pi3X checkpoint directory (from_pretrained format)
└── sam3/sam3.pt
```

**WSL**
```bash
# The hf command comes with the pinned huggingface_hub; do not upgrade it (transformers needs <1.0)
hf auth login                      # SAM3 is gated: first request access at huggingface.co/facebook/sam3

mkdir -p ~/models/sam3
hf download facebook/sam3 sam3.pt --local-dir ~/models/sam3

# Pi3X: take the Hugging Face repo id from https://github.com/yyfz/Pi3 (at time of writing it is
# listed as yyfz233/Pi3X; confirm before downloading). The directory name must end in "pi3x".
hf download yyfz233/Pi3X --local-dir ~/models/pi3x
```

## 6. Configure the service

**WSL**
```bash
cp .env.example .env
openssl rand -hex 32               # copy this token; the client needs the same value
```

Edit `~/ATAS/geometry/.env`:

```bash
GEOMETRY_PYTHON=/home/<you>/envs/atas-gpu/bin/python
GEOMETRY_MODEL_ROOT=/home/<you>/models
GEOMETRY_HOT_ROOT=/home/<you>/geometry-hot      # worlds live here: keep it, back it up
GEOMETRY_HOST=127.0.0.1                         # tunnel only; see section 8 for direct LAN access
GEOMETRY_PORT=18300
GEOMETRY_API_TOKEN=<the token from above>
GEOMETRY_GPU_IDS=0
GEOMETRY_MAX_GPUS=1
# Leave GEOMETRY_RESIDENT_MODELS unset: each job then loads Pi3X/SAM3 and exits, returning VRAM
# to ComfyUI.
```

## 7. Start it and check it

Run it inside `tmux`, so it keeps running after you close the terminal. WSL also stays up only
while something is running in it.

**WSL**
```bash
tmux new -s geometry
cd ~/ATAS/geometry && ./run_geometry_service.sh
# detach with Ctrl-b d; reattach later with: tmux attach -t geometry
```

In a second WSL shell:

```bash
curl -s http://127.0.0.1:18300/healthz
curl -s -H "Authorization: Bearer $(grep ^GEOMETRY_API_TOKEN ~/ATAS/geometry/.env | cut -d= -f2)" \
     http://127.0.0.1:18300/readyz | python3 -m json.tool
```

`/readyz` lists its checks: storage writable, Python, Pi3X, the SAM3 checkpoint and
`nvidia-smi`. All must be `"ok": true`. If one fails, it shows which path it looked at.

Optional: run the service's own tests from `~/ATAS/geometry` with `python -m pytest`.

## 8. Connect the Mac client

### Recommended: SSH tunnel

The service stays bound to localhost, and the token never crosses the network in plain text.

1. Enable the Windows OpenSSH server.

   **PowerShell (admin)**
   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType Automatic
   ```
2. Open the tunnel. With mirrored networking, Windows' `127.0.0.1:18300` is the WSL service.

   **Mac**
   ```bash
   ssh -N -L 18300:127.0.0.1:18300 <windows-user>@192.168.86.67
   ```
3. In `client/.env` on the Mac:
   ```bash
   GEOMETRY_SERVICE_URL=http://127.0.0.1:18300
   GEOMETRY_API_TOKEN=<same token>
   ```
4. Check from the Mac: `curl -s http://127.0.0.1:18300/healthz`

### Alternative: direct on the home LAN

Set `GEOMETRY_HOST=0.0.0.0`. Allow the port through the Windows firewall, and through the
Hyper-V firewall that mirrored mode adds:

**PowerShell (admin)**
```powershell
New-NetFirewallRule -DisplayName "ATAS geometry" -Direction Inbound -Protocol TCP -LocalPort 18300 -Action Allow
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

Then set `GEOMETRY_SERVICE_URL=http://192.168.86.67:18300` on the Mac. The token travels in
plain text, so only do this on a network you trust.

## 9. Sharing the GPU with ComfyUI

- A reconstruction starts only when 24 GB of VRAM is free, and a render when 15 GB is free.
  Until then the job waits, and if it waits too long it fails with a retryable error.
- The ComfyUI adapter unloads its models after each clip. In the smoke test this took free VRAM
  from 5 GB to 31 GB.
- If you also use ComfyUI by hand while a story is running, its models stay loaded and geometry
  jobs will wait. Click *Unload models* in ComfyUI, or run:
  `curl -X POST http://192.168.86.67:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'`

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nvidia-smi` not found in WSL | Windows driver too old, or WSL not updated (`wsl --update`) |
| `torch.cuda.is_available()` is False | A CPU-only torch was installed; reinstall from the `cu128` index |
| `no kernel image is available for execution on the device` | A package was built without sm_120; for flash_attn, rebuild with `TORCH_CUDA_ARCH_LIST="12.0"` |
| Jobs sit in the queue | Not enough free VRAM: check `nvidia-smi` and unload ComfyUI's models |
| Everything is extremely slow | Sysmem fallback is still on, or `GEOMETRY_HOT_ROOT` is under `/mnt/c` |
| Mac cannot reach the service | Tunnel not running, `.wslconfig` not applied (`wsl --shutdown`), or firewall (LAN option) |
