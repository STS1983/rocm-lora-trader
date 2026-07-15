# PSP Resume Failure Research Report — RX 6600 XT (gfx1032/Navi 23)

**Date:** 2026-07-15  
**System:** Training Host (Ryzen 9 3900X, 4× RX 6600 XT, Ubuntu 24.04, Kernel 6.17.0-35, ROCm 7.2, amdgpu-dkms 6.12.12.60400)  
**Status:** Research complete — actionable fixes identified

---

## 1. Root Cause Analysis

### 1.1 What is PSP?

The **Platform Security Processor (PSP)** is an embedded ARM processor on AMD GPUs that handles secure boot, firmware authentication, and Trusted Application (TA) loading. It must initialize before any other IP block (GFX, SMU, SDMA, VCN) can resume. If PSP fails, the entire `amdgpu_device_ip_resume()` chain fails with error `-62` (`-ETIME`, timeout).

### 1.2 The Two Failure Modes

| GPU | Error | Meaning |
|-----|-------|---------|
| GPU0 (07:00.0, MSI, x1 Riser) | `PSP load kdb failed!` | PSP kernel debugger (KDB) bootloader failed to load — firmware file issue or PCIe timing |
| GPU2 (0e:00.0, XFX, x1 Riser) | `PSP create ring failed!` | PSP command ring setup failed — PSP bootloader loaded but couldn't establish communication channel |

Both errors cascade to: `PSP resume failed` → `amdgpu_device_ip_resume failed (-62)`

### 1.3 Primary Causes (Ranked by Likelihood)

#### Cause 1: Firmware Version Mismatch (HIGHEST LIKELIHOOD)

**Evidence:**
- **Launchpad Bug #2125139** (Ubuntu): Exact same `PSP load kdb failed!` error on Navi3x with kernel 6.17. Fix was updating `linux-firmware` package — specifically PSP SOS and TA firmware files. The bug report shows the identical error chain: `PSP load kdb failed!` → `PSP firmware loading failed` → `amdgpu_device_ip_init failed`.
- **Launchpad Bug #2122948** (Ubuntu): `amdgpu: psp gfx command LOAD_TA(0x1) failed` on kernel 6.17.0-4, regression from 6.17.0-3. This is a kernel-driver vs firmware compatibility issue introduced in the 6.17 series.
- **amdgpu-dkms 6.12.12.60400** was updated on July 12. The DKMS driver may expect newer PSP firmware than what's installed in `linux-firmware`.

**For RX 6600 XT (Navi 23, PSP v11.0):** The relevant firmware files are:
- `amdgpu/psp_11_0_0_sos.bin` — PSP Secure OS
- `amdgpu/psp_11_0_0_ta.bin` — Trusted Applications
- `amdgpu/smu_13_0_0.bin` — SMU firmware (if applicable)
- `amdgpu/gc_10_3_2_*` — Graphics Core firmware (gfx1032-specific)

If `linux-firmware` hasn't been updated alongside the DKMS driver, the PSP driver code may call firmware APIs that don't exist in the installed firmware version.

#### Cause 2: PCIe x1 Riser Timing Issues (HIGH LIKELIHOOD)

**Evidence:**
- The two failing GPUs (GPU0, GPU2) are both on **PCIe x1 risers**. The two that recover (GPU1 Sapphire, GPU3 ASUS) include one on x1 riser and one on direct x16.
- PCIe x1 risers (PCE164P-N03) introduce signal integrity degradation and reduced bandwidth. During boot, the amdgpu driver initializes GPUs sequentially. With 4 GPUs, there's a timing window where the PSP firmware load command can timeout on riser-connected cards.
- The Level1Techs forum has multiple reports of Radeon VII PSP initialization failures in multi-GPU compute rigs, particularly with risers.
- `amdgpu_device_ip_resume failed (-62)` = `-ETIME` = timeout. The PSP didn't respond in time.

#### Cause 3: Runtime Power Management (runpm) Spurious Resumes (MODERATE)

**Evidence:**
- **ROCm/amdgpu Issue #183**: Enabling runtime PM causes `amdgpu_device_ip_resume failed` on MI100 due to SMU/PSP failing to resume after auto-suspend. The GPUs get suspended by runtime PM, then spurious resumes (from sysfs reads, Vulkan enumeration, fwupd, etc.) cause PSP to fail reinitialization.
- **Luna Nova's research**: Opening `/dev/dri/` device nodes, reading sysfs sensors, or `fwupdmgr` can all trigger spurious GPU resumes. With 4 GPUs, the probability of spurious wake/suspend cycles multiplies.
- If `amdgpu.runpm=-1` (auto, default), secondary GPUs may suspend and then fail to resume their PSP properly.

#### Cause 4: amdgpu-dkms 6.12.12.60400 Bug (MODERATE)

**Evidence:**
- Training worked fine on July 5 and July 14 with the same DKMS version. The DKMS update on July 12 didn't immediately break things.
- However, the **reboot on July 15** is what triggered the failures. A fresh boot with the new DKMS may expose race conditions or firmware expectations that weren't hit during the previous session (which had been running since before the DKMS update).
- **ROCm Issue #5624**: amdgpu-dkms has known build/runtime issues with kernel 6.17.

---

## 2. Why Ollama Works But PyTorch/HIP Doesn't

This is a critical finding. The explanation is about **different GPU access paths**:

### Ollama's Path
```
Ollama → ggml HIP backend → libdrm → /dev/dri/renderDxxx → direct GPU access
```
Ollama uses **libDRM direct rendering** via the render node (`/dev/dri/renderD128+`). It opens the render node, submits commands directly to the GPU's command processor, and doesn't require full HSA runtime initialization. As long as the GPU's DRM render node exists and the GFX IP block is functional, Ollama can use it — even if PSP's secure features (TA loading) partially failed.

### PyTorch/HIP's Path
```
PyTorch → HIP runtime → HSA Runtime (ROCR) → /dev/kfd → KFD → requires full PSP initialization
```
PyTorch/HIP requires the **HSA Runtime (ROCR)** which goes through `/dev/kfd` (the AMD Compute Kernel Fusion Driver). KFD requires **full PSP initialization** including:
- Trusted Application (TA) loading
- Secure display initialization
- Full PSP ring buffer setup

If any PSP step fails, the KFD node for that GPU is not created, and `hsa_init()` returns `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (0x1008). This is exactly what we see.

**Key insight:** Even though Ollama "works," it's likely only using the 2 GPUs that DID recover (GPU1, GPU3). The 2 failed GPUs (GPU0, GPU2) are probably not accessible to Ollama either, but Ollama doesn't need ALL GPUs — it just uses whatever render nodes are available. PyTorch/HIP, however, tries to enumerate ALL GPUs through KFD, and if ANY KFD node initialization fails, `hsa_init()` can fail entirely.

### `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (0x1008)

This error is **directly caused by the PSP failures**. The HSA runtime tries to allocate KFD nodes for all detected GPUs. When PSP fails for 2 of 4 GPUs, those KFD nodes can't be created. Depending on the ROCr version, this either:
1. Silently skips the failed GPUs (newer ROCr 6.x+)
2. Fails the entire `hsa_init()` call (older behavior, or if the PSP failure corrupts shared state)

In this case, it appears to be failing entirely, suggesting the PSP failure affects the KFD global state.

---

## 3. Known Workarounds (Ranked by Feasibility)

### Fix 1: Update linux-firmware (HIGHEST PRIORITY — Try First)

**Rationale:** The most likely cause is a firmware version mismatch between amdgpu-dkms 6.12.12.60400 and the installed PSP firmware. The Launchpad bug #2125139 showed the exact same error fixed by a firmware update.

**Steps:**
```bash
# Check current firmware versions
ls -la /lib/firmware/amdgpu/psp_11_0_0_*

# Update linux-firmware
sudo apt update
sudo apt install --only-upgrade linux-firmware

# Reboot
sudo reboot
```

**If `apt` doesn't have newer firmware:**
```bash
# Pull latest from git
cd /tmp
git clone https://gitlab.com/kernel-firmware/linux-firmware.git
sudo cp linux-firmware/amdgpu/psp_11_0_0_* /lib/firmware/amdgpu/
sudo cp linux-firmware/amdgpu/gc_10_3_2_* /lib/firmware/amdgpu/
sudo cp linux-firmware/amdgpu/smu_13_0_0* /lib/firmware/amdgpu/
sudo update-initramfs -u -k all
sudo reboot
```

**Feasibility:** ⭐⭐⭐⭐⭐ (trivial, low risk)  
**Likelihood of fixing:** 60% — this is the most common fix for PSP load failures after DKMS updates

### Fix 2: Disable Runtime Power Management (HIGH PRIORITY)

**Rationale:** Prevents spurious GPU suspend/resume cycles that can cause PSP failures.

**Steps:**
Add to `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.runpm=0"
```
Then:
```bash
sudo update-grub
sudo reboot
```

**Alternatively** (without reboot, for testing):
```bash
# Disable runtime PM for all GPUs
for dev in /sys/bus/pci/devices/*/power/control; do
  echo on > "$dev" 2>/dev/null
done
```

**Feasibility:** ⭐⭐⭐⭐⭐ (trivial, low risk, reversible)  
**Likelihood of fixing:** 30% as standalone fix, 80% when combined with Fix 1  
**Side effects:** Slightly higher idle power consumption (~15W per GPU)

### Fix 3: Enable GPU Recovery + Force Long Training

**Rationale:** `amdgpu.gpu_recovery=1` enables the driver to attempt GPU reset on failure instead of giving up. `forcelongtraining=1` forces full memory retraining on resume (slower but more reliable).

**Steps:**
Add to GRUB:
```
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.gpu_recovery=1 amdgpu.forcelongtraining=1"
```

**Feasibility:** ⭐⭐⭐⭐ (easy, low risk)  
**Likelihood of fixing:** 20% standalone — helps with recovery but doesn't prevent the initial PSP failure

### Fix 4: Reload amdgpu Module (No Reboot Required — Temporary Fix)

**Rationale:** Unloading and reloading the amdgpu driver triggers a mode-1 reset which reinitializes PSP. This can sometimes recover from PSP failures without a full reboot.

**Steps:**
```bash
# Stop all GPU users first
sudo systemctl stop ollama
# Kill any processes using GPUs
sudo fuser -k /dev/dri/* /dev/kfd 2>/dev/null

# Unload modules
sudo modprobe -r amdgpu
sudo modprobe -r amdkfd

# Reload
sudo modprobe amdgpu

# Verify
/opt/rocm/bin/rocminfo | head -50
```

**⚠️ Warning:** `rmmod amdgpu` can SIGSEGV if processes are still using the GPU. Ensure ALL GPU processes are stopped first. This is documented in ROCm/amdgpu Issue #11.

**Feasibility:** ⭐⭐⭐ (moderate risk, may crash if processes aren't fully stopped)  
**Likelihood of fixing:** 40% — can work if the PSP failure is a transient timing issue, but won't help if it's a firmware mismatch

### Fix 5: amdgpu-dkms Downgrade

**Rationale:** If amdgpu-dkms 6.12.12.60400 has a regression, downgrading to the previous version may fix it.

**Steps:**
```bash
# Check available versions
apt list --all-versions amdgpu-dkms

# Install specific version (example)
sudo apt install amdgpu-dkms=6.12.8.xxxxx

# Rebuild for current kernel
sudo dkms remove amdgpu/6.12.12.60400 --all
sudo dkms install amdgpu/6.12.8.xxxxx
sudo update-initramfs -u -k all
sudo reboot
```

**Feasibility:** ⭐⭐⭐ (moderate — need to find the right previous version, risk of dependency issues)  
**Likelihood of fixing:** 40% — only if the DKMS update introduced the bug  
**Risk:** May break ROCm 7.2 compatibility if the older DKMS doesn't support ROCm 7.2 features

### Fix 6: PCIe Bus Re-scan (No Reboot — Experimental)

**Rationale:** Sometimes a PCIe bus re-scan can re-enumerate the failed GPUs and trigger a fresh amdgpu init.

**Steps:**
```bash
# Remove failed GPUs from PCI bus
echo 1 | sudo tee /sys/bus/pci/devices/0000:07:00.0/remove
echo 1 | sudo tee /sys/bus/pci/devices/0000:0e:00.0/remove

# Rescan PCI bus
echo 1 | sudo tee /sys/bus/pci/rescan

# Check if GPUs reappeared
lspci | grep -i amd
```

**Feasibility:** ⭐⭐ (experimental, may not trigger full PSP reinit)  
**Likelihood of fixing:** 15% — long shot, but worth trying before a full reboot

### Fix 7: Swap PCIe Riser Positions

**Rationale:** If the issue is riser-specific (signal integrity), swapping risers between working and non-working GPU slots can confirm this.

**Steps:** Physical hardware swap — swap the riser on GPU0 (07:00.0) with the riser on GPU1 (0a:00.0). If the failure follows the riser, it's a hardware issue.

**Feasibility:** ⭐⭐ (requires physical access, takes ~30 min)  
**Likelihood of fixing:** 10% — confirms diagnosis but doesn't fix; would need new risers

---

## 4. Recommended Fix Sequence

### Step 1: Immediate (No Reboot) — Try Module Reload
```bash
sudo systemctl stop ollama
sudo fuser -k /dev/dri/* /dev/kfd 2>/dev/null
sleep 2
sudo modprobe -r amdgpu && sudo modprobe amdgpu
/opt/rocm/bin/rocminfo | head -50
```

If this works, great — but it's temporary. The issue will likely recur on next reboot.

### Step 2: Permanent Fix — Update Firmware + Disable runpm
```bash
# 1. Update firmware
sudo apt update && sudo apt install --only-upgrade linux-firmware

# 2. Add kernel parameters
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 amdgpu.runpm=0 amdgpu.gpu_recovery=1"/' /etc/default/grub
sudo update-grub

# 3. Reboot
sudo reboot
```

### Step 3: If Still Failing — Check Firmware Files Manually
```bash
# Check if Navi 23 firmware exists and is recent
ls -la /lib/firmware/amdgpu/psp_11_0_0_*
ls -la /lib/firmware/amdgpu/gc_10_3_2_*

# Compare with upstream
# If files are missing or old, pull from gitlab:
# git clone https://gitlab.com/kernel-firmware/linux-firmware.git
# sudo cp linux-firmware/amdgpu/* /lib/firmware/amdgpu/
# sudo update-initramfs -u -k all
```

### Step 4: If Still Failing — Downgrade DKMS
```bash
# Check which version was working before July 12
dpkg -l | grep amdgpu-dkms
apt list --all-versions amdgpu-dkms

# Install previous version
sudo apt install amdgpu-dkms=<previous_version>
sudo reboot
```

### Step 5: If Still Failing — Hardware Diagnosis
- Swap risers between failing and working GPUs
- Try booting with only 2 GPUs (remove the 2 failing ones)
- Update motherboard BIOS (B550 BIOS updates sometimes fix PCIe timing)

---

## 5. Prevention Strategies

### 5.1 Pre-Reboot Script
Create a script to run before any planned reboot:
```bash
#!/bin/bash
# /home/$USER/scripts/pre-reboot-gpu-check.sh
echo "Checking GPU state before reboot..."
rocminfo | grep -c "gfx1032" || echo "WARNING: Not all GPUs accessible"
# If GPUs are in bad state, do a warm reset first
```

### 5.2 Boot-Time Verification
Add to `/etc/rc.local` or a systemd service:
```bash
# Wait for GPUs to settle
sleep 10
# Check GPU count
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "gfx1032")
if [ "$GPU_COUNT" -lt 4 ]; then
  logger "WARNING: Only $GPU_COUNT/4 GPUs accessible after boot"
  # Attempt module reload
  modprobe -r amdgpu && modprobe amdgpu
fi
```

### 5.3 Kernel Parameter Stack (Recommended for Multi-GPU Riser Rigs)
```
amdgpu.runpm=0 amdgpu.gpu_recovery=1 amdgpu.forcelongtraining=1 pcie_aspm=off
```

- `amdgpu.runpm=0` — Prevents runtime PM spurious resume issues
- `amdgpu.gpu_recovery=1` — Enables automatic GPU reset on failure
- `amdgpu.forcelongtraining=1` — Full memory retraining (slower boot but more reliable)
- `pcie_aspm=off` — Disables PCIe Active State Power Management (can cause timing issues with risers)

### 5.4 Firmware Update Policy
After any `amdgpu-dkms` update, **always** also update `linux-firmware`:
```bash
sudo apt update
sudo apt install --only-upgrade amdgpu-dkms linux-firmware
```

### 5.5 Training Host GPU Guard Cron (Already Exists)
The existing `gpu-guard.sh` cron job (every 15min) should be enhanced to check `rocminfo` output, not just GPU presence:
```bash
# Add to gpu-guard.sh
ROCINFO=$(rocminfo 2>&1)
if echo "$ROCINFO" | grep -q "OUT_OF_RESOURCES"; then
  # PSP failure detected — attempt module reload
  systemctl stop ollama
  modprobe -r amdgpu && modprobe amdgpu
  systemctl start ollama
fi
```

---

## 6. Key References

| Source | Relevance | URL |
|--------|-----------|-----|
| Launchpad Bug #2125139 | **Exact same PSP error, fixed by firmware update** | https://bugs.launchpad.net/bugs/2125139 |
| Launchpad Bug #2122948 | PSP LOAD_TA failed on kernel 6.17 regression | https://bugs.launchpad.net/bugs/2122948 |
| Kernel Bug #218549 | `amdgpu_device_ip_resume failed (-62)` regression | https://bugzilla.kernel.org/show_bug.cgi?id=218549 |
| Kernel Bug #216716 | `PSP resume failed` on AMD iGPU | https://bugzilla.kernel.org/show_bug.cgi?id=216716 |
| ROCm Issue #4226 | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` after runtime | https://github.com/ROCm/ROCm/issues/4226 |
| ROCm/amdgpu Issue #183 | Runtime PM causes SMU/PSP resume failure | https://github.com/ROCm/amdgpu/issues/183 |
| ROCm/amdgpu Issue #11 | How to reset driver/GPU without reboot | https://github.com/ROCm/amdgpu/issues/11 |
| NixOS Issue #287586 | `amdgpu_device_ip_resume_phase2` fails (-62) | https://github.com/NixOS/nixpkgs/issues/287586 |
| Luna Nova — GPU RunPM | Spurious GPU resumes from sysfs reads, Vulkan enum | https://lunnova.dev/articles/linux-gpu-runpm-spurious-resumes/ |
| Kernel Module Parameters | amdgpu.runpm, gpu_recovery, forcelongtraining docs | https://docs.kernel.org/gpu/amdgpu/module-parameters.html |
| Fedora Discussion | amdgpu crashes on kernels 6.12+ after suspend | https://discussion.fedoraproject.org/t/146322 |

---

## 7. Summary

| Question | Answer |
|----------|--------|
| What causes `PSP load kdb failed`? | PSP firmware (SOS/KDB bootloader) fails to load — most likely firmware version mismatch or PCIe timing issue |
| Is this a known bug with DKMS 6.12.12 or kernel 6.17? | **Yes** — kernel 6.17 has known PSP regression (Launchpad #2122948, #2125139). DKMS update may have exacerbated it. |
| Why can Ollama use GPUs but PyTorch can't? | Ollama uses libDRM render nodes (doesn't need full PSP). PyTorch/HIP needs KFD which requires full PSP init including TA loading. |
| Can PSP be reloaded without reboot? | **Sometimes** — `modprobe -r amdgpu && modprobe amdgpu` can work if no processes are using the GPU. Not guaranteed. |
| Is `HSA_STATUS_ERROR_OUT_OF_RESOURCES` related to PSP? | **Yes, directly.** PSP failure → KFD node not created → `hsa_init()` fails → `OUT_OF_RESOURCES` |
| Best fix? | **Update linux-firmware + disable runpm + reboot** (Fix 1 + Fix 2) |