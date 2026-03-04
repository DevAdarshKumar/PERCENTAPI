# perchance_server_with_pyvirtualdisplay.py
"""
Perchance Image-Generation Server v2.0
This variant adds optional pyvirtualdisplay support so the server can be
hosted on headless environments (Hugging Face Spaces etc.) while keeping
all original behaviour unchanged.
Behaviour:
 - If ZD_HEADLESS is True, zendriver will run headless as before.
 - If ZD_HEADLESS is False and USE_VIRTUAL_DISPLAY is True and a DISPLAY
   is not present, we attempt to start a pyvirtualdisplay.Display (Xvfb)
   automatically before launching browsers. If pyvirtualdisplay is not
   installed or starting Xvfb fails, we log a warning and continue.
 - If USE_VIRTUAL_DISPLAY is False we will NOT attempt to start a virtual
   display — you must provide a DISPLAY yourself (or set ZD_HEADLESS=True)
   if running on a headless host.
To run on Hugging Face Spaces, add `pyvirtualdisplay` to requirements.txt
and ensure `xvfb` is available in the runtime (HF Spaces typically provide it).
All original defaults and constants are preserved from the original file.
"""

import asyncio
import base64
import json
import logging
import os
import random
import string
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudscraper
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
import zendriver as zd
from zendriver import cdp

# Try to import pyvirtualdisplay (optional)
try:
    from pyvirtualdisplay import Display
    _HAS_PYVIRTUALDISPLAY = True
except Exception:
    Display = None
    _HAS_PYVIRTUALDISPLAY = False

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# --- Perchance API ---
BASE_URL        = "https://image-generation.perchance.org"
API_GENERATE    = "/api/generate"
API_DOWNLOAD    = "/api/downloadTemporaryImage"
API_AWAIT       = "/api/awaitExistingGenerationRequest"
API_ACCESS_CODE = "/api/getAccessCodeForAdPoweredStuff"

# --- Browser automation (zendriver) ---
TARGET_URL       = "https://perchance.org/ai-text-to-image-generator"
IMAGE_GEN_ORIGIN = "https://image-generation.perchance.org"
ZD_TIMEOUT       = 90          # seconds for key-fetch attempt
ZD_HEADLESS      = False       # True → hide browser window
CLICK_INTERVAL   = 0.35
CLICK_JITTER     = 8.0
KEY_PREFIX       = "userKey"

# --- Virtual display toggle (new) ---
# If True the server will attempt to auto-start a pyvirtualdisplay (Xvfb)
# when no DISPLAY is present and ZD_HEADLESS is False. If False the server
# will not try to start a virtual display and you must provide a DISPLAY
# or set ZD_HEADLESS=True.
USE_VIRTUAL_DISPLAY = True

# --- HTTP / generation ---
HTTP_TIMEOUT      = 30
MAX_DOWNLOAD_WAIT = 180
BACKOFF_INIT      = 0.7
MAX_GEN_RETRIES   = 6          # retries inside generate_one()

# --- Key-refresh policy ---
MAX_KEY_RETRIES       = 3      # per-image retries when key is invalid
KEY_REFRESH_COOLDOWN  = 30     # min seconds between two refreshes
MAX_REFRESH_FAILURES  = 5      # consecutive failures → stop auto-refresh

# --- Server ---
WORKER_COUNT     = 3
MAX_QUEUE_SIZE   = 1000
EXECUTOR_THREADS = 16
OUTPUT_DIR       = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ═══════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("perchance")


# ═══════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

USER_KEY: Optional[str] = None

# -- set in lifespan --
_key_lock:          Optional[asyncio.Lock]  = None   # guard USER_KEY reads/writes
_key_valid:         Optional[asyncio.Event] = None   # cleared while refreshing
_key_refresh_lock:  Optional[asyncio.Lock]  = None   # one refresh at a time
_key_last_ts:       float                   = 0.0    # last successful refresh
_key_fail_count:    int                     = 0      # consecutive refresh failures

JOB_QUEUE:          Optional[asyncio.Queue] = None

TASKS:       Dict[str, Dict[str, Any]]  = {}
TASK_QUEUES: Dict[str, asyncio.Queue]   = {}         # SSE event queues

EXECUTOR = ThreadPoolExecutor(max_workers=EXECUTOR_THREADS)
SCRAPER  = cloudscraper.create_scraper()

# pyvirtualdisplay handle (optional)
VDISPLAY: Optional[Display] = None


# ═══════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe(s: str) -> str:
    """Sanitise string for filenames."""
    ok = set(string.ascii_letters + string.digits + "-_.()")
    return "".join(c if c in ok else "_" for c in s)[:120]


def _sid() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _reqid() -> str:
    return f"{time.time():.6f}-{_sid()}"


def _stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


# ═══════════════════════════════════════════════════════════════
#  Virtual display helpers (pyvirtualdisplay)
# ═══════════════════════════════════════════════════════════════

def _start_virtual_display_if_needed(headless: bool):
    """
    Start pyvirtualdisplay.Display() if we're running non-headless in an
    environment without DISPLAY and USE_VIRTUAL_DISPLAY is True.
    This function is synchronous and safe to be run in a thread executor.
    """
    global VDISPLAY

    if headless:
        log.info("ZD_HEADLESS=True → not starting virtual display")
        return

    if not USE_VIRTUAL_DISPLAY:
        log.info("USE_VIRTUAL_DISPLAY=False → not starting virtual display; expecting manual DISPLAY or headless mode.")
        return

    if os.environ.get("DISPLAY"):
        log.info("DISPLAY already set: %s", os.environ.get("DISPLAY"))
        return

    if not _HAS_PYVIRTUALDISPLAY or Display is None:
        log.warning(
            "pyvirtualdisplay not installed — cannot create virtual DISPLAY. "
            "Install pyvirtualdisplay in your environment to enable Xvfb.")
        return

    try:
        VDISPLAY = Display(visible=0, size=(1280, 720))
        VDISPLAY.start()
        # pyvirtualdisplay sets DISPLAY env itself; log for visibility
        log.info("Started virtual display via pyvirtualdisplay (DISPLAY=%s)", os.environ.get("DISPLAY"))
    except Exception as exc:
        VDISPLAY = None
        log.exception("Failed to start virtual display: %s", exc)


def _stop_virtual_display_if_needed():
    global VDISPLAY
    if VDISPLAY is None:
        return
    try:
        VDISPLAY.stop()
        log.info("Stopped virtual display")
    except Exception:
        log.exception("Error while stopping virtual display")
    finally:
        VDISPLAY = None


# ═══════════════════════════════════════════════════════════════
#  PERCHANCE HTTP CLIENT  (blocking – runs in ThreadPoolExecutor)
# ═══════════════════════════════════════════════════════════════

class PerchanceClient:
    """All blocking HTTP work against the Perchance API."""

    def __init__(self):
        self.base = BASE_URL.rstrip("/")
        self.s    = SCRAPER
        self.h    = {
            "Accept":       "*/*",
            "Content-Type": "application/json;charset=UTF-8",
            "Origin":       IMAGE_GEN_ORIGIN,
            "Referer":      f"{IMAGE_GEN_ORIGIN}/embed",
            "User-Agent":   (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        }

    # ---- low-level helpers ----

    def get_ad_code(self) -> str:
        try:
            r = self.s.get(
                f"{self.base}{API_ACCESS_CODE}",
                timeout=HTTP_TIMEOUT, headers=self.h,
            )
            r.raise_for_status()
            return r.text.strip()
        except Exception:
            return ""

    def _post(self, body: dict, params: dict) -> dict:
        try:
            r = self.s.post(
                f"{self.base}{API_GENERATE}",
                json=body, params=params,
                timeout=HTTP_TIMEOUT, headers=self.h,
            )
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {"status": "invalid_json", "raw": r.text}
        except Exception as exc:
            return {"status": "fetch_failure", "error": str(exc)}

    def _await_prev(self, key: str):
        try:
            self.s.get(
                f"{self.base}{API_AWAIT}",
                params={"userKey": key, "__cacheBust": random.random()},
                timeout=20, headers=self.h,
            )
        except Exception:
            pass

    # ---- generate one image ----

    def generate_one(
        self, *,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        resolution: str = "512x768",
        guidance_scale: float = 7.0,
        channel: str = "ai-text-to-image-generator",
        sub_channel: str = "private",
        user_key: str = "",
        ad_access_code: str = "",
        request_id: str = "",
    ) -> dict:
        """
        Returns ONE of:
          {"imageId": ..., "seed": ...}
          {"inline": ...,  "seed": ...}
          {"error": "invalid_key"}           ← caller must refresh key
          {"error": "<other>", ...}
        """
        request_id = request_id or _reqid()
        params = {
            "userKey":       user_key,
            "requestId":     request_id,
            "adAccessCode":  ad_access_code,
            "__cacheBust":   random.random(),
        }
        body = {
            "prompt":         prompt,
            "negativePrompt": negative_prompt,
            "seed":           seed,
            "resolution":     resolution,
            "guidanceScale":  guidance_scale,
            "channel":        channel,
            "subChannel":     sub_channel,
            "userKey":        user_key,
            "adAccessCode":   ad_access_code,
            "requestId":      request_id,
        }

        ad_refreshed = False

        for att in range(1, MAX_GEN_RETRIES + 1):
            res = self._post(body, params)
            st  = res.get("status")

            # ---- success ----
            if st == "success":
                iid  = res.get("imageId")
                urls = res.get("imageDataUrls")
                if iid:
                    log.info("Got imageId: %s", iid)
                    return {"imageId": iid, "seed": res.get("seed")}
                if urls:
                    return {"inline": urls[0], "seed": res.get("seed")}
                log.error("success but empty payload: %s", str(res)[:300])
                return {"error": "empty_success", "raw": res}

            # ---- invalid key → return immediately (do NOT retry here) ----
            if st == "invalid_key":
                log.warning("Server says invalid_key")
                return {"error": "invalid_key"}

            # ---- previous request still running ----
            if st == "waiting_for_prev_request_to_finish":
                log.info("Waiting for prev request to finish …")
                self._await_prev(user_key)
                time.sleep(0.3 + random.random() * 0.3)
                continue

            # ---- ad access code expired ----
            if st == "invalid_ad_access_code" and not ad_refreshed:
                code = self.get_ad_code()
                if code:
                    ad_access_code = code
                    params["adAccessCode"] = code
                    body["adAccessCode"]   = code
                    ad_refreshed = True
                    log.info("Refreshed ad code → retry")
                    time.sleep(0.8)
                    continue
                return {"error": "invalid_ad_access_code"}

            # ---- transient gen failure ----
            if st == "gen_failure" and res.get("type") == 1:
                log.warning("gen_failure type 1 → retry after 2.5 s")
                time.sleep(2.5)
                continue

            # ---- network / stale ----
            if st in (None, "fetch_failure", "invalid_json", "stale_request"):
                log.info("Transient error (status=%s) attempt %d/%d", st, att, MAX_GEN_RETRIES)
                time.sleep(1.0)
                continue

            # ---- anything else ----
            log.error("Unhandled status '%s': %s", st, str(res)[:300])
            return {"error": f"unhandled_{st}", "raw": res}

        return {"error": "max_retries_exceeded"}

    # ---- download ----

    def download_image(self, image_id: str, prefix: str = "img") -> str:
        """Poll until the image is ready, save to OUTPUT_DIR, return path."""
        url = f"{self.base}{API_DOWNLOAD}?imageId={image_id}"
        t0  = time.time()
        bk  = BACKOFF_INIT

        while True:
            elapsed = time.time() - t0
            if elapsed >= MAX_DOWNLOAD_WAIT:
                raise TimeoutError(
                    f"Download timed out ({elapsed:.0f}s) for {image_id}"
                )
            try:
                r = self.s.get(url, timeout=HTTP_TIMEOUT,
                               headers=self.h, stream=True)
                if r.status_code == 200:
                    ct  = r.headers.get("Content-Type", "")
                    ext = (
                        ".png"  if "png"  in ct else
                        ".webp" if "webp" in ct else ".jpg"
                    )
                    fn = _safe(f"{prefix}_{image_id[:12]}{ext}")
                    fp = str(OUTPUT_DIR / fn)
                    with open(fp, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    log.info("Saved → %s", fp)
                    return fp
            except Exception:
                pass

            time.sleep(bk)
            bk = min(bk * 1.8, 8.0)


CLIENT = PerchanceClient()


# ═══════════════════════════════════════════════════════════════
#  ZENDRIVER – browser automation to extract userKey
# ═══════════════════════════════════════════════════════════════

async def _cdp_mouse(tab, typ, x, y, **kw):
    await tab.send(
        cdp.input_.dispatch_mouse_event(
            type_=typ, x=float(x), y=float(y), **kw,
        )
    )


async def _viewport_center(tab):
    try:
        v = await tab.evaluate(
            "(()=>({w:innerWidth,h:innerHeight}))()",
            await_promise=False, return_by_value=True,
        )
        return (v["w"] / 2.0, v["h"] / 2.0)
    except Exception:
        return (600.0, 400.0)


async def _ls_get(tab, key):
    try:
        return await tab.evaluate(
            f"localStorage&&localStorage.getItem({json.dumps(key)})",
            await_promise=True, return_by_value=True,
        )
    except Exception:
        return None


async def _clicker_loop(tab, stop: asyncio.Event):
    """Simulate steady centre-clicks on *tab* until *stop* is set."""
    try:
        await tab.evaluate(
            "window.focus&&window.focus()",
            await_promise=False, return_by_value=False,
        )
    except Exception:
        pass

    centre     = await _viewport_center(tab)
    centre_upd = time.time()

    while not stop.is_set():
        if time.time() - centre_upd > 2.5:
            centre     = await _viewport_center(tab)
            centre_upd = time.time()

        jx = random.uniform(-CLICK_JITTER, CLICK_JITTER)
        jy = random.uniform(-CLICK_JITTER, CLICK_JITTER)
        cx, cy = centre[0] + jx, centre[1] + jy

        try:
            await _cdp_mouse(tab, "mouseMoved", cx, cy, pointer_type="mouse")
            await asyncio.sleep(random.uniform(0.02, 0.08))
            await _cdp_mouse(
                tab, "mousePressed", cx, cy,
                button=cdp.input_.MouseButton.LEFT,
                click_count=1, buttons=1,
            )
            await asyncio.sleep(random.uniform(0.03, 0.12))
            await _cdp_mouse(
                tab, "mouseReleased", cx, cy,
                button=cdp.input_.MouseButton.LEFT,
                click_count=1, buttons=0,
            )
        except Exception:
            pass

        # interruptible sleep
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=CLICK_INTERVAL * random.uniform(0.85, 1.15),
            )
            break
        except asyncio.TimeoutError:
            pass


async def _poll_for_key(tab, stop: asyncio.Event, max_sec: int):
    """Poll localStorage every 250 ms for a userKey entry."""
    t0 = time.time()
    while not stop.is_set() and (time.time() - t0) < max_sec:
        val = await _ls_get(tab, f"{KEY_PREFIX}-0")
        if val:
            return val
        try:
            keys = await tab.evaluate(
                "Object.keys(localStorage||{}).filter(k=>k.includes('userKey'))",
                await_promise=False, return_by_value=True,
            )
            for k in (keys or []):
                v = await _ls_get(tab, k)
                if v:
                    return v
        except Exception:
            pass
        await asyncio.sleep(0.25)
    return None


async def fetch_key_via_browser(
    timeout: int = ZD_TIMEOUT,
    headless: bool = ZD_HEADLESS,
) -> Optional[str]:
    """
    Launch Chrome → navigate to Perchance → click to trigger
    ad/verification → read userKey from localStorage → close browser.
    Returns the key string or None.
    """
    log.info(
        "Launching browser for userKey (timeout=%ds, headless=%s)",
        timeout, headless,
    )

    # If we're in non-headless mode on a display-less host, ensure a virtual
    # DISPLAY is started first. This call is synchronous so we run it in the
    # event loop's default executor when called from async code.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, partial(_start_virtual_display_if_needed, headless))
    except Exception:
        log.exception("Error while attempting to start virtual display")

    try:
        browser = await zd.start(headless=headless ,no_sandbox=True)
    except Exception as exc:
        log.exception("Browser start failed: %s", exc)
        return None

    stop   = asyncio.Event()
    result = None

    try:
        page_tab = await browser.get(TARGET_URL)
        log.info("Opened %s", TARGET_URL)
        await asyncio.sleep(2.0)

        origin_tab = await browser.get(IMAGE_GEN_ORIGIN, new_tab=True)
        log.info("Opened %s", IMAGE_GEN_ORIGIN)
        await asyncio.sleep(1.0)

        await page_tab.bring_to_front()
        await asyncio.sleep(0.5)

        clicker = asyncio.create_task(_clicker_loop(page_tab, stop))
        poller  = asyncio.create_task(_poll_for_key(origin_tab, stop, timeout))

        try:
            done, _ = await asyncio.wait({poller}, timeout=timeout)
            if poller in done:
                result = poller.result()
        finally:
            stop.set()
            if not clicker.done():
                clicker.cancel()
                try:
                    await clicker
                except asyncio.CancelledError:
                    pass

        for t in (origin_tab, page_tab):
            try:
                await t.close()
            except Exception:
                pass
    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    if result:
        log.info("Fetched userKey (len=%d)", len(result))
    else:
        log.warning("Could not fetch userKey within %ds", timeout)
    return result


# ═══════════════════════════════════════════════════════════════
#  KEY MANAGEMENT – coordinated refresh across workers
# ═══════════════════════════════════════════════════════════════

async def _broadcast(event: dict):
    """Push an event into every active task's SSE queue."""
    for tid, q in TASK_QUEUES.items():
        task = TASKS.get(tid)
        if task and task["status"] in ("queued", "running"):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


async def refresh_user_key() -> Optional[str]:
    """
    Coordinate a single key refresh.  If another coroutine is already
    refreshing, we simply wait for it to finish and return the new key.
    Returns the new key string, or None on failure.
    """
    global USER_KEY, _key_last_ts, _key_fail_count

    async with _key_refresh_lock:
        # ── double-check: maybe another coroutine just refreshed ──
        age = time.time() - _key_last_ts
        if age < KEY_REFRESH_COOLDOWN and USER_KEY:
            log.info(
                "Key was refreshed %.1fs ago → reusing existing key", age,
            )
            return USER_KEY

        # ── too many consecutive failures? ──
        if _key_fail_count >= MAX_REFRESH_FAILURES:
            log.error(
                "Key refresh disabled: %d consecutive failures. "
                "Set key manually via POST /set_user_key",
                _key_fail_count,
            )
            await _broadcast({
                "type": "key_refresh_failed",
                "time": _now(),
                "message": (
                    f"Auto-refresh disabled after {_key_fail_count} failures. "
                    "Please set userKey manually via /set_user_key"
                ),
            })
            return None

        # ── signal "key is being refreshed" ──
        _key_valid.clear()
        log.info("Starting userKey refresh via browser …")

        await _broadcast({
            "type":    "key_refreshing",
            "time":    _now(),
            "message": "UserKey expired — refreshing via browser automation …",
        })

        try:
            new_key = await fetch_key_via_browser(
                timeout=ZD_TIMEOUT, headless=ZD_HEADLESS,
            )

            if new_key:
                async with _key_lock:
                    USER_KEY = new_key
                _key_last_ts   = time.time()
                _key_fail_count = 0

                log.info("UserKey refreshed OK (len=%d)", len(new_key))
                await _broadcast({
                    "type":    "key_refreshed",
                    "time":    _now(),
                    "message": "UserKey refreshed – resuming generation.",
                })
                return new_key

            # fetch returned None
            _key_fail_count += 1
            log.error(
                "Key refresh returned nothing (failure #%d/%d)",
                _key_fail_count, MAX_REFRESH_FAILURES,
            )
            await _broadcast({
                "type":    "key_refresh_failed",
                "time":    _now(),
                "message": (
                    f"Key refresh failed (attempt {_key_fail_count}"
                    f"/{MAX_REFRESH_FAILURES})"
                ),
            })
            return None

        except Exception as exc:
            _key_fail_count += 1
            log.exception(
                "Key refresh error (failure #%d/%d): %s",
                _key_fail_count, MAX_REFRESH_FAILURES, exc,
            )
            await _broadcast({
                "type":    "key_refresh_failed",
                "time":    _now(),
                "message": f"Key refresh error: {exc}",
            })
            return None

        finally:
            # ALWAYS unblock waiters, even on failure
            _key_valid.set()


# ═══════════════════════════════════════════════════════════════
#  TASK MODEL
# ═══════════════════════════════════════════════════════════════

def create_task(
    prompts: List[str],
    count: int,
    resolution: str,
    guidance: float,
    negative: str,
    sub_channel: str,
) -> dict:
    tid = str(uuid.uuid4())
    task = {
        "id":           tid,
        "prompts":      prompts,
        "count":        count,
        "resolution":   resolution,
        "guidance":     guidance,
        "negative":     negative,
        "sub_channel":  sub_channel,
        "created_at":   _now(),
        "status":       "queued",        # queued → running → done / failed
        "total_images": len(prompts) * count,
        "completed":    0,
        "results":      [],
        "error":        None,
    }
    TASKS[tid]       = task
    TASK_QUEUES[tid] = asyncio.Queue()
    return task


# ═══════════════════════════════════════════════════════════════
#  WORKER — image generation + key-refresh retry loop
# ═══════════════════════════════════════════════════════════════

async def _save_inline(data_url: str, prompt: str) -> str:
    """Decode base-64 data URL → file.  Returns path."""
    loop = asyncio.get_running_loop()
    header, b64 = (data_url.split(",", 1) + [""])[:2] if "," in data_url else ("", data_url)
    ext = ".png" if "png" in header else ".jpg"
    fn  = _safe(f"{prompt[:30]}_{_stamp()}_{_sid()}{ext}")
    fp  = OUTPUT_DIR / fn
    raw = base64.b64decode(b64)
    await loop.run_in_executor(EXECUTOR, fp.write_bytes, raw)
    log.info("Saved inline → %s", fp)
    return str(fp)


async def _download(image_id: str, prompt: str) -> str:
    """Download via PerchanceClient (blocking, in executor)."""
    loop   = asyncio.get_running_loop()
    prefix = f"{_safe(prompt[:30])}_{_stamp()}_{_sid()}"
    return await loop.run_in_executor(
        EXECUTOR,
        partial(CLIENT.download_image, image_id, prefix),
    )


async def _generate_single(
    prompt: str,
    task: dict,
    idx: int,
    queue: asyncio.Queue,
    ad_code: str,
) -> Optional[str]:
    """
    Generate + save one image.
    On 'invalid_key', triggers a coordinated key refresh and retries
    up to MAX_KEY_RETRIES times.  Returns the saved filepath or None.
    """
    loop    = asyncio.get_running_loop()
    tid     = task["id"]

    for key_try in range(1, MAX_KEY_RETRIES + 1):

        # ── wait if a refresh is in progress ──
        await _key_valid.wait()

        # ── read current key ──
        async with _key_lock:
            active_key = USER_KEY

        if not active_key:
            await queue.put({
                "type":    "error",
                "time":    _now(),
                "task_id": tid,
                "message": "No userKey available. Set via /set_user_key",
            })
            return None

        # ── blocking generation in thread-pool ──
        result = await loop.run_in_executor(
            EXECUTOR,
            partial(
                CLIENT.generate_one,
                prompt=prompt,
                negative_prompt=task["negative"],
                seed=-1,
                resolution=task["resolution"],
                guidance_scale=task["guidance"],
                channel="ai-text-to-image-generator",
                sub_channel=task["sub_channel"],
                user_key=active_key,
                ad_access_code=ad_code,
                request_id=_reqid(),
            ),
        )

        # ── invalid_key → refresh + retry ──
        if result.get("error") == "invalid_key":
            log.warning(
                "invalid_key for task %s (key_try %d/%d) → refreshing",
                tid, key_try, MAX_KEY_RETRIES,
            )
            await queue.put({
                "type":         "key_invalid",
                "time":         _now(),
                "task_id":      tid,
                "attempt":      key_try,
                "max_attempts": MAX_KEY_RETRIES,
                "message":      "UserKey invalid — refreshing …",
            })

            new_key = await refresh_user_key()
            if new_key:
                # also refresh ad code with fresh key
                ad_code = await loop.run_in_executor(
                    EXECUTOR, CLIENT.get_ad_code,
                )
                continue                          # ← retry generation
            else:
                await queue.put({
                    "type":    "error",
                    "time":    _now(),
                    "task_id": tid,
                    "message": "Could not refresh userKey — aborting image",
                })
                return None

        # ── other errors ──
        if result.get("error"):
            log.warning(
                "Gen error task=%s prompt='%.40s': %s",
                tid, prompt, result,
            )
            await queue.put({
                "type":    "gen_error",
                "time":    _now(),
                "task_id": tid,
                "prompt":  prompt,
                "index":   idx,
                "error":   result,
            })
            return None

        # ── success → save ──
        try:
            if result.get("inline"):
                fp = await _save_inline(result["inline"], prompt)
            elif result.get("imageId"):
                fp = await _download(result["imageId"], prompt)
            else:
                log.error("Unexpected result: %s", result)
                return None

            seed = result.get("seed")
            task["completed"] += 1
            task["results"].append({
                "prompt": prompt,
                "index":  idx,
                "path":   fp,
                "seed":   seed,
            })
            await queue.put({
                "type":      "image_ready",
                "time":      _now(),
                "task_id":   tid,
                "prompt":    prompt,
                "index":     idx,
                "path":      fp,
                "seed":      seed,
                "completed": task["completed"],
                "total":     task["total_images"],
            })
            return fp

        except Exception as exc:
            log.exception("Save/download error task=%s: %s", tid, exc)
            await queue.put({
                "type":    "download_error",
                "time":    _now(),
                "task_id": tid,
                "prompt":  prompt,
                "index":   idx,
                "error":   str(exc),
            })
            return None

    # exhausted key retries
    log.error("Exhausted key retries for task %s prompt='%.40s'", tid, prompt)
    return None


async def worker_loop(worker_id: int, semaphore: asyncio.Semaphore):
    """Long-running coroutine: pull jobs → generate images."""
    log.info("Worker %d started", worker_id)
    loop = asyncio.get_running_loop()

    while True:
        job = await JOB_QUEUE.get()

        # shutdown sentinel
        if job is None:
            log.info("Worker %d shutting down", worker_id)
            JOB_QUEUE.task_done()
            break

        task  = job["task"]
        tid   = task["id"]
        queue = TASK_QUEUES.get(tid)

        log.info(
            "Worker %d → task %s (%d images)",
            worker_id, tid, task["total_images"],
        )
        task["status"] = "running"
        if queue:
            await queue.put({
                "type":         "started",
                "time":         _now(),
                "task_id":      tid,
                "total_images": task["total_images"],
            })

        # fetch ad code once per task
        ad_code = await loop.run_in_executor(EXECUTOR, CLIENT.get_ad_code)

        # heartbeat coroutine
        async def _heartbeat():
            while task["status"] == "running":
                await asyncio.sleep(5.0)
                if queue and task["status"] == "running":
                    try:
                        queue.put_nowait({
                            "type":      "heartbeat",
                            "time":      _now(),
                            "task_id":   tid,
                            "completed": task["completed"],
                            "total":     task["total_images"],
                        })
                    except asyncio.QueueFull:
                        pass

        hb = asyncio.create_task(_heartbeat())

        try:
            for prompt in task["prompts"]:
                for i in range(task["count"]):
                    async with semaphore:
                        await _generate_single(
                            prompt, task, i, queue, ad_code,
                        )
                    if task["status"] == "failed":
                        break
                if task["status"] == "failed":
                    break

            # decide final status
            if task["status"] != "failed":
                if task["completed"] == 0 and task["total_images"] > 0:
                    task["status"] = "failed"
                    task["error"]  = "No images generated successfully"
                else:
                    task["status"] = "done"

            if queue:
                await queue.put({
                    "type":      task["status"],   # "done" or "failed"
                    "time":      _now(),
                    "task_id":   tid,
                    "completed": task["completed"],
                    "total":     task["total_images"],
                    "error":     task.get("error"),
                })

        except Exception as exc:
            log.exception("Worker %d task %s crashed: %s", worker_id, tid, exc)
            task["status"] = "failed"
            task["error"]  = str(exc)
            if queue:
                await queue.put({
                    "type":    "failed",
                    "time":    _now(),
                    "task_id": tid,
                    "error":   str(exc),
                })

        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass

            if queue:
                await queue.put({"type": "eof", "time": _now(), "task_id": tid})

            JOB_QUEUE.task_done()
            log.info(
                "Worker %d task %s finished (%s, %d/%d)",
                worker_id, tid, task["status"],
                task["completed"], task["total_images"],
            )


# ═══════════════════════════════════════════════════════════════
#  FASTAPI – lifespan + app + endpoints
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global USER_KEY, _key_lock, _key_valid, _key_refresh_lock
    global _key_last_ts, _key_fail_count, JOB_QUEUE

    # ── create asyncio primitives in uvicorn's loop ──
    _key_lock         = asyncio.Lock()
    _key_valid        = asyncio.Event()
    _key_valid.set()                           # assume usable initially
    _key_refresh_lock = asyncio.Lock()
    JOB_QUEUE         = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

    # If configured to start virtual display, attempt to do so when needed.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, partial(_start_virtual_display_if_needed, ZD_HEADLESS))
    except Exception:
        log.exception("Failed to ensure virtual display at startup")

    # ── initial key fetch (skip if already set from __main__) ──
    if USER_KEY:
        log.info("Using pre-fetched userKey (len=%d)", len(USER_KEY))
        _key_last_ts = time.time()
    else:
        skip = os.environ.get("NO_INITIAL_FETCH", "") in ("1", "true", "True")
        if not skip:
            try:
                key = await fetch_key_via_browser(
                    timeout=ZD_TIMEOUT, headless=ZD_HEADLESS,
                )
                if key:
                    USER_KEY     = key
                    _key_last_ts = time.time()
                    log.info("Fetched userKey at startup (len=%d)", len(key))
                else:
                    log.warning(
                        "Startup key fetch failed. "
                        "Use /set_user_key or /fetch_user_key."
                    )
            except Exception as exc:
                log.exception("Startup key fetch error: %s", exc)
        else:
            log.info("NO_INITIAL_FETCH=1 → skipping browser key fetch")

    # ── launch workers ──
    sem     = asyncio.Semaphore(WORKER_COUNT)
    workers = [
        asyncio.create_task(worker_loop(i + 1, sem))
        for i in range(WORKER_COUNT)
    ]
    log.info("Launched %d workers", WORKER_COUNT)

    # ---------- server is running ----------
    yield
    # ---------- shutdown begins ------------

    log.info("Shutdown: sending stop sentinels to workers …")
    for _ in range(WORKER_COUNT):
        await JOB_QUEUE.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    try:
        SCRAPER.close()
    except Exception:
        pass
    EXECUTOR.shutdown(wait=True)

    # Stop virtual display if we started one
    try:
        await loop.run_in_executor(None, _stop_virtual_display_if_needed)
    except Exception:
        log.exception("Failed to stop virtual display cleanly")

    log.info("Shutdown complete")


# ── app ──
app = FastAPI(
    title="Perchance Image Generation Server v2 (pyvirtualdisplay)",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────── endpoints ─────────────

@app.get("/health")
async def health():
    async with _key_lock:
        has_key = USER_KEY is not None
    return {
        "status":       "ok",
        "has_user_key":  has_key,
        "queue_size":    JOB_QUEUE.qsize() if JOB_QUEUE else 0,
        "active_tasks":  sum(
            1 for t in TASKS.values() if t["status"] in ("queued", "running")
        ),
    }


@app.get("/user_key")
async def user_key_info():
    async with _key_lock:
        has = USER_KEY is not None
        ln  = len(USER_KEY) if has else 0
    return {"has_user_key": has, "key_length": ln}


@app.post("/set_user_key")
async def set_user_key(payload: Dict[str, str]):
    global USER_KEY, _key_last_ts, _key_fail_count
    key = payload.get("userKey", "").strip()
    if not key:
        raise HTTPException(400, "userKey required")
    async with _key_lock:
        USER_KEY = key
    _key_last_ts    = time.time()
    _key_fail_count = 0
    _key_valid.set()                          # unblock any waiting workers
    log.info("userKey set via API (len=%d)", len(key))
    return {"status": "ok", "key_length": len(key)}


@app.post("/fetch_user_key")
async def fetch_user_key_endpoint():
    """Trigger a background browser-based key fetch."""
    global _key_fail_count

    async def _bg():
        global _key_fail_count
        _key_fail_count = 0                   # reset so refresh is allowed
        await refresh_user_key()

    asyncio.create_task(_bg())
    return {"status": "started", "note": "Browser key fetch running in background"}


@app.post("/generate")
async def submit_job(payload: Dict[str, Any]):
    """
    POST /generate
    Body:
        {
          "prompts": ["a cat in space", "sunset over mountains"],
          "count": 2,
          "resolution": "512x768",
          "guidance": 7.0,
          "negative": "",
          "subChannel": "private"
        }
    Returns:
        { "task_id": "...", "stream_url": "/stream/...", "queue_position": N }
    """
    prompts = payload.get("prompts") or payload.get("prompt") or []
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(400, "prompts must be a non-empty list")

    count       = max(1, int(payload.get("count", 1)))
    resolution  = payload.get("resolution", "512x768")
    guidance    = float(payload.get("guidance", 7.0))
    negative    = payload.get("negative", "") or ""
    sub_channel = payload.get("subChannel", "private")

    task = create_task(prompts, count, resolution, guidance, negative, sub_channel)

    try:
        await JOB_QUEUE.put({"task": task})
    except asyncio.QueueFull:
        raise HTTPException(503, "Server queue full — try again later")

    position = JOB_QUEUE.qsize()
    q = TASK_QUEUES.get(task["id"])
    if q:
        await q.put({
            "type":           "queued",
            "time":           _now(),
            "task_id":        task["id"],
            "queue_position": position,
            "total_images":   task["total_images"],
        })

    return {
        "task_id":        task["id"],
        "stream_url":     f"/stream/{task['id']}",
        "queue_position": position,
    }


@app.get("/stream/{task_id}")
async def stream_task(request: Request, task_id: str):
    """
    SSE stream.  Event types:
      meta · queued · started · heartbeat · image_ready
      key_invalid · key_refreshing · key_refreshed · key_refresh_failed
      gen_error · download_error · done · failed · eof
    """
    if task_id not in TASKS:
        raise HTTPException(404, "unknown task id")

    task  = TASKS[task_id]
    queue = TASK_QUEUES[task_id]

    async def event_gen():
        # ── initial snapshot ──
        yield {
            "event": "meta",
            "data":  json.dumps({
                "task_id":      task_id,
                "status":       task["status"],
                "total_images": task["total_images"],
                "created_at":   task["created_at"],
            }),
        }

        # ── if already finished, replay results + EOF ──
        if task["status"] in ("done", "failed"):
            for r in task["results"]:
                yield {
                    "event": "image_ready",
                    "data":  json.dumps({
                        "task_id":   task_id,
                        "prompt":    r["prompt"],
                        "index":     r["index"],
                        "path":      r["path"],
                        "seed":      r["seed"],
                        "completed": task["completed"],
                        "total":     task["total_images"],
                    }),
                }
            yield {
                "event": task["status"],
                "data":  json.dumps({
                    "task_id":   task_id,
                    "completed": task["completed"],
                    "total":     task["total_images"],
                    "error":     task.get("error"),
                }),
            }
            yield {
                "event": "eof",
                "data":  json.dumps({"task_id": task_id}),
            }
            return

        # ── live stream ──
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # keep-alive ping
                yield {
                    "event": "ping",
                    "data":  json.dumps({"time": _now()}),
                }
                if await request.is_disconnected():
                    log.info("SSE client disconnected (task %s)", task_id)
                    break
                continue

            yield {
                "event": ev.get("type", "event"),
                "data":  json.dumps(ev),
            }
            if ev.get("type") == "eof":
                break

    return EventSourceResponse(event_gen())


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "unknown task id")
    return {"task": task}


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    fp = OUTPUT_DIR / filename
    if not fp.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(fp, media_type="application/octet-stream", filename=filename)


# ═══════════════════════════════════════════════════════════════
#  MAIN (when run directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    # Allow overriding via environment variables
    ZD_HEADLESS = os.environ.get("ZD_HEADLESS", str(ZD_HEADLESS)) in ("1", "true", "True")
    USE_VIRTUAL_DISPLAY = os.environ.get("USE_VIRTUAL_DISPLAY", str(USE_VIRTUAL_DISPLAY)) in ("1", "true", "True")

    # If running locally and not headless, start virtual display if needed
    try:
        _start_virtual_display_if_needed(ZD_HEADLESS)
    except Exception:
        log.exception("Failed to ensure virtual display in __main__")

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
