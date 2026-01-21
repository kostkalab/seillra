from __future__ import annotations
from . import download_from_gdrive as dfgd
from pathlib import Path
from typing import Optional
import requests
import torch
import hashlib
import platformdirs
import logging

logger = logging.getLogger(__name__)

class StateDictLoadable:
    """Protocol for models that support state_dict loading."""
    def load_state_dict(self, state_dict, strict: bool = True):
        ...

# --- File/network utilities ---
def download_file_atomic(url: str, dest: Path, *, session: requests.Session,
                        timeout: int = 30, chunk_size: int = 8192) -> None:
    """Download a file to a temporary .part file, then atomically rename to dest."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Starting download: {url} -> {dest}")
    success = False
    try:
        # Google Drive handling
        if "drive.google.com" in url:
            import re
            match = re.search(r'id=([\w-]+)', url)
            if not match:
                # Try /file/d/FILEID/ pattern
                match = re.search(r'/file/d/([\w-]+)', url)
            if match:
                file_id = match.group(1)
                logger.info(f"Detected Google Drive URL, using download_from_gdrive for file_id={file_id}")
                ok = dfgd.download_from_gdrive(file_id, tmp)
                if ok:
                    tmp.replace(dest)
                    logger.info(f"Download complete: {dest}")
                    success = True
                    return
                else:
                    logger.error(f"Google Drive download failed for {url}")
                    raise RuntimeError(f"Google Drive download failed for {url}")
            else:
                logger.info(f"Google Drive URL detected but no file_id found in url: {url}")
        # Standard download
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
            logger.info(f"Download complete: {dest}")
            success = True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        raise
    finally:
        if not success and tmp.exists():
            tmp.unlink(missing_ok=True)
            logger.info(f"Cleaned up partial file: {tmp}")

def url_to_cache_filename(url: str, suffix: str = "") -> str:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"url_{url_hash}{suffix}"

# --- Checksum handling ---
def read_or_fetch_checksum(*, url_wts_sha: str, cache_dir: Path,
                           session: requests.Session, expected: Optional[str],
                           verify: bool, timeout: int = 30) -> Optional[str]:
    """Return expected SHA256.

    Precedence:
    1. If verify is False -> returns None.
    2. If expected provided -> use it directly (offline / pinned case).
    3. Else read cached sha256 file if present.
    4. Else download url_wts_sha, cache and return.
    """
    if not verify:
        return None
    if expected:
        return expected

    checksum_filename = url_to_cache_filename(url_wts_sha, suffix="_sha")
    checksum_path = cache_dir / checksum_filename
    if checksum_path.exists():
        return checksum_path.read_text().split()[0].strip()

    resp = session.get(url_wts_sha, timeout=timeout)
    resp.raise_for_status()
    checksum_path.write_text(resp.text)
    return resp.text.split()[0].strip()

# --- Weight file management ---
def ensure_weight_file(url_wts: str, cache_dir: Path,
                      session: requests.Session, expected_sha256: Optional[str],
                      verify: bool, timeout: int = 30) -> Path:
    """Ensure weight file exists, downloading and verifying if needed.
    Returns path to verified weight file.
    """
    weight_filename = url_to_cache_filename(url_wts, suffix="_wts")
    weight_path = cache_dir / weight_filename
    if weight_path.exists() and verify and expected_sha256:
        actual = calculate_file_sha256(weight_path)
        if actual == expected_sha256:
            logger.info(f"Checksum verified for {weight_filename}: {actual}")
            return weight_path
        weight_path.unlink(missing_ok=True)
        logger.error(f"Checksum mismatch for {weight_filename}: expected {expected_sha256}, got {actual}. File deleted.")
        raise ValueError(f"Checksum mismatch for {weight_filename}. Consider clearing cache.")
    elif weight_path.exists() and not verify:
        logger.info(f"File exists and verification skipped: {weight_filename}")
        return weight_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_file_atomic(url_wts, weight_path, session=session, timeout=timeout)
    if verify and expected_sha256:
        actual = calculate_file_sha256(weight_path)
        if actual != expected_sha256:
            weight_path.unlink(missing_ok=True)
            logger.error(f"Checksum mismatch for {weight_filename}: expected {expected_sha256}, got {actual}. File deleted.")
            raise ValueError(f"Checksum mismatch for {weight_filename}: expected {expected_sha256} got {actual}")
        logger.info(f"Checksum verified for {weight_filename}: {actual}")
    logger.info(f"Weight file ready: {weight_path}")
    return weight_path

# --- Model loading ---
def load_state_dict_from_path(model: StateDictLoadable, weight_path: Path, strict: bool = True) -> StateDictLoadable:
    """Load state_dict from file into model, set to eval mode if available."""
    logger.info(f"Loading state dict from {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    if hasattr(model, 'eval'):
        model.eval()
    logger.info(f"Model weights loaded and set to eval mode.")
    return model

# --- Utility ---
def calculate_file_sha256(path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_app_cache_dir(app_name: str) -> Path:
    """Returns the cache directory for a given application name."""
    return Path(platformdirs.user_cache_dir(appname=app_name))


class LocalFileAdapter(requests.adapters.BaseAdapter):
    """
    A requests TransportAdapter for handling file:// URLs, with proper streaming support.
    """
    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        """
        Sends the request by opening a local file, supporting both streaming and non-streaming.
        """
        from urllib.parse import urlparse
        parsed_url = urlparse(request.url)
        if parsed_url.scheme != 'file':
            raise requests.exceptions.ConnectionError(f"LocalFileAdapter received a non-file scheme: {parsed_url.scheme}")

        path_str = parsed_url.path
        if parsed_url.netloc:
            path_str = parsed_url.netloc + path_str
        
        path = Path(path_str)

        if not path.is_file():
            raise requests.exceptions.ConnectionError(f"File not found at local path: {path}")

        response = requests.models.Response()
        response.url = request.url
        response.request = request
        response.status_code = 200
        response.reason = 'OK'

        try:
            if stream:
                # For streaming requests, open the file and attach the handle for later use
                file_handle = path.open('rb')
                response.raw = file_handle
                response.close = file_handle.close
            else:
                # For non-streaming requests, read the content immediately and close the file
                response._content = path.read_bytes()
                response.raw = None  # No raw stream needed
        except Exception as e:
            response.status_code = 500
            response.reason = str(e)
            response.raw = None
            response._content = None

        return response

    def close(self):
        pass
    

# --------------------------------------------------------------------------------------
# Session / cache helpers
# --------------------------------------------------------------------------------------

def get_session() -> requests.Session:
    """Return a shared session with file:// support mounted.

    A new session is created per call (cheap) – callers can cache if desired.
    """
    session = requests.Session()
    session.mount("file://", LocalFileAdapter())
    return session


def resolve_cache_dir(app_name: str, version: str) -> Path:
    """Return (and create) the versioned cache directory."""
    return get_app_cache_dir(app_name) / version

def load_model_state_dict(model: StateDictLoadable, *, url_wts: str, url_wts_sha: str, app_name: str,
                        version: str = "latest", strict: bool = True,
                        expected_sha256: Optional[str] = None, verify: bool = True,
                        timeout: int = 30) -> StateDictLoadable:
    """
    Fetch weights from url_wts, verify with url_wts_sha, load into model and return.
    This is the simplest inference-oriented API for StateDictLoadable models.
    """
    cache_dir = resolve_cache_dir(app_name, version)
    session = get_session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = read_or_fetch_checksum(url_wts_sha=url_wts_sha, cache_dir=cache_dir,
                                      session=session, expected=expected_sha256, verify=verify,
                                      timeout=timeout)
    weight_path = ensure_weight_file(url_wts=url_wts, cache_dir=cache_dir,
                                     session=session, expected_sha256=expected, verify=verify,
                                     timeout=timeout)
    return load_state_dict_from_path(model, weight_path, strict=strict)

# --- Main API ---
def functional_load_model_state_dict(model_cls: type[StateDictLoadable], *, url_wts: str, url_wts_sha: str, app_name: str,
                                    version: str = "latest", strict: bool = True,
                                    expected_sha256: Optional[str] = None, verify: bool = True,
                                    timeout: int = 30) -> StateDictLoadable:
    """
    Instantiate model_cls, fetch weights, load and return the model.
    This is the simplest inference-oriented API for StateDictLoadable models.
    """
    cache_dir = resolve_cache_dir(app_name, version)
    session = get_session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = read_or_fetch_checksum(url_wts_sha=url_wts_sha, cache_dir=cache_dir,
                                      session=session, expected=expected_sha256, verify=verify,
                                      timeout=timeout)
    weight_path = ensure_weight_file(url_wts=url_wts, cache_dir=cache_dir,
                                     session=session, expected_sha256=expected, verify=verify,
                                     timeout=timeout)
    model = model_cls()
    return load_state_dict_from_path(model, weight_path, strict=strict)


def load_quantized_model(
    *,
    url_wts: str,
    url_wts_sha: str,
    app_name: str,
    version: str = "latest",
    expected_sha256: Optional[str] = None,
    verify: bool = True,
    timeout: int = 30,
    device: str = "cuda:0"
) -> torch.jit.ScriptModule:
    """
    Load a quantized TensorRT model from URL.
    
    Unlike load_model_state_dict, this loads a complete TorchScript model
    (not just weights), since TensorRT engines are self-contained.
    """
    try:
        import torch_tensorrt
        torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    except ImportError:
        raise ImportError(
            "torch_tensorrt is required to load quantized models. "
            "Install with: pip install torch-tensorrt"
        )
    cache_dir = resolve_cache_dir(app_name, version)
    session = get_session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    expected = read_or_fetch_checksum(
        url_wts_sha=url_wts_sha,
        cache_dir=cache_dir,
        session=session,
        expected=expected_sha256,
        verify=verify,
        timeout=timeout
    )
    
    weight_path = ensure_weight_file(
        url_wts=url_wts,
        cache_dir=cache_dir,
        session=session,
        expected_sha256=expected,
        verify=verify,
        timeout=timeout
    )
    
    logger.info(f"Loading quantized TensorRT model from {weight_path}")
    # model = torch.jit.load(str(weight_path), map_location=device)
    model = torch.jit.load(str(weight_path))
    model.eval()
    logger.info(f"Quantized model loaded")
    
    return model