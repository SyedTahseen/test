import asyncio
import datetime
import io
import mimetypes
import os
import random
import re
import time
from collections import defaultdict

import pytz
import requests
from PIL import Image
from pyrogram import Client, filters, enums
from pyrogram.errors import FloodWait
from pyrogram.types import Message, InputMediaPhoto

from utils.scripts import import_library
from utils.config import gemini_key
from utils.db import db
from utils import modules_help, prefix

try:
    from modules.custom_modules.elevenlabs import generate_elevenlabs_audio
except Exception:  # elevenlabs module missing -> voice replies fall back to text
    generate_elevenlabs_audio = None

# google-generativeai is deprecated; the official SDK is now `google-genai`.
genai = import_library("google.genai", "google-genai")
from google.genai import types as gtypes  # noqa: E402
from google.genai import errors as gerrors  # noqa: E402


# ----------------------------------------------------------------------------
# Settings / constants
# ----------------------------------------------------------------------------

# gemini-2.0-* was shut down by Google on 2026-06-01. Use a current model.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
RETIRED_MODEL_PREFIXES = ("gemini-1.", "gemini-2.0")

DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_HISTORY_HEAD = 50
DEFAULT_HISTORY_TAIL = 50

GEMINI_CONCURRENCY = 4
GENERATE_TIMEOUT = 120          # seconds per generate_content call
FILE_PROCESS_TIMEOUT = 240      # seconds to wait for Files API processing
MAX_KEY_WAIT = 45               # max seconds to wait for a rate-limited key
INLINE_LIMIT_BYTES = 10 * 1024 * 1024    # <= this is sent inline (no Files API)
MAX_MEDIA_BYTES = 200 * 1024 * 1024      # refuse to download bigger files
TEXT_DOC_LIMIT = 20000                   # chars of a text document sent to the model

TEXT_DEBOUNCE = 8
IMAGE_WINDOW = 10

history_collection = "custom.gchat"
settings_collection = "custom.gsettings"
enabled_users = db.get(settings_collection, "enabled_users") or []
disabled_users = db.get(settings_collection, "disabled_users") or []
gchat_for_all = db.get(settings_collection, "gchat_for_all") or False

smileys = ["-.-", "):", ":)", "*.*", ")*"]
la_timezone = pytz.timezone("America/Los_Angeles")
ROLES_URL = "https://gist.githubusercontent.com/iTahseen/00890d65192ca3bd9b2a62eb034b96ab/raw/roles.json"
# Placeholder! Set the real one with:  .setgchat picgroup <chat_id>
BOT_PIC_GROUP_ID = -1001234567890

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


def build_safety_settings():
    return [
        gtypes.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in SAFETY_CATEGORIES
    ]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _sync_write_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(TEXT_DOC_LIMIT + 1)


def _fetch_roles_sync():
    r = requests.get(ROLES_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def mask_key(key):
    if not key:
        return "None"
    if len(key) <= 12:
        return key[:3] + "…"
    return f"{key[:6]}…{key[-4:]}"


def is_user_allowed(user_id):
    if user_id in disabled_users:
        return False
    return gchat_for_all or user_id in enabled_users


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_history_limits():
    head = _to_int(db.get(settings_collection, "history_head"), DEFAULT_HISTORY_HEAD)
    tail = _to_int(db.get(settings_collection, "history_tail"), DEFAULT_HISTORY_TAIL)
    return head, tail


def get_max_output_tokens():
    n = _to_int(db.get(settings_collection, "max_output_tokens"), DEFAULT_MAX_OUTPUT_TOKENS)
    return n if n > 0 else DEFAULT_MAX_OUTPUT_TOKENS


def get_chat_history(user_id, user_message, user_name):
    max_head, max_tail = get_history_limits()
    chat_history = db.get(history_collection, f"chat_history.{user_id}") or []
    chat_history.append(f"{user_name}: {user_message}")
    db.set(history_collection, f"chat_history.{user_id}", chat_history)
    if len(chat_history) > max_head + max_tail:
        return chat_history[:max_head] + ["..."] + chat_history[-max_tail:]
    return chat_history


# ----------------------------------------------------------------------------
# Model / keys / clients
# ----------------------------------------------------------------------------

_gemini_model_cache = None


def get_gemini_model():
    global _gemini_model_cache
    if _gemini_model_cache is not None:
        return _gemini_model_cache
    model = db.get(settings_collection, "gemini_model") or DEFAULT_GEMINI_MODEL
    if model.lower().startswith(RETIRED_MODEL_PREFIXES):
        # a retired model saved in the DB would make every request fail
        model = DEFAULT_GEMINI_MODEL
        db.set(settings_collection, "gemini_model", model)
    _gemini_model_cache = model
    return model


def set_gemini_model(model_name: str):
    global _gemini_model_cache
    model_name = model_name.strip()
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    db.set(settings_collection, "gemini_model", model_name)
    _gemini_model_cache = model_name


def get_api_keys():
    keys = [k for k in (db.get(settings_collection, "gemini_keys") or []) if k]
    if not keys and gemini_key:
        keys = [gemini_key]
    return keys


_clients = {}


def get_client(api_key):
    """One client per key. Never use global state, so concurrent requests with
    different keys cannot get mixed up (and files stay tied to their key)."""
    client = _clients.get(api_key)
    if client is None:
        client = genai.Client(
            api_key=api_key,
            http_options=gtypes.HttpOptions(timeout=300_000),  # ms
        )
        _clients[api_key] = client
    return client


class GeminiEmptyResponse(Exception):
    pass


class GeminiKeysUnavailable(Exception):
    pass


_key_cooldown = {}  # api_key -> unix time when it may be used again


def _cooldown_key(key, seconds):
    _key_cooldown[key] = time.time() + seconds


def _is_zero_quota(text):
    return bool(
        re.search(r'quota_limit_value"\s*value:\s*"0"', text)
        or re.search(r"limit:\s*0\b", text)
    )


def _retry_seconds(exc):
    text = str(exc)
    if _is_zero_quota(text) or "PerDay" in text:
        return 3600  # no point in retrying soon
    m = re.search(r"retry in ([\d.]+)s", text, re.I) or re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", text
    )
    if m:
        return min(max(float(m.group(1)) + 1, 5), 3600)
    return 60


async def pick_api_key(keys):
    """Return the first usable key, starting at the saved index."""
    n = len(keys)
    start = _to_int(db.get(settings_collection, "current_key_index"), 0) % n
    while True:
        now = time.time()
        for off in range(n):
            i = (start + off) % n
            if _key_cooldown.get(keys[i], 0) <= now:
                if i != start:
                    db.set(settings_collection, "current_key_index", i)
                return keys[i]
        soonest = min(_key_cooldown.get(k, 0) for k in keys) - now
        if soonest > MAX_KEY_WAIT:
            raise GeminiKeysUnavailable(
                f"All {n} Gemini key(s) are rate-limited or rejected "
                f"(next retry in ~{int(soonest)}s)."
            )
        await asyncio.sleep(max(soonest, 0) + 0.5)


def describe_exception(e):
    text = f"{type(e).__name__}: {e}".strip()
    if len(text) > 700:
        text = text[:700] + "…"
    hint = ""
    if isinstance(e, gerrors.APIError):
        code = getattr(e, "code", None)
        raw = str(e)
        if code == 429 and _is_zero_quota(raw):
            hint = (
                "\n💡 This key has 0 quota for this model/region (free tier not "
                "available). Use another key/project, enable billing, or try a "
                "different model: .setgchat model <name>"
            )
        elif code == 404:
            hint = (
                f"\n💡 Model '{get_gemini_model()}' not found / retired. "
                f"Try: .setgchat model {DEFAULT_GEMINI_MODEL}"
            )
        elif code in (401, 403) or (code == 400 and "api key" in raw.lower()):
            hint = "\n💡 The API key looks invalid or lacks permission."
    return text + hint


_last_error_report = {}


async def report_error(client, title, message=None, exc=None, extra=""):
    """Send an error to Saved Messages (identical errors throttled to 1 / 2 min)."""
    try:
        who = ""
        if message is not None:
            chat = getattr(message, "chat", None)
            who = f"\n@{getattr(chat, 'username', None) or getattr(chat, 'id', '?')}"
        body = describe_exception(exc) if exc is not None else extra
        throttle_key = (title, body[:200])
        now = time.time()
        if now - _last_error_report.get(throttle_key, 0) < 120:
            return
        _last_error_report[throttle_key] = now
        text = f"⚠️ {title}{who}"
        if body:
            text += f"\n{body}"
        await send_reply(client.send_message, ["me", text[:3900]], {}, client)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Reply queue
# ----------------------------------------------------------------------------

_reply_queue = None
_reply_worker_task = None
_gemini_semaphore = None
_user_locks = {}


def _get_reply_queue():
    global _reply_queue
    if _reply_queue is None:
        _reply_queue = asyncio.Queue()
    return _reply_queue


def _get_semaphore():
    global _gemini_semaphore
    if _gemini_semaphore is None:
        _gemini_semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)
    return _gemini_semaphore


def _get_user_lock(user_id):
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


async def _call_with_floodwait(client, reply_func, args, kwargs):
    try:
        await reply_func(*args, **kwargs)
    except FloodWait as e:
        try:
            await client.send_message("me", f"⚠️ FloodWait\nSleeping {e.value}s")
        except Exception:
            pass
        await asyncio.sleep(e.value + 1)
        await reply_func(*args, **kwargs)


async def reply_worker(client):
    queue = _get_reply_queue()
    while True:
        reply_func, args, kwargs = await queue.get()
        kwargs = dict(kwargs)
        cleanup_file = kwargs.pop("cleanup_file", None)
        try:
            await _call_with_floodwait(client, reply_func, args, kwargs)
        except Exception as e:
            try:
                await client.send_message("me", f"⚠️ Reply queue error\n{e}")
            except Exception:
                pass
        finally:
            _safe_remove(cleanup_file)
        await asyncio.sleep(1)


def ensure_reply_worker(client):
    global _reply_worker_task
    if _reply_worker_task is None or _reply_worker_task.done():
        _reply_worker_task = asyncio.create_task(reply_worker(client))


async def send_reply(reply_func, args, kwargs, client):
    ensure_reply_worker(client)
    await _get_reply_queue().put((reply_func, list(args), dict(kwargs)))


async def instant_reply(reply_func, args, kwargs, client):
    kwargs = dict(kwargs)
    cleanup_file = kwargs.pop("cleanup_file", None)
    try:
        await _call_with_floodwait(client, reply_func, list(args), kwargs)
    except Exception as e:
        try:
            await client.send_message("me", f"⚠️ Reply error\n{e}")
        except Exception:
            pass
    finally:
        _safe_remove(cleanup_file)


# ----------------------------------------------------------------------------
# Voice / settings
# ----------------------------------------------------------------------------

def get_voice_generation_enabled():
    enabled = db.get(settings_collection, "voice_generation_enabled")
    if enabled is None:
        enabled = True
        db.set(settings_collection, "voice_generation_enabled", True)
    return enabled


def set_voice_generation_enabled(enabled: bool):
    db.set(settings_collection, "voice_generation_enabled", enabled)


# ----------------------------------------------------------------------------
# .gpic support
# ----------------------------------------------------------------------------

_pics_cache = {"time": 0, "photos": [], "group": None}
PICS_CACHE_TTL = 600


async def fetch_bot_pics(client, max_photos=200):
    group_id = db.get(settings_collection, "bot_pic_group_id") or BOT_PIC_GROUP_ID
    now = time.time()
    if (
        _pics_cache["photos"]
        and _pics_cache["group"] == group_id
        and now - _pics_cache["time"] < PICS_CACHE_TTL
    ):
        return _pics_cache["photos"]
    photos = []
    async for msg in client.get_chat_history(group_id, limit=max_photos):
        if msg.photo:
            photos.append(msg.photo.file_id)
    _pics_cache.update(time=now, photos=photos, group=group_id)
    return photos


async def handle_gpic_message(client, chat_id, bot_response):
    if not bot_response or not isinstance(bot_response, str):
        return False
    if not re.match(r"^\.gpic(\s|$)", bot_response):
        return False
    parts = bot_response.split(maxsplit=2)
    n = 1
    caption = ""
    if len(parts) >= 2 and parts[1].isdigit():
        n = int(parts[1])
    if len(parts) == 3:
        caption = parts[2]
    try:
        photos = await fetch_bot_pics(client)
    except Exception as e:
        await report_error(
            client, "gpic error",
            extra=f"{describe_exception(e)}\nCheck the pictures group: .setgchat picgroup <chat_id>",
        )
        return True
    if not photos:
        await report_error(client, "gpic", extra="No bot pictures found in group/channel.")
        return True
    selected = random.sample(photos, min(max(n, 1), len(photos), 10))
    if len(selected) > 1:
        media = [
            InputMediaPhoto(pic, caption=caption if i == 0 else "")
            for i, pic in enumerate(selected)
        ]
        await send_reply(client.send_media_group, [chat_id, media], {}, client)
    else:
        await send_reply(client.send_photo, [chat_id, selected[0]], {"caption": caption}, client)
    return True


async def handle_voice_message(client, chat_id, bot_response):
    if not isinstance(bot_response, str) or not re.match(r"^\.el(\s|$)", bot_response):
        return False
    text = bot_response[3:].strip()
    if not text:
        return True
    if not get_voice_generation_enabled() or generate_elevenlabs_audio is None:
        await send_reply(client.send_message, [chat_id, text], {}, client)
        return True
    try:
        audio_path = await generate_elevenlabs_audio(text=text)
        if audio_path and os.path.exists(audio_path):
            await send_reply(
                client.send_voice, [chat_id],
                {"voice": audio_path, "cleanup_file": audio_path}, client,
            )
        else:
            await send_reply(client.send_message, [chat_id, text], {}, client)
    except Exception:
        await send_reply(client.send_message, [chat_id, text], {}, client)
    return True


# ----------------------------------------------------------------------------
# Roles / prompt
# ----------------------------------------------------------------------------

_roles_cache = None
_roles_cache_time = 0
ROLES_CACHE_TTL = 300


async def fetch_roles():
    global _roles_cache, _roles_cache_time
    now = time.time()
    if _roles_cache is not None and (now - _roles_cache_time) < ROLES_CACHE_TTL:
        return _roles_cache
    try:
        roles = await asyncio.to_thread(_fetch_roles_sync)
        if isinstance(roles, dict):
            default_role_name = db.get(settings_collection, "default_role") or "default"
            if default_role_name in roles:
                roles["default"] = roles[default_role_name]
            _roles_cache = roles
            _roles_cache_time = now
            return roles
    except Exception:
        pass
    return _roles_cache or {}


async def get_bot_role(user_id):
    """Per-user custom role first (works even if the roles gist is down), then default."""
    custom = db.get(settings_collection, f"custom_roles.{user_id}")
    if custom:
        return custom
    roles = await fetch_roles()
    return roles.get("default")


def build_system_instruction(bot_role):
    if isinstance(bot_role, list):
        return "\n".join(str(x) for x in bot_role)
    return str(bot_role)


def build_prompt(chat_history, instruction=None):
    timestamp = datetime.datetime.now(la_timezone).strftime("%Y-%m-%d %H:%M:%S")
    chat_context = "\n".join(chat_history)
    prompt = f"Current Time: {timestamp}\nChat History:\n{chat_context}"
    if instruction:
        prompt += f"\n\n[{instruction}]"
    return prompt


# ----------------------------------------------------------------------------
# Gemini calls
# ----------------------------------------------------------------------------

def build_thinking_config(model_name):
    """Thinking tokens count against max_output_tokens. For short chat replies we
    keep thinking minimal, otherwise the answer can come back empty."""
    m = model_name.lower()
    try:
        if "gemini-3" in m:
            return gtypes.ThinkingConfig(thinking_level="low")
        if "gemini-2.5" in m and "pro" not in m:
            return gtypes.ThinkingConfig(thinking_budget=0)
    except Exception:
        # older google-genai without these fields -> `pip install -U google-genai`
        return None
    return None


def build_generation_config(bot_role=None, max_tokens=None):
    kwargs = {
        "max_output_tokens": max_tokens or get_max_output_tokens(),
        "safety_settings": build_safety_settings(),
    }
    if bot_role:
        kwargs["system_instruction"] = build_system_instruction(bot_role)
    thinking = build_thinking_config(get_gemini_model())
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    return gtypes.GenerateContentConfig(**kwargs)


def extract_text(response):
    try:
        text = response.text
    except Exception:
        text = None
    return text.strip() if isinstance(text, str) else ""


def describe_empty(response):
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None) if feedback else None
    if block:
        return f"prompt blocked: {getattr(block, 'name', block)}"
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        return f"finish_reason={getattr(fr, 'name', fr)}"
    return "no candidates returned"


def _file_state(f):
    state = getattr(f, "state", None)
    if state is None:
        return "ACTIVE"
    return getattr(state, "name", str(state)).upper()


async def upload_file_to_gemini(client, file_path, mime_type):
    """Files API upload (only used for files too big to send inline)."""
    uploaded = await client.aio.files.upload(
        file=file_path, config=gtypes.UploadFileConfig(mime_type=mime_type)
    )
    try:
        deadline = time.time() + FILE_PROCESS_TIMEOUT
        while _file_state(uploaded) == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError("Gemini file processing timed out.")
            await asyncio.sleep(3)
            uploaded = await client.aio.files.get(name=uploaded.name)
        if _file_state(uploaded) == "FAILED":
            raise ValueError("File failed to process on Gemini side.")
    except BaseException:
        await _delete_uploaded(client, [uploaded])
        raise
    return uploaded


async def _delete_uploaded(client, files):
    for f in files:
        try:
            await client.aio.files.delete(name=f.name)
        except Exception:
            pass


async def build_contents(client, prompt, media, uploaded):
    """media items: {"data": bytes, "mime": str} | {"path": str, "mime": str} | {"text": str}.
    Files are uploaded with the SAME client/key used for generation (files are
    private to the key's project -> otherwise: 403 'File ... may not exist')."""
    contents = [prompt]
    for item in media or []:
        if "text" in item:
            contents.append(item["text"])
        elif "data" in item:
            contents.append(gtypes.Part.from_bytes(data=item["data"], mime_type=item["mime"]))
        else:
            path, mime = item["path"], item["mime"]
            if os.path.getsize(path) <= INLINE_LIMIT_BYTES:
                data = await asyncio.to_thread(_read_bytes, path)
                contents.append(gtypes.Part.from_bytes(data=data, mime_type=mime))
            else:
                f = await upload_file_to_gemini(client, path, mime)
                uploaded.append(f)
                contents.append(gtypes.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type or mime))
    return contents


def _is_transient_network_error(e):
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return type(e).__module__.split(".")[0] in ("httpx", "httpcore", "aiohttp")


async def generate_gemini_response(prompt, user_id, bot_role=None, media=None):
    keys = get_api_keys()
    if not keys:
        raise GeminiKeysUnavailable("No Gemini API key configured. Add one: .setgchat add <key>")
    model_name = get_gemini_model()
    config = build_generation_config(bot_role)
    transient_failures = 0
    empty_retries = 0
    last_exc = None

    for _ in range(len(keys) + 5):
        try:
            key = await pick_api_key(keys)
        except GeminiKeysUnavailable as e:
            if last_exc is not None:
                raise GeminiKeysUnavailable(f"{e}\nLast error: {describe_exception(last_exc)}")
            raise
        client = get_client(key)
        uploaded = []
        try:
            contents = await build_contents(client, prompt, media, uploaded)
            async with _get_semaphore():
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name, contents=contents, config=config
                    ),
                    timeout=GENERATE_TIMEOUT,
                )
            text = extract_text(response)
            if not text:
                reason = describe_empty(response)
                last_exc = GeminiEmptyResponse(reason)
                if empty_retries < 1 and not reason.startswith("prompt blocked"):
                    empty_retries += 1
                    await asyncio.sleep(1.5)
                    continue
                raise last_exc
            full_history = db.get(history_collection, f"chat_history.{user_id}") or []
            full_history.append(text)
            db.set(history_collection, f"chat_history.{user_id}", full_history)
            return text
        except GeminiEmptyResponse:
            raise
        except gerrors.APIError as e:
            last_exc = e
            code = getattr(e, "code", None)
            if code == 429:
                _cooldown_key(key, _retry_seconds(e))
                continue
            if code in (401, 403) or (code == 400 and "api key" in str(e).lower()):
                _cooldown_key(key, 600)
                continue
            if isinstance(code, int) and code >= 500:
                transient_failures += 1
                if transient_failures > 3:
                    raise
                await asyncio.sleep(min(2 ** transient_failures, 10) + random.random())
                continue
            raise
        except Exception as e:
            if _is_transient_network_error(e):
                last_exc = e
                transient_failures += 1
                if transient_failures > 3:
                    raise
                await asyncio.sleep(min(2 ** transient_failures, 10) + random.random())
                continue
            raise
        finally:
            if uploaded:
                await _delete_uploaded(client, uploaded)

    raise last_exc or RuntimeError("Gemini request failed")


async def send_typing_action(client, chat_id, user_message):
    try:
        await client.send_chat_action(chat_id=chat_id, action=enums.ChatAction.TYPING)
        await asyncio.sleep(min(len(user_message) / 10, 5))
    except Exception:
        pass


async def deliver_response(client, message, response, tag="resp", quote=False):
    """Send Gemini's answer (handles .gpic / .el commands and long texts)."""
    if not response:
        return False
    chat_id = message.chat.id
    if await handle_gpic_message(client, chat_id, response):
        return True
    if await handle_voice_message(client, chat_id, response):
        return True
    reply_kwargs = {"reply_to_message_id": message.id} if quote else {}
    if len(response) > 4000:
        fp = f"gchat_{tag}_{chat_id}_{int(time.time())}.txt"
        await asyncio.to_thread(_sync_write_file, fp, response)
        await send_reply(
            client.send_document, [chat_id, fp],
            {"caption": "Response", "cleanup_file": fp, **reply_kwargs}, client,
        )
    else:
        await send_reply(message.reply_text, [response], reply_kwargs, client)
    return True


# ----------------------------------------------------------------------------
# Stickers / GIFs
# ----------------------------------------------------------------------------

sticker_gif_buffer = defaultdict(list)
sticker_gif_timer = {}


async def process_sticker_gif_buffer(client, user_id):
    try:
        await asyncio.sleep(8)
        msgs = sticker_gif_buffer.pop(user_id, [])
        sticker_gif_timer.pop(user_id, None)
        if not msgs:
            return
        last_msg = msgs[-1]
        await asyncio.sleep(random.uniform(5, 10))
        await send_reply(last_msg.reply_text, [random.choice(smileys)], {}, client)
    except Exception as e:
        await report_error(client, "Sticker/GIF buffer error", exc=e)


@Client.on_message(
    (filters.sticker | filters.animation) & filters.private & ~filters.me & ~filters.bot,
    group=1,
)
async def handle_sticker_gif_buffered(client: Client, message: Message):
    if not message.from_user:
        return
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "User"
    if not is_user_allowed(user_id):
        return
    try:
        full_history = db.get(history_collection, f"chat_history.{user_id}") or []
        if not full_history:
            bot_role = await get_bot_role(user_id)
            if not bot_role:
                await report_error(client, "Sticker handler", message, extra="'default' role missing.")
                return
            async with _get_user_lock(user_id):
                chat_history = get_chat_history(user_id, "hello", user_name)
                await send_typing_action(client, message.chat.id, "hello")
                response = await generate_gemini_response(
                    build_prompt(chat_history), user_id, bot_role=bot_role
                )
                if not response:
                    await report_error(client, "Gemini empty response", message)
                else:
                    await deliver_response(client, message, response, tag="resp")
            return
    except Exception as e:
        await report_error(client, "Sticker handler error", message, exc=e)
    sticker_gif_buffer[user_id].append(message)
    if sticker_gif_timer.get(user_id):
        sticker_gif_timer[user_id].cancel()
    sticker_gif_timer[user_id] = asyncio.create_task(process_sticker_gif_buffer(client, user_id))


# ----------------------------------------------------------------------------
# Text messages
# ----------------------------------------------------------------------------

_text_buffers = defaultdict(list)
_text_tasks = {}
_text_inflight = set()


async def process_text_buffer(client, message, user_id, user_name):
    await asyncio.sleep(TEXT_DEBOUNCE)  # cancelled if the user keeps typing
    async with _get_user_lock(user_id):
        buffered = _text_buffers.pop(user_id, [])
        if not buffered:
            return
        # From here on this task is no longer cancelled by new messages
        # (they start a new batch) -> messages are never lost mid-request.
        _text_inflight.add(user_id)
        try:
            combined = " ".join(buffered)
            bot_role = await get_bot_role(user_id)
            if not bot_role:
                await report_error(client, "gchat", message, extra="'default' role missing.")
                return
            chat_history = get_chat_history(user_id, combined, user_name)
            await asyncio.sleep(random.choice([3, 5, 7]))
            await send_typing_action(client, message.chat.id, combined)
            try:
                response = await generate_gemini_response(
                    build_prompt(chat_history), user_id, bot_role=bot_role
                )
                await deliver_response(client, message, response, tag="resp")
            except Exception as e:
                await report_error(client, "gchat error", message, exc=e)
        finally:
            _text_inflight.discard(user_id)


@Client.on_message(filters.text & filters.private & ~filters.me & ~filters.bot, group=1)
async def gchat(client: Client, message: Message):
    try:
        if not message.from_user:
            return
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        user_message = message.text.strip()
        if not user_message or not is_user_allowed(user_id):
            return
        _text_buffers[user_id].append(user_message)
        previous = _text_tasks.get(user_id)
        if previous and not previous.done() and user_id not in _text_inflight:
            previous.cancel()
        _text_tasks[user_id] = asyncio.create_task(
            process_text_buffer(client, message, user_id, user_name)
        )
    except Exception as e:
        await report_error(client, "gchat module error", message, exc=e)


# ----------------------------------------------------------------------------
# Photos / files
# ----------------------------------------------------------------------------

VIDEO_MIMES = {
    "video/mp4": "video/mp4", "video/mpeg": "video/mpeg", "video/mpg": "video/mpg",
    "video/quicktime": "video/mov", "video/mov": "video/mov",
    "video/x-msvideo": "video/avi", "video/avi": "video/avi",
    "video/x-flv": "video/x-flv", "video/webm": "video/webm",
    "video/x-ms-wmv": "video/wmv", "video/wmv": "video/wmv", "video/3gpp": "video/3gpp",
}
AUDIO_MIMES = {
    "audio/wav": "audio/wav", "audio/x-wav": "audio/wav", "audio/wave": "audio/wav",
    "audio/mp3": "audio/mp3", "audio/mpeg": "audio/mp3",
    "audio/aiff": "audio/aiff", "audio/x-aiff": "audio/aiff",
    "audio/aac": "audio/aac", "audio/x-aac": "audio/aac", "audio/mp4": "audio/aac",
    "audio/x-m4a": "audio/aac", "audio/m4a": "audio/aac",
    "audio/ogg": "audio/ogg", "audio/opus": "audio/ogg",
    "audio/flac": "audio/flac", "audio/x-flac": "audio/flac",
}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".css", ".js", ".ts",
    ".py", ".log", ".yaml", ".yml", ".ini", ".cfg", ".rtf", ".tsv", ".sql",
}

_image_buffers = {}


def _prepare_image(path):
    """Decode, downscale and re-encode as JPEG bytes (robust for any input format)."""
    with Image.open(path) as img:
        img.load()
        img.thumbnail((2048, 2048))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()


async def process_images(client, user_id, user_name):
    await asyncio.sleep(IMAGE_WINDOW)
    buf = _image_buffers.pop(user_id, None)
    if not buf:
        return
    paths, captions, message = buf["paths"], buf["captions"], buf["message"]
    try:
        media = []
        for p in paths:
            try:
                data = await asyncio.to_thread(_prepare_image, p)
                media.append({"data": data, "mime": "image/jpeg"})
            except Exception:
                continue
        if not media:
            await report_error(client, "process_images", message, extra="No valid images to process.")
            return
        bot_role = await get_bot_role(user_id)
        if not bot_role:
            await report_error(client, "process_images", message, extra="'default' role missing.")
            return
        caption = " | ".join(captions)
        async with _get_user_lock(user_id):
            chat_history = get_chat_history(user_id, caption or "[image]", user_name)
            instruction = f"User sent {len(media)} image(s)." + (
                f" Caption: {caption}" if caption else " React to the image(s) naturally, in character."
            )
            response = await generate_gemini_response(
                build_prompt(chat_history, instruction), user_id, bot_role=bot_role, media=media
            )
            if not response:
                await report_error(client, "Gemini empty response", message)
                return
            await deliver_response(client, message, response, tag="img_resp", quote=True)
    except Exception as e:
        await report_error(client, "process_images error", message, exc=e)
    finally:
        for p in paths:
            _safe_remove(p)


def _normalize_mime(mime):
    return (mime or "").lower().split(";")[0].strip()


def resolve_media(message):
    """-> (kind, label, telegram_object) or None"""
    if message.video:
        return "video", "video", message.video
    if message.video_note:
        return "video", "video note", message.video_note
    if message.voice:
        return "voice", "voice message", message.voice
    if message.audio:
        return "audio", "audio", message.audio
    if message.document:
        return "document", "file", message.document
    return None


async def handle_media_message(client, message, user_id, user_name):
    resolved = resolve_media(message)
    if not resolved:
        return
    kind, label, obj = resolved
    file_name = getattr(obj, "file_name", None) or ""
    mime = _normalize_mime(getattr(obj, "mime_type", None))
    if not mime and file_name:
        mime = _normalize_mime(mimetypes.guess_type(file_name)[0])
    ext = os.path.splitext(file_name)[1].lower()
    caption = message.caption.strip() if message.caption else ""
    size = getattr(obj, "file_size", 0) or 0

    # Decide how to feed the file to Gemini
    gemini_mime = None
    as_text = False
    note = ""
    if kind == "video":
        gemini_mime = VIDEO_MIMES.get(mime or "video/mp4")
    elif kind == "voice":
        gemini_mime = AUDIO_MIMES.get(mime or "audio/ogg")
    elif kind == "audio":
        gemini_mime = AUDIO_MIMES.get(mime or "audio/mpeg")
    else:  # document
        if mime == "application/pdf" or ext == ".pdf":
            gemini_mime, label = "application/pdf", "PDF"
        elif ext in TEXT_EXTENSIONS or mime.startswith("text/") or mime in (
            "application/json", "application/xml"
        ):
            as_text = True
    if size and size > MAX_MEDIA_BYTES:
        gemini_mime, as_text = None, False
        note = "The file is too large to open."
    elif not gemini_mime and not as_text and not note:
        note = "The file type can't be opened, only its name is known."
    if as_text and size > 2 * 1024 * 1024:
        as_text = False
        note = "The text file is too large to read."

    file_path = None
    try:
        media = []
        if gemini_mime or as_text:
            file_path = await client.download_media(obj)
            if not file_path:
                note, gemini_mime, as_text = "The file couldn't be downloaded.", None, False
            elif as_text:
                content = await asyncio.to_thread(_read_text, file_path)
                if len(content) > TEXT_DOC_LIMIT:
                    content = content[:TEXT_DOC_LIMIT] + "\n[...truncated]"
                media.append({"text": f"[Contents of {file_name or 'file'}]\n{content}"})
            else:
                media.append({"path": file_path, "mime": gemini_mime})

        bot_role = await get_bot_role(user_id)
        if not bot_role:
            await report_error(client, "handle_files", message, extra="'default' role missing.")
            return
        instruction = f"User sent a {label}" + (f" named '{file_name}'" if file_name else "") + "."
        if note:
            instruction += f" {note}"
        instruction += f" Caption: {caption}" if caption else " React to it naturally, in character."

        async with _get_user_lock(user_id):
            chat_history = get_chat_history(user_id, caption or f"[{label}]", user_name)
            response = await generate_gemini_response(
                build_prompt(chat_history, instruction), user_id, bot_role=bot_role, media=media
            )
            if not response:
                await report_error(client, "Gemini empty response", message)
                return
            await deliver_response(client, message, response, tag="file_resp", quote=True)
    finally:
        _safe_remove(file_path)


@Client.on_message(filters.private & ~filters.me & ~filters.bot, group=1)
async def handle_files(client: Client, message: Message):
    if not message.from_user:
        return
    if not any([message.photo, message.video, message.video_note,
                message.audio, message.voice, message.document]):
        return
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "User"
    if not is_user_allowed(user_id):
        return
    try:
        if message.photo:
            path = await client.download_media(message.photo)
            if not path:
                return
            buf = _image_buffers.get(user_id)
            if buf is None:
                buf = {"paths": [], "captions": [], "message": message}
                _image_buffers[user_id] = buf
                asyncio.create_task(process_images(client, user_id, user_name))
            buf["paths"].append(path)
            buf["message"] = message
            if message.caption and message.caption.strip():
                buf["captions"].append(message.caption.strip())
            return
        await handle_media_message(client, message, user_id, user_name)
    except Exception as e:
        await report_error(client, "handle_files error", message, exc=e)


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

@Client.on_message(filters.command(["setgchat", "setgc"], prefix) & filters.me)
async def set_gemini_key(client: Client, message: Message):
    try:
        command = message.text.strip().split()
        subcommand = command[1].lower() if len(command) > 1 else None
        key = command[2] if len(command) > 2 else None
        gemini_keys = db.get(settings_collection, "gemini_keys") or []
        current_key_index = _to_int(db.get(settings_collection, "current_key_index"), 0)

        if subcommand == "model":
            if key:
                set_gemini_model(key)
                warn = ""
                if key.lower().startswith(RETIRED_MODEL_PREFIXES):
                    warn = "\n⚠️ This model family was shut down by Google."
                await instant_reply(message.edit_text, [f"Gemini model set to: {get_gemini_model()}{warn}"], {}, client)
            else:
                await instant_reply(message.edit_text, [f"Current Gemini model: {get_gemini_model()}"], {}, client)
            return
        if subcommand == "voice":
            enabled = not get_voice_generation_enabled()
            set_voice_generation_enabled(enabled)
            await instant_reply(message.edit_text, [f"Voice: {'ON' if enabled else 'OFF'}"], {}, client)
            return
        if subcommand == "tokens":
            if key and key.isdigit() and int(key) > 0:
                db.set(settings_collection, "max_output_tokens", int(key))
                await instant_reply(message.edit_text, [f"Max output tokens: {key}"], {}, client)
            else:
                await instant_reply(message.edit_text, [f"Max output tokens: {get_max_output_tokens()}"], {}, client)
            return
        if subcommand == "picgroup":
            if key and re.fullmatch(r"-?\d+", key):
                db.set(settings_collection, "bot_pic_group_id", int(key))
                _pics_cache["photos"] = []
                await instant_reply(message.edit_text, [f"Pictures group set to: {key}"], {}, client)
            else:
                current = db.get(settings_collection, "bot_pic_group_id") or BOT_PIC_GROUP_ID
                await instant_reply(message.edit_text, [f"Pictures group: {current}"], {}, client)
            return
        if subcommand == "add" and key:
            if key in gemini_keys:
                await instant_reply(message.edit_text, ["Key already added!"], {}, client)
                return
            gemini_keys.append(key)
            db.set(settings_collection, "gemini_keys", gemini_keys)
            await instant_reply(message.edit_text, ["Gemini key added!"], {}, client)
            return
        if subcommand == "set" and key:
            index = _to_int(key, 0) - 1
            if 0 <= index < len(gemini_keys):
                db.set(settings_collection, "current_key_index", index)
                _key_cooldown.pop(gemini_keys[index], None)
                await instant_reply(message.edit_text, [f"Current key set to: {key}"], {}, client)
            else:
                await instant_reply(message.edit_text, [f"Invalid key index: {key}"], {}, client)
            return
        if subcommand == "del" and key:
            index = _to_int(key, 0) - 1
            if 0 <= index < len(gemini_keys):
                removed = gemini_keys.pop(index)
                _key_cooldown.pop(removed, None)
                _clients.pop(removed, None)
                db.set(settings_collection, "gemini_keys", gemini_keys)
                if index < current_key_index:
                    current_key_index -= 1
                current_key_index = max(0, min(current_key_index, len(gemini_keys) - 1))
                db.set(settings_collection, "current_key_index", current_key_index)
                await instant_reply(message.edit_text, [f"Key {key} deleted!"], {}, client)
            else:
                await instant_reply(message.edit_text, [f"Invalid key index: {key}"], {}, client)
            return
        if subcommand == "role":
            roles = await fetch_roles()
            if key:
                role_name = key.lower()
                if role_name in roles:
                    db.set(settings_collection, "default_role", role_name)
                    global _roles_cache_time
                    _roles_cache_time = 0  # re-map "default" on next fetch
                    await instant_reply(message.edit_text, [f"Default: {role_name}"], {}, client)
                else:
                    await instant_reply(message.edit_text, [f"Not found: {role_name}"], {}, client)
            else:
                roles_list = "\n".join(f"- {r}" for r in roles.keys()) if roles else "No roles found."
                await instant_reply(message.edit_text, [f"Available roles:\n{roles_list}"], {}, client)
            return
        if subcommand == "history":
            # two numbers first (head tail), otherwise a single number for both
            if len(command) > 3 and command[2].isdigit() and command[3].isdigit():
                head, tail = int(command[2]), int(command[3])
                db.set(settings_collection, "history_head", head)
                db.set(settings_collection, "history_tail", tail)
                await instant_reply(message.edit_text, [f"History head: {head}, tail: {tail}"], {}, client)
                return
            if key and key.isdigit():
                n = int(key)
                db.set(settings_collection, "history_head", n)
                db.set(settings_collection, "history_tail", n)
                await instant_reply(message.edit_text, [f"History head/tail set to: {n}"], {}, client)
                return

        keys = get_api_keys()
        keys_list = "\n".join(f"{i + 1}. {mask_key(k)}" for i, k in enumerate(keys)) or "None"
        current_key = mask_key(keys[current_key_index]) if keys and current_key_index < len(keys) else "None"
        head, tail = get_history_limits()
        menu_text = (
            f"Keys:\n{keys_list}\n\n"
            f"Current: {current_key}\nModel: {get_gemini_model()}\n"
            f"Max tokens: {get_max_output_tokens()}\n"
            f"Voice: {'ON' if get_voice_generation_enabled() else 'OFF'}\n"
            f"Role: {db.get(settings_collection, 'default_role') or 'default'}\n"
            f"History head: {head}, tail: {tail}"
        )
        if len(menu_text) > 3800:
            fp = f"gchat_menu_{int(time.time())}.txt"
            await asyncio.to_thread(_sync_write_file, fp, menu_text)
            await instant_reply(client.send_document, [message.chat.id, fp],
                                {"caption": "gchat menu", "cleanup_file": fp}, client)
        else:
            await instant_reply(message.edit_text, [menu_text], {}, client)
    except Exception as e:
        await instant_reply(client.send_message, ["me", f"⚠️ setgchat error\n{describe_exception(e)}"], {}, client)


@Client.on_message(filters.command(["gchat", "gc"], prefix) & filters.me)
async def gchat_command(client: Client, message: Message):
    global gchat_for_all
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await instant_reply(message.edit_text, ["Usage: gchat [on|off|del|all|r] [user_id]"], {}, client)
            return
        command = parts[1].lower()
        user_id = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else message.chat.id
        if command == "on":
            if user_id in disabled_users:
                disabled_users.remove(user_id)
                db.set(settings_collection, "disabled_users", disabled_users)
            if user_id not in enabled_users:
                enabled_users.append(user_id)
                db.set(settings_collection, "enabled_users", enabled_users)
            await instant_reply(message.edit_text, [f"<spoiler>ON: {user_id}</spoiler>"], {}, client)
        elif command == "off":
            if user_id not in disabled_users:
                disabled_users.append(user_id)
                db.set(settings_collection, "disabled_users", disabled_users)
            if user_id in enabled_users:
                enabled_users.remove(user_id)
                db.set(settings_collection, "enabled_users", enabled_users)
            await instant_reply(message.edit_text, [f"<spoiler>OFF: {user_id}</spoiler>"], {}, client)
        elif command == "del":
            db.remove(history_collection, f"chat_history.{user_id}")
            await instant_reply(message.edit_text, [f"<spoiler>Deleted: {user_id}</spoiler>"], {}, client)
        elif command == "all":
            gchat_for_all = not gchat_for_all
            db.set(settings_collection, "gchat_for_all", gchat_for_all)
            await instant_reply(message.edit_text, [f"All: {'enabled' if gchat_for_all else 'disabled'}"], {}, client)
        elif command == "r":
            changed = False
            if user_id in enabled_users:
                enabled_users.remove(user_id)
                db.set(settings_collection, "enabled_users", enabled_users)
                changed = True
            if user_id in disabled_users:
                disabled_users.remove(user_id)
                db.set(settings_collection, "disabled_users", disabled_users)
                changed = True
            await instant_reply(
                message.edit_text,
                [f"<spoiler>Removed: {user_id}</spoiler>" if changed else f"<spoiler>Not found: {user_id}</spoiler>"],
                {}, client)
        else:
            await instant_reply(message.edit_text, ["Usage: gchat [on|off|del|all|r] [user_id]"], {}, client)
        await instant_reply(message.delete, [], {}, client)
    except Exception as e:
        await instant_reply(client.send_message, ["me", f"⚠️ gchat command error\n{describe_exception(e)}"], {}, client)


@Client.on_message(filters.command("gswitch", prefix) & filters.me)
async def switch_role(client: Client, message: Message):
    try:
        roles = await fetch_roles()
        if not roles:
            await instant_reply(client.send_message, ["me", "⚠️ Role fetch error."], {}, client)
            await instant_reply(message.edit_text, ["Failed to fetch roles."], {}, client)
            return
        user_id = message.chat.id
        parts = message.text.strip().split()
        if len(parts) == 1:
            available_roles = "\n".join(f"- {r}" for r in roles.keys())
            await instant_reply(message.edit_text, [f"Roles:\n{available_roles}"], {}, client)
            return
        role_name = parts[1].lower()
        if role_name in roles:
            db.set(settings_collection, f"custom_roles.{user_id}", roles[role_name])
            await instant_reply(message.edit_text, [f"Switched: {role_name}"], {}, client)
        else:
            await instant_reply(message.edit_text, [f"Not found: {role_name}"], {}, client)
        await instant_reply(message.delete, [], {}, client)
    except Exception as e:
        await instant_reply(client.send_message, ["me", f"⚠️ switch command error\n{describe_exception(e)}"], {}, client)


@Client.on_message(filters.command("role", prefix) & filters.me)
async def set_custom_role(client: Client, message: Message):
    try:
        parts = message.text.strip().split()
        user_id = message.chat.id
        custom_role = None
        if len(parts) == 2 and parts[1].isdigit():
            user_id = int(parts[1])
        elif len(parts) > 2 and parts[1].isdigit():
            user_id = int(parts[1])
            custom_role = " ".join(parts[2:]).strip()
        elif len(parts) > 1:
            custom_role = " ".join(parts[1:]).strip()
        if not custom_role:
            db.remove(settings_collection, f"custom_roles.{user_id}")
            db.remove(history_collection, f"chat_history.{user_id}")
            await instant_reply(message.edit_text, [f"<spoiler>Role reset: {user_id}</spoiler>"], {}, client)
        else:
            db.set(settings_collection, f"custom_roles.{user_id}", custom_role)
            db.remove(history_collection, f"chat_history.{user_id}")
            await instant_reply(message.edit_text, [f"<spoiler>Role set: {user_id}</spoiler>\n{custom_role}"], {}, client)
        await instant_reply(message.delete, [], {}, client)
    except Exception as e:
        await instant_reply(client.send_message, ["me", f"⚠️ role command error\n{describe_exception(e)}"], {}, client)


@Client.on_message(filters.command("test", prefix) & filters.me)
async def test_keys(client: Client, message: Message):
    file_path = None
    try:
        await message.edit_text("Testing Gemini keys...")
        keys = get_api_keys()
        if not keys:
            await message.edit_text("No Gemini keys configured.")
            return
        model = get_gemini_model()
        config = build_generation_config(None, max_tokens=128)
        lines = ["Gemini API Key Test Results", "", f"Model: {model}", "-" * 40]
        for idx, key in enumerate(keys, 1):
            try:
                resp = await asyncio.wait_for(
                    get_client(key).aio.models.generate_content(
                        model=model, contents="ping", config=config
                    ),
                    timeout=60,
                )
                status = "OK" if extract_text(resp) else f"No text ({describe_empty(resp)})"
            except Exception as e:
                status = f"ERROR: {type(e).__name__}: {str(e)[:160]}"
            lines.append(f"{idx}. {mask_key(key)} → {status}")
        file_path = f"gemini_test_results_{int(time.time())}.txt"
        await asyncio.to_thread(_sync_write_file, file_path, "\n".join(lines))
        await client.send_document(
            chat_id=message.chat.id, document=file_path,
            caption="✅ Gemini API key test results",
        )
        await message.delete()
    except Exception as e:
        await client.send_message("me", f"⚠️ test command error\n{describe_exception(e)}")
    finally:
        _safe_remove(file_path)


modules_help["gchat"] = {
    "gchat on/off/del/all/r [user_id]": "Manage gchat for users.",
    "role [user_id] <role>": "Set or reset user role.",
    "gswitch [role]": "Show or switch the role of the current chat.",
    "setgchat add/set/del <key|index>": "Manage Gemini API keys.",
    "setgchat": "Show Gemini config & status.",
    "setgchat model <name>": "Set/show Gemini model.",
    "setgchat tokens <n>": "Set/show max output tokens per reply.",
    "setgchat voice": "Toggle voice reply.",
    "setgchat role <role>": "Set/show global role.",
    "setgchat history <n> | <head> <tail>": "Set chat history head/tail.",
    "setgchat picgroup <chat_id>": "Set the chat used by the .gpic feature.",
    "test": "Test Gemini keys",
}
