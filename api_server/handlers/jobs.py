import re
from typing import Optional
from aiohttp import web

try:
    from tools.cronjob import (
        list_jobs as _cron_list,
        get_job as _cron_get,
        create_job as _cron_create,
        update_job as _cron_update,
        remove_job as _cron_remove,
        pause_job as _cron_pause,
        resume_job as _cron_resume,
        trigger_job as _cron_trigger,
    )
    _CRON_AVAILABLE = True
except ImportError:
    _CRON_AVAILABLE = False
    _cron_list = None
    _cron_get = None
    _cron_create = None
    _cron_update = None
    _cron_remove = None
    _cron_pause = None
    _cron_resume = None
    _cron_trigger = None

_MAX_NAME_LENGTH = 100
_MAX_PROMPT_LENGTH = 5000
_UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "repeat"}
_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _check_jobs_available() -> Optional[web.Response]:
    if not _CRON_AVAILABLE:
        return web.json_response({"error": "Cron module not available"}, status=501)
    return None


def _check_job_id(request: web.Request) -> tuple:
    job_id = request.match_info["job_id"]
    if not _JOB_ID_RE.fullmatch(job_id):
        return job_id, web.json_response({"error": "Invalid job ID format"}, status=400)
    return job_id, None


async def handle_list_jobs(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    try:
        include_disabled = request.query.get("include_disabled", "").lower() in ("true", "1")
        jobs = _cron_list(include_disabled=include_disabled)
        return web.json_response({"jobs": jobs})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_create_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    try:
        body = await request.json()
        name = (body.get("name") or "").strip()
        schedule = (body.get("schedule") or "").strip()
        prompt = body.get("prompt", "")
        deliver = body.get("deliver", "local")
        skills = body.get("skills")
        repeat = body.get("repeat")

        if not name:
            return web.json_response({"error": "Name is required"}, status=400)
        if len(name) > _MAX_NAME_LENGTH:
            return web.json_response(
                {"error": f"Name must be ≤ {_MAX_NAME_LENGTH} characters"}, status=400,
            )
        if not schedule:
            return web.json_response({"error": "Schedule is required"}, status=400)
        if len(prompt) > _MAX_PROMPT_LENGTH:
            return web.json_response(
                {"error": f"Prompt must be ≤ {_MAX_PROMPT_LENGTH} characters"}, status=400,
            )
        if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
            return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

        kwargs = {
            "prompt": prompt,
            "schedule": schedule,
            "name": name,
            "deliver": deliver,
        }
        if skills:
            kwargs["skills"] = skills
        if repeat is not None:
            kwargs["repeat"] = repeat

        job = _cron_create(**kwargs)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        job = _cron_get(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_update_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        body = await request.json()
        sanitized = {k: v for k, v in body.items() if k in _UPDATE_ALLOWED_FIELDS}
        if not sanitized:
            return web.json_response({"error": "No valid fields to update"}, status=400)
        if "name" in sanitized and len(sanitized["name"]) > _MAX_NAME_LENGTH:
            return web.json_response(
                {"error": f"Name must be ≤ {_MAX_NAME_LENGTH} characters"}, status=400,
            )
        if "prompt" in sanitized and len(sanitized["prompt"]) > _MAX_PROMPT_LENGTH:
            return web.json_response(
                {"error": f"Prompt must be ≤ {_MAX_PROMPT_LENGTH} characters"}, status=400,
            )
        job = _cron_update(job_id, sanitized)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_delete_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        success = _cron_remove(job_id)
        if not success:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_pause_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        job = _cron_pause(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_resume_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        job = _cron_resume(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_run_job(request: web.Request, *, check_auth) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    cron_err = _check_jobs_available()
    if cron_err:
        return cron_err
    job_id, id_err = _check_job_id(request)
    if id_err:
        return id_err
    try:
        job = _cron_trigger(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
