"""API Server HTTP handlers — thin wrappers around gateway adapter methods."""

from api_server.handlers.chat_completions import handle_chat_completions, write_sse_chat_completion
from api_server.handlers.config import handle_get_config, handle_update_config
from api_server.handlers.health import handle_health, handle_health_detailed
from api_server.handlers.jobs import (
    handle_list_jobs,
    handle_create_job,
    handle_get_job,
    handle_update_job,
    handle_delete_job,
    handle_pause_job,
    handle_resume_job,
    handle_run_job,
)
from api_server.handlers.memory import (
    handle_get_memory,
    handle_add_memory,
    handle_replace_memory,
    handle_delete_memory,
)
from api_server.handlers.models import handle_models
from api_server.handlers.nodes import handle_list_nodes, handle_node_invoke
from api_server.handlers.responses import handle_responses, write_sse_responses
from api_server.handlers.runs import (
    handle_runs,
    handle_get_run,
    handle_run_events,
    handle_stop_run,
    sweep_orphaned_runs,
)
from api_server.handlers.sessions import (
    handle_list_sessions,
    handle_get_session,
    handle_create_session,
    handle_update_session,
    handle_delete_session,
    handle_get_session_messages,
    handle_fork_session,
)
from api_server.handlers.skills import handle_list_skills, handle_view_skill

__all__ = [
    "handle_chat_completions",
    "write_sse_chat_completion",
    "handle_get_config",
    "handle_update_config",
    "handle_health",
    "handle_health_detailed",
    "handle_list_jobs",
    "handle_create_job",
    "handle_get_job",
    "handle_update_job",
    "handle_delete_job",
    "handle_pause_job",
    "handle_resume_job",
    "handle_run_job",
    "handle_get_memory",
    "handle_add_memory",
    "handle_replace_memory",
    "handle_delete_memory",
    "handle_models",
    "handle_list_nodes",
    "handle_node_invoke",
    "handle_responses",
    "write_sse_responses",
    "handle_runs",
    "handle_get_run",
    "handle_run_events",
    "handle_stop_run",
    "sweep_orphaned_runs",
    "handle_list_sessions",
    "handle_get_session",
    "handle_create_session",
    "handle_update_session",
    "handle_delete_session",
    "handle_get_session_messages",
    "handle_fork_session",
    "handle_list_skills",
    "handle_view_skill",
]
