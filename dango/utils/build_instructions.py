"""
Build dynamic system prompt with contextual user information.
Called on every Agent.arun() via the instructions callable.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from .runtime_config import runtime_config as _global_runtime_config
from . import workspace_context


def build_instructions(
    base_prompt: str,
    author_name: str,
    unique_users: set[str],
    enable_contextual: bool,
    history_limit: int | None = None,
    timezone: str | None = None,
) -> str:
    """Return the system prompt, optionally enhanced with conversation context."""
    ctx = workspace_context.get()

    if not enable_contextual:
        return f"{base_prompt}\n\n---\n\n{ctx}" if ctx else base_prompt

    tz_name = timezone or _global_runtime_config.timezone
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    formatted_time = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")

    all_participants = unique_users | {author_name}
    participants_str = ", ".join(all_participants) if all_participants else "Unknown"
    limit = history_limit or _global_runtime_config.history_limit

    contextual = f"""
Priority Contextual System Guidance:

You have access to up to {limit} relevant messages from the current requester's conversation. Use only history that is relevant to the current request, and do not assume unrelated details apply.

Use retained historical details when they naturally help maintain continuity. Do not force personalization or repeat details that are unrelated to the current question.

Key information to use:
- The current requester is {author_name}. Address or reference them by this name only when appropriate, unless they specify otherwise.
- Retained conversation participants: {participants_str}.
- Current time: {formatted_time}
- Timezone: {tz_name}
"""
    parts = [base_prompt]
    if ctx:
        parts.append(ctx)
    parts.append(contextual)
    return "\n\n---\n\n".join(parts)
