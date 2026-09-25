"""
Fetch Discord channel history and format it as Agno Messages.
Combines the original FetchDiscordHistory + ProcessMessageHistory nodes.
"""

import json
import re

import aiohttp
import discord
from agno.models.message import Message
from agno.workflow import StepInput, StepOutput

from ..utils.discord_helpers import ROLE_MENTION_RE, SYSINFO_MARKER, USER_MENTION_RE, format_reply_context, resolve_mentions


async def fetch_and_process_history(step_input: StepInput) -> StepOutput:
    """Fetch channel history and convert it to requester-scoped Agno Messages."""
    message_data = step_input.input
    bot = message_data["_bot"]
    history_limit = max(0, int(message_data["_history_limit"]))

    channel_id = message_data["channel_id"]
    message_id = message_data["message_id"]
    bot_user_id = message_data["bot_user_id"]
    requester_id = message_data["author_id"]
    shared_channel = not message_data.get("is_dm", False) and not message_data.get("is_thread", False)

    print(
        f"🔍 [fetch_and_process_history] Fetching channel {channel_id}, message {message_id}"
    )

    channel = bot.get_channel(channel_id)
    if not channel:
        try:
            channel = await bot.fetch_channel(channel_id)
        except (discord.NotFound, discord.Forbidden) as e:
            print(f"❌ [fetch_and_process_history] Cannot access channel: {e}")
            return StepOutput(
                content={
                    "error": True,
                    "error_message": f"Cannot access channel: {e}",
                    "message_data": message_data,
                }
            )

    try:
        scan_limit = min(max(history_limit * 20, 50), 500) if history_limit else 0
        msgs = [
            m
            async for m in channel.history(
                limit=scan_limit, oldest_first=False
            )
        ]

        dango_map = await _extract_dango_attachments(msgs, bot_user_id)
        retained, extra_sources = await _select_history_messages(
            msgs=msgs,
            channel=channel,
            bot_user_id=bot_user_id,
            requester_id=requester_id,
            shared_channel=shared_channel,
            history_limit=history_limit,
            current_message_id=message_id,
            dango_map=dango_map,
        )
        retained.reverse()

        print(
            f"📜 [fetch_and_process_history] Retained {len(retained)} of {len(msgs)} scanned messages"
        )

        table_content_map = await _extract_table_attachments(retained)
        deep_map = {
            message_id: info
            for message_id, info in dango_map.items()
            if info["kind"] in {"deep", "skill"}
        }
        reply_map, reply_sources = await _build_reply_map(
            retained, channel, bot_user_id
        )
        mention_map = await _build_mention_map(
            retained + extra_sources + reply_sources
        )
        formatted_history, unique_users = _process_messages(
            retained, bot_user_id, table_content_map, deep_map, mention_map, reply_map
        )

        return StepOutput(
            content={
                "formatted_history": formatted_history,
                "unique_users": list(unique_users),
                "mention_map": mention_map,
                "message_data": message_data,
            }
        )

    except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
        print(f"❌ [fetch_and_process_history] Discord error: {e}")
        return StepOutput(
            content={
                "error": True,
                "error_message": f"Discord error: {e}",
                "message_data": message_data,
            }
        )


async def _extract_table_attachments(msgs: list) -> dict:
    """Download table attachment content from previous bot messages."""
    table_content_map = {}
    for msg in msgs:
        if not msg.attachments:
            continue
        for attachment in msg.attachments:
            if attachment.filename.startswith(
                "dango_replaced_table_"
            ) and attachment.filename.endswith(".md"):
                parts = attachment.filename.replace(".md", "").split("_")
                if len(parts) >= 5 and parts[3].isdigit() and parts[4].isdigit():
                    key = f"{parts[3]}_{parts[4]}"
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(attachment.url) as resp:
                                if resp.status == 200:
                                    table_content_map[key] = await resp.text()
                    except Exception as e:
                        print(
                            f"❌ [fetch_and_process_history] Failed to download table attachment: {e}"
                        )
    return table_content_map


def _dango_attachment_kind(filename: str) -> str | None:
    if filename.startswith("dango_deep_") and filename.endswith(".json"):
        return "deep"
    if filename.startswith("dango_skill_") and filename.endswith(".json"):
        return "skill"
    if filename.startswith("dango_newchat_") and filename.endswith(".json"):
        return "newchat"
    return None


async def _extract_dango_attachments(msgs: list, bot_user_id: int) -> dict[int, dict]:
    """Read validated Dango command/reset metadata from bot-authored messages."""
    candidates = []
    for msg in msgs:
        if msg.author.id != bot_user_id:
            continue
        for attachment in getattr(msg, "attachments", []):
            inferred_kind = _dango_attachment_kind(attachment.filename)
            if inferred_kind:
                candidates.append((msg, attachment, inferred_kind))

    dango_map: dict[int, dict] = {}
    if not candidates:
        return dango_map

    async with aiohttp.ClientSession() as session:
        for msg, attachment, inferred_kind in candidates:
            try:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        continue
                    info = json.loads(await resp.text())
            except Exception as e:
                print(
                    f"❌ [fetch_and_process_history] Failed to download Dango attachment: {e}"
                )
                continue

            if not isinstance(info, dict):
                continue
            try:
                author_id = int(info["author_id"])
            except (KeyError, TypeError, ValueError):
                continue

            kind = info.get("kind", inferred_kind)
            if kind not in {"deep", "skill", "newchat"}:
                continue
            if kind in {"deep", "skill"} and not isinstance(info.get("content"), str):
                continue

            dango_map[msg.id] = {
                "version": info.get("version", 0),
                "kind": kind,
                "author_id": author_id,
                "author_name": str(info.get("author_name", "User")),
                "content": info.get("content", ""),
            }

    return dango_map


def _interaction_owner_id(msg) -> int | None:
    metadata = getattr(msg, "interaction_metadata", None) or getattr(msg, "interaction", None)
    user = getattr(metadata, "user", None)
    user_id = getattr(user, "id", None)
    return int(user_id) if user_id is not None else None


def _is_reset_for_requester(
    msg,
    dango_info: dict | None,
    requester_id: int,
    shared_channel: bool,
) -> bool:
    if dango_info and dango_info["kind"] == "newchat":
        return not shared_channel or dango_info["author_id"] == requester_id
    if "[new chat] ---" not in getattr(msg, "content", ""):
        return False
    interaction_owner = _interaction_owner_id(msg)
    if interaction_owner is not None:
        return not shared_channel or interaction_owner == requester_id
    return not shared_channel


async def _resolve_message_owner(
    msg,
    channel,
    bot_user_id: int,
    message_cache: dict[int, object],
    dango_map: dict[int, dict],
    extra_sources: list,
    visited: set[int] | None = None,
) -> int | None:
    if msg.author.id != bot_user_id:
        return int(msg.author.id)

    dango_info = dango_map.get(msg.id)
    if dango_info and dango_info["kind"] in {"deep", "skill"}:
        return dango_info["author_id"]

    reference = getattr(msg, "reference", None)
    reference_id = getattr(reference, "message_id", None)
    if reference_id is None:
        return None

    visited = visited or set()
    if msg.id in visited or len(visited) >= 4:
        return None
    visited.add(msg.id)

    source = message_cache.get(reference_id)
    if source is None:
        resolved = getattr(reference, "resolved", None)
        if resolved is not None and hasattr(resolved, "author"):
            source = resolved
        else:
            try:
                source = await channel.fetch_message(reference_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return None
        message_cache[source.id] = source
        extra_sources.append(source)
        dango_map.update(await _extract_dango_attachments([source], bot_user_id))

    return await _resolve_message_owner(
        source,
        channel,
        bot_user_id,
        message_cache,
        dango_map,
        extra_sources,
        visited,
    )


async def _select_history_messages(
    msgs: list,
    channel,
    bot_user_id: int,
    requester_id: int,
    shared_channel: bool,
    history_limit: int,
    current_message_id: int | None,
    dango_map: dict[int, dict],
) -> tuple[list, list]:
    """Select newest requester-owned events until the relevant history limit."""
    retained = []
    extra_sources = []
    message_cache = {msg.id: msg for msg in msgs}
    scoped_msgs = []

    for msg in msgs:
        if msg.id == current_message_id:
            continue
        is_newer = current_message_id is not None and msg.id > current_message_id
        dango_info = dango_map.get(msg.id)
        if is_newer and (
            msg.author.id != bot_user_id
            or (dango_info and dango_info["kind"] == "newchat")
        ):
            continue
        if _is_reset_for_requester(
            msg, dango_info, requester_id, shared_channel
        ):
            print("✂️ [fetch_and_process_history] Reached requester reset marker")
            break
        scoped_msgs.append(msg)

    replied_source_ids: set[int] = set()
    owned_bot_ids: set[int] = set()
    if shared_channel:
        for msg in scoped_msgs:
            if msg.author.id != bot_user_id:
                continue
            owner_id = await _resolve_message_owner(
                msg,
                channel,
                bot_user_id,
                message_cache,
                dango_map,
                extra_sources,
            )
            if owner_id == requester_id:
                owned_bot_ids.add(msg.id)
                reference_id = getattr(getattr(msg, "reference", None), "message_id", None)
                if reference_id is not None:
                    replied_source_ids.add(reference_id)

    retained_user_turns = 0
    for msg in scoped_msgs:
        dango_info = dango_map.get(msg.id)
        if "[new chat] ---" in getattr(msg, "content", ""):
            continue
        if dango_info and dango_info["kind"] == "newchat":
            continue
        if getattr(msg, "content", "").strip().startswith(SYSINFO_MARKER):
            continue

        if not shared_channel:
            keep = True
        elif msg.author.id == bot_user_id:
            if dango_info and dango_info["kind"] in {"deep", "skill"}:
                keep = msg.id in replied_source_ids
            else:
                keep = msg.id in owned_bot_ids
        else:
            keep = msg.author.id == requester_id and msg.id in replied_source_ids

        if keep:
            retained.append(msg)
            is_user_turn = msg.author.id != bot_user_id or (
                dango_info and dango_info["kind"] in {"deep", "skill"}
            )
            if is_user_turn:
                retained_user_turns += 1
                if retained_user_turns >= history_limit:
                    break

    return retained, extra_sources


async def _build_mention_map(msgs: list) -> dict[str, str]:
    """Collect mention tokens in retained messages and resolve display names."""
    guild = next((m.guild for m in msgs if getattr(m, "guild", None)), None)
    if guild is None:
        return {}

    user_ids: set[int] = set()
    role_ids: set[int] = set()
    for msg in msgs:
        content = getattr(msg, "content", "")
        for match in USER_MENTION_RE.finditer(content):
            user_ids.add(int(match.group(1)))
        for match in ROLE_MENTION_RE.finditer(content):
            role_ids.add(int(match.group(1)))

    mention_map: dict[str, str] = {}

    for uid in user_ids:
        member = guild.get_member(uid)
        if member is None:
            try:
                member = await guild.fetch_member(uid)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass
        name = member.display_name if member else str(uid)
        mention_map[f"<@{uid}>"] = f"@{name}"
        mention_map[f"<@!{uid}>"] = f"@{name}"

    for rid in role_ids:
        role = guild.get_role(rid)
        name = role.name if role else str(rid)
        mention_map[f"<@&{rid}>"] = f"@{name}"

    if mention_map:
        print(f"🏷️  [fetch_and_process_history] Resolved {len(user_ids)} user(s), {len(role_ids)} role mention(s)")
    return mention_map


async def _build_reply_map(msgs: list, channel, bot_user_id: int) -> tuple[dict[int, dict], list]:
    """Build one-hop reply context for retained human messages."""
    local_map = {msg.id: msg for msg in msgs}
    reply_map: dict[int, dict] = {}
    extra_msgs: list = []

    for msg in msgs:
        if msg.author.id == bot_user_id:
            continue
        reference = getattr(msg, "reference", None)
        reference_id = getattr(reference, "message_id", None)
        if reference_id is None:
            continue
        ref_msg = local_map.get(reference_id)
        if ref_msg is None:
            resolved = getattr(reference, "resolved", None)
            if resolved is not None and hasattr(resolved, "author"):
                ref_msg = resolved
            else:
                try:
                    ref_msg = await channel.fetch_message(reference_id)
                    extra_msgs.append(ref_msg)
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    continue
        if ref_msg and ref_msg.content:
            reply_map[msg.id] = {
                "author_name": ref_msg.author.display_name,
                "content": ref_msg.content,
            }

    if reply_map:
        print(
            f"↩️  [fetch_and_process_history] Resolved {len(reply_map)} reply reference(s) in history"
            + (f" ({len(extra_msgs)} fetched outside window)" if extra_msgs else "")
        )
    return reply_map, extra_msgs


def _replace_table_placeholders(content: str, table_map: dict) -> str:
    """Replace table image placeholders with actual table markdown."""
    placeholder_pattern = r"> `\[dango_replaced_table_(\d+)_(\d+)_as_image\]`"

    def replacer(match):
        key = f"{match.group(1)}_{match.group(2)}"
        return table_map.get(key, match.group(0))

    return re.sub(placeholder_pattern, replacer, content)


def _process_messages(
    msgs: list,
    bot_user_id: int,
    table_content_map: dict,
    deep_map: dict | None = None,
    mention_map: dict | None = None,
    reply_map: dict | None = None,
) -> tuple[list[Message], set[str]]:
    """Normalize retained Discord messages into real text-only model turns."""
    deep_map = deep_map or {}
    mention_map = mention_map or {}
    reply_map = reply_map or {}
    raw_messages = []
    unique_users: set[str] = set()

    for msg in msgs:
        if msg.id in deep_map:
            info = deep_map[msg.id]
            author_name = info.get("author_name", "User")
            content = resolve_mentions(info.get("content", ""), mention_map)
            if not content:
                continue
            unique_users.add(author_name)
            raw_messages.append({
                "role": "user",
                "content": content,
                "author_id": info.get("author_id"),
                "author_name": author_name,
            })
        elif msg.author.id == bot_user_id:
            content = resolve_mentions(msg.content.strip(), mention_map)
            if not content or content.startswith(SYSINFO_MARKER):
                continue
            raw_messages.append(
                {"role": "assistant", "content": content}
            )
        else:
            author_name = msg.author.display_name
            content = resolve_mentions(msg.content.strip(), mention_map)
            if msg.id in reply_map:
                ref_info = reply_map[msg.id]
                ref_content = resolve_mentions(ref_info["content"], mention_map)
                content = format_reply_context(
                    current_author=author_name,
                    ref_author=ref_info["author_name"],
                    ref_content=ref_content,
                    current_content=content,
                )
            sticker_names = [
                sticker.name
                for sticker in getattr(msg, "stickers", [])
                if sticker.name
            ]
            if sticker_names:
                note = f"[sticker: {', '.join(sticker_names)}]"
                content = f"{content} {note}" if content else note
            if not content:
                continue
            unique_users.add(author_name)
            raw_messages.append(
                {
                    "role": "user",
                    "content": content,
                    "author_id": msg.author.id,
                    "author_name": author_name,
                }
            )

    normalized: list[dict] = []
    for current in raw_messages:
        if current["role"] == "assistant":
            if normalized and normalized[-1]["role"] == "assistant":
                normalized[-1]["content"] += f"\n{current['content']}"
            else:
                normalized.append({"role": "assistant", "content": current["content"]})
            continue

        labelled = f"{current['author_name']}: {current['content']}"
        if normalized and normalized[-1]["role"] == "user":
            normalized[-1]["content"] += f"\n{labelled}"
        else:
            normalized.append({"role": "user", "content": labelled})

    if table_content_map:
        for msg in normalized:
            msg["content"] = _replace_table_placeholders(
                msg["content"], table_content_map
            )

    while normalized and normalized[0]["role"] == "assistant":
        normalized.pop(0)

    formatted_history = [
        Message(role=m["role"], content=m["content"])
        for m in normalized
    ]
    print(
        f"✅ [fetch_and_process_history] {len(formatted_history)} formatted messages, {len(unique_users)} unique users"
    )
    return formatted_history, unique_users
