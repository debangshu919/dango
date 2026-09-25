import asyncio
from types import SimpleNamespace

from agno.models.message import Message


def _author(user_id, name):
    return SimpleNamespace(id=user_id, display_name=name)


def _message(message_id, author, content, reference_id=None):
    reference = (
        SimpleNamespace(message_id=reference_id, resolved=None)
        if reference_id is not None
        else None
    )
    return SimpleNamespace(
        id=message_id,
        author=author,
        content=content,
        clean_content=content,
        reference=reference,
        attachments=[],
        stickers=[],
        guild=None,
        interaction_metadata=None,
    )


class _Channel:
    def __init__(self, messages):
        self.id = 100
        self.messages = messages
        self.by_id = {message.id: message for message in messages}

    def history(self, limit, oldest_first):
        async def iterate():
            for message in self.messages[:limit]:
                yield message

        return iterate()

    async def fetch_message(self, message_id):
        return self.by_id[message_id]


async def _fetch_history(messages, requester_id, requester_name, current_message_id=999):
    from dango.steps.fetch_history import fetch_and_process_history

    channel = _Channel(messages)
    bot = SimpleNamespace(get_channel=lambda _channel_id: channel)
    message_data = {
        "_bot": bot,
        "_history_limit": 10,
        "channel_id": channel.id,
        "message_id": current_message_id,
        "bot_user_id": 900,
        "author_id": requester_id,
        "author_name": requester_name,
        "is_dm": False,
        "is_thread": False,
    }
    result = await fetch_and_process_history(SimpleNamespace(input=message_data))
    return result.content["formatted_history"]


class TestPerUserHistory:
    def test_shared_channel_isolates_alice_and_bob(self):
        alice = _author(1, "Alice")
        bob = _author(2, "Bob")
        bot = _author(900, "Dango")
        messages = [
            _message(5, bot, "Bob answer", 4),
            _message(4, bob, "BOB_SECRET"),
            _message(3, alice, "unrelated Alice chatter"),
            _message(2, bot, "Alice answer", 1),
            _message(1, alice, "ALICE_SECRET"),
        ]

        alice_history = asyncio.run(_fetch_history(messages, 1, "Alice"))
        bob_history = asyncio.run(_fetch_history(messages, 2, "Bob"))

        assert [message.content for message in alice_history] == [
            "Alice: ALICE_SECRET",
            "Alice answer",
        ]
        assert [message.content for message in bob_history] == [
            "Bob: BOB_SECRET",
            "Bob answer",
        ]
        assert all("BOB_SECRET" not in message.content for message in alice_history)
        assert all("ALICE_SECRET" not in message.content for message in bob_history)
        assert all("unrelated Alice chatter" not in message.content for message in alice_history)

    def test_queued_request_sees_completed_previous_exchange(self):
        alice = _author(1, "Alice")
        bot = _author(900, "Dango")
        messages = [
            _message(3, bot, "first answer", 1),
            _message(2, alice, "queued current message"),
            _message(1, alice, "first question"),
        ]

        history = asyncio.run(_fetch_history(messages, 1, "Alice", current_message_id=2))

        assert [message.content for message in history] == [
            "Alice: first question",
            "first answer",
        ]
        assert all("queued current message" not in message.content for message in history)

    def test_cross_user_reply_quotes_one_message_only(self):
        alice = _author(1, "Alice")
        bob = _author(2, "Bob")
        bot = _author(900, "Dango")
        messages = [
            _message(3, bot, "Alice answer", 2),
            _message(2, alice, "my reply", 1),
            _message(1, bob, "BOB_QUOTE_ONLY"),
        ]

        history = asyncio.run(_fetch_history(messages, 1, "Alice"))

        assert [message.content for message in history] == [
            'Alice: Bob said "BOB_QUOTE_ONLY" my reply',
            "Alice answer",
        ]

    def test_requester_reset_does_not_reset_another_user(self):
        from dango.steps.fetch_history import _select_history_messages

        alice = _author(1, "Alice")
        bot = _author(900, "Dango")
        post_user = _message(4, alice, "after reset")
        post_bot = _message(5, bot, "after answer", 4)
        reset = _message(3, bot, "[new chat] ---")
        old_user = _message(1, alice, "before reset")
        old_bot = _message(2, bot, "before answer", 1)
        messages = [post_bot, post_user, reset, old_bot, old_user]
        channel = _Channel(messages)

        async def select(reset_owner):
            return await _select_history_messages(
                msgs=messages,
                channel=channel,
                bot_user_id=900,
                requester_id=1,
                shared_channel=True,
                history_limit=10,
                current_message_id=999,
                dango_map={
                    3: {
                        "kind": "newchat",
                        "author_id": reset_owner,
                        "author_name": "Owner",
                    }
                },
            )

        alice_retained, _ = asyncio.run(select(1))
        bob_retained, _ = asyncio.run(select(2))

        assert [message.id for message in alice_retained] == [5, 4]
        assert [message.id for message in bob_retained] == [5, 4, 2, 1]

    def test_normalization_never_invents_assistant_turns(self):
        from dango.steps.fetch_history import _process_messages

        alice = _message(1, _author(1, "Alice"), "hello")
        bob = _message(2, _author(2, "Bob"), "hi")
        history, users = _process_messages([alice, bob], 900, {})

        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Alice: hello\nBob: hi"
        assert users == {"Alice", "Bob"}
        assert all(message.content != "..." for message in history)


class TestConversationLocks:
    def test_same_user_workflows_are_serialized(self):
        from dango.commands.chat_commands import ChatCog

        bot = SimpleNamespace(user=SimpleNamespace(id=900))
        cog = ChatCog(bot, SimpleNamespace(), "system", SimpleNamespace())
        active = 0
        max_active = 0

        async def run_locked(_messages):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1

        cog._run_workflow_locked = run_locked
        channel = SimpleNamespace(id=100)
        author = SimpleNamespace(id=1)
        first = SimpleNamespace(channel=channel, author=author)
        second = SimpleNamespace(channel=channel, author=author)

        async def run():
            await asyncio.gather(
                cog._run_workflow_for([first]),
                cog._run_workflow_for([second]),
            )

        asyncio.run(run())
        assert max_active == 1

    def test_different_users_can_run_concurrently(self):
        from dango.commands.chat_commands import ChatCog

        bot = SimpleNamespace(user=SimpleNamespace(id=900))
        cog = ChatCog(bot, SimpleNamespace(), "system", SimpleNamespace())
        active = 0
        max_active = 0

        async def run_locked(_messages):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1

        cog._run_workflow_locked = run_locked
        channel = SimpleNamespace(id=100)
        alice = SimpleNamespace(channel=channel, author=SimpleNamespace(id=1))
        bob = SimpleNamespace(channel=channel, author=SimpleNamespace(id=2))

        async def run():
            await asyncio.gather(
                cog._run_workflow_for([alice]),
                cog._run_workflow_for([bob]),
            )

        asyncio.run(run())
        assert max_active == 2


class TestContextTrimming:
    def test_drops_complete_exchange_with_selected_model(self, monkeypatch):
        from dango.steps.call_agent import _trim_to_token_budget

        model_ids = []

        def count_tokens(messages, model_id):
            model_ids.append(model_id)
            return len(messages) * 10

        monkeypatch.setattr("agno.utils.tokens.count_tokens", count_tokens)
        messages = [
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="new question"),
            Message(role="assistant", content="new answer"),
            Message(role="user", content="current"),
        ]

        trimmed = _trim_to_token_budget(messages, 30, "provider:deep-model")

        assert [message.content for message in trimmed] == [
            "new question",
            "new answer",
            "current",
        ]
        assert trimmed[0].role == "user"
        assert set(model_ids) == {"deep-model"}
