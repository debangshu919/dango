"""
Tests for step modules.
"""


class TestFetchAndProcessHistory:
    """Tests for fetch_and_process_history step."""

    def test_importable(self):
        from dango.steps.fetch_history import fetch_and_process_history

        assert fetch_and_process_history is not None

    def test_is_async(self):
        import asyncio

        from dango.steps.fetch_history import fetch_and_process_history

        assert asyncio.iscoroutinefunction(fetch_and_process_history)


class TestCallDiscordAgent:
    """Tests for call_discord_agent step."""

    def test_importable(self):
        from dango.steps.call_agent import call_discord_agent

        assert call_discord_agent is not None

    def test_is_async(self):
        import asyncio

        from dango.steps.call_agent import call_discord_agent

        assert asyncio.iscoroutinefunction(call_discord_agent)


class TestTextOnlyModelInput:
    def test_current_turn_does_not_include_image_payloads(self, monkeypatch):
        from types import SimpleNamespace

        import dango.steps.call_agent as call_agent
        from agno.run.base import RunStatus

        captured = {}
        agent = SimpleNamespace(model=None)

        async def capture_run(_agent, messages, _session_state):
            captured["messages"] = messages
            return SimpleNamespace(status=RunStatus.completed, content="reply")

        monkeypatch.setattr(call_agent, "_initialize_agents", lambda: None)
        monkeypatch.setattr(call_agent, "fast_agent", agent)
        monkeypatch.setattr(call_agent, "deep_agent", None)
        monkeypatch.setattr(
            call_agent,
            "_select_agent",
            lambda *args, **kwargs: (agent, "text-model", 0),
        )
        monkeypatch.setattr(call_agent, "_arun_agent", capture_run)

        message_data = {
            "author_name": "Alice",
            "author_id": 1,
            "content": "What is this?",
            "attachments": [{"url": "https://cdn.example/image.png", "content_type": "image/png"}],
            "stickers": [{"name": "wave", "url": "https://cdn.example/sticker.png"}],
            "_chat_sys_prompt": "system",
        }
        result = __import__("asyncio").run(
            call_agent.call_discord_agent(
                SimpleNamespace(
                    previous_step_content={
                        "message_data": message_data,
                        "formatted_history": [],
                    }
                )
            )
        )

        current_message = captured["messages"][-1]
        assert current_message.content == "Alice: What is this? [sticker: wave]"
        assert getattr(current_message, "images", None) is None
        assert "image.png" not in current_message.content
        assert result.content["llm_response"] == "reply"

    def test_history_messages_are_text_only(self):
        from types import SimpleNamespace

        from dango.steps.fetch_history import _process_messages

        sticker = SimpleNamespace(name="wave", url="https://cdn.example/sticker.png")
        message = SimpleNamespace(
            id=1,
            author=SimpleNamespace(id=123, display_name="Alice"),
            content="hello",
            stickers=[sticker],
            attachments=[SimpleNamespace(url="https://cdn.example/image.png")],
        )
        history, _ = _process_messages([message], 999, {})

        assert history[0].content == "Alice: hello [sticker: wave]"
        assert getattr(history[0], "images", None) is None
        assert "cdn.example" not in history[0].content


class TestGenshinWiki:
    def test_wiki_url_uses_api_reader(self, monkeypatch):
        import json

        import dango.steps.call_agent as call_agent

        monkeypatch.setattr(
            call_agent,
            "_get_genshin_wiki_articles",
            lambda titles: [{"title": titles[0], "text": "Profile text"}],
        )

        result = json.loads(
            call_agent._read_genshin_wiki_url(
                "https://genshin-impact.fandom.com/wiki/Columbina/Profile"
            )
        )

        assert result == {
            "articles": [{"title": "Columbina/Profile", "text": "Profile text"}]
        }

    def test_wiki_url_decodes_article_title(self, monkeypatch):
        import json

        import dango.steps.call_agent as call_agent

        monkeypatch.setattr(
            call_agent,
            "_get_genshin_wiki_articles",
            lambda titles: [{"title": titles[0], "text": "Character text"}],
        )

        result = json.loads(
            call_agent._read_genshin_wiki_url(
                "https://genshin-impact.fandom.com/wiki/Columbina_Hyposelenia"
            )
        )

        assert result["articles"][0]["title"] == "Columbina Hyposelenia"

    def test_non_wiki_url_is_not_intercepted(self):
        from dango.steps.call_agent import _read_genshin_wiki_url

        assert _read_genshin_wiki_url("https://example.com/wiki/Columbina") is None

    def test_grounding_policy_is_appended_last(self, monkeypatch):
        import dango.steps.call_agent as call_agent

        monkeypatch.setattr(call_agent, "ENABLE_CONTEXTUAL_SYSTEM_PROMPT", False)

        result = call_agent._dynamic_instructions({"chat_sys_prompt": "Base prompt"})

        assert result.startswith("Base prompt")
        assert result.endswith(call_agent._GENSHIN_GROUNDING_POLICY)
        assert "Never include citations, URLs, links" in result

    def test_search_results_contain_no_links(self, monkeypatch):
        import json

        import dango.steps.call_agent as call_agent

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"query": {"search": [{"title": "Columbina"}]}}

        monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
        monkeypatch.setattr(
            call_agent,
            "_get_genshin_wiki_articles",
            lambda titles: [{"title": titles[0], "text": "Character text"}],
        )

        result = json.loads(call_agent._search_genshin_wiki("Columbina"))

        assert result["articles"] == [
            {"title": "Columbina", "text": "Character text"}
        ]
        assert "http" not in json.dumps(result)


class TestExtractAndRenderTables:
    """Tests for extract_and_render_tables step."""

    def test_importable(self):
        from dango.steps.table_steps import extract_and_render_tables

        assert extract_and_render_tables is not None

    def test_is_async(self):
        import asyncio

        from dango.steps.table_steps import extract_and_render_tables

        assert asyncio.iscoroutinefunction(extract_and_render_tables)

    def test_parse_table(self):
        from dango.steps.table_steps import _parse_table

        table_text = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = _parse_table(table_text)
        assert result["valid"] is True
        assert result["headers"] == ["A", "B"]
        assert result["rows"] == [["1", "2"]]

    def test_parse_table_invalid(self):
        from dango.steps.table_steps import _parse_table

        result = _parse_table("| A |\n|---|")
        assert result["valid"] is False


class TestSendDiscordResponse:
    """Tests for send_discord_response step."""

    def test_importable(self):
        from dango.steps.send_response import send_discord_response

        assert send_discord_response is not None

    def test_is_async(self):
        import asyncio

        from dango.steps.send_response import send_discord_response

        assert asyncio.iscoroutinefunction(send_discord_response)


class TestBuildInstructions:
    """Tests for build_instructions utility."""

    def test_disabled_returns_base(self):
        from dango.utils.build_instructions import build_instructions

        result = build_instructions(
            base_prompt="Base prompt",
            author_name="Alice",
            unique_users=set(),
            enable_contextual=False,
        )
        assert result == "Base prompt"

    def test_enabled_includes_author(self):
        from dango.utils.build_instructions import build_instructions

        result = build_instructions(
            base_prompt="Base prompt",
            author_name="Alice",
            unique_users=set(),
            enable_contextual=True,
        )
        assert "Alice" in result
        assert "Base prompt" in result


class TestCreateDiscordWorkflow:
    """Tests for create_discord_workflow factory."""

    def test_creates_workflow(self, monkeypatch):
        import dango.workflow as workflow_module

        # FAST_MODEL is read at import time, so patch agent initialization
        # instead of the environment to keep the test env-independent.
        monkeypatch.setattr(workflow_module, "_initialize_agents", lambda: None)

        wf = workflow_module.create_discord_workflow()
        assert wf is not None
        assert wf.name == "DiscordAIPipeline"
