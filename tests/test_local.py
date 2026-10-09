import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core import confirm, undo
from core.local_backend import LocalChat, LocalTools
from core.tool_schema import gemini_tools_to_openai


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tools = LocalTools(self.temp.name)
        undo.clear()
        confirm.bind(None, None)

    def tearDown(self):
        confirm.resolve(False)
        confirm.bind(None, None)
        undo.clear()
        self.temp.cleanup()

    def test_schema_data_and_nested_types(self):
        declaration = {"name": "test", "parameters": {
            "type": "OBJECT", "properties": {"type": {
                "type": "ARRAY", "items": {"anyOf": [{"type": "STRING"}, {"type": "NULL"}]},
                "enum": ["OBJECT"]}}}}
        before = copy.deepcopy(declaration)
        params = gemini_tools_to_openai([declaration])[0]["function"]["parameters"]
        self.assertEqual(params["properties"]["type"]["items"]["anyOf"][0]["type"], "string")
        self.assertEqual(params["properties"]["type"]["enum"], ["OBJECT"])
        self.assertEqual(declaration, before)

    def test_note_and_undo(self):
        self.tools.run("write_note", {"name": "test.txt", "text": "before"})
        self.tools.run("write_note", {"name": "test.txt", "text": "after"})
        self.tools.run("undo_last", {})
        self.assertEqual((Path(self.temp.name) / "test.txt").read_text(), "before")
        self.tools.run("undo_last", {})
        self.assertFalse((Path(self.temp.name) / "test.txt").exists())

    def test_rejects_escape_symlink_unknown_and_invalid(self):
        for name in ("../outside.txt", "/tmp/outside.txt", "sub/note.txt"):
            self.assertIn("rejected", self.tools.run("write_note", {"name": name, "text": "bad"}))
        target = Path(self.temp.name) / "target"
        target.write_text("original")
        (Path(self.temp.name) / "link.txt").symlink_to(target)
        self.assertIn("rejected", self.tools.run("write_note", {"name": "link.txt", "text": "bad"}))
        self.assertEqual(target.read_text(), "original")
        self.assertIn("Unavailable", self.tools.run("run_shell", {"command": "docker stop all"}))
        self.assertIn("rejected", self.tools.run("write_note", {"name": "test.txt", "text": 1}))

    def test_confirmation_cannot_be_forged_by_model(self):
        path = Path(self.temp.name) / "test.txt"
        path.write_text("keep")
        self.assertIn("not done", self.tools.run("delete_note", {"name": "test.txt"}))
        shown = []
        confirm.bind(lambda *args: shown.append(args), lambda: None)
        self.assertIn("rejected", self.tools.run("delete_note", {"name": "test.txt", "confirmed": True}))
        self.assertIn("CONFIRMATION_PENDING", self.tools.run("delete_note", {"name": "test.txt"}))
        self.assertTrue(path.exists())
        confirm.resolve(False)
        self.assertTrue(path.exists())
        self.assertTrue(shown)

    def test_accepted_confirmation_and_undo(self):
        path = Path(self.temp.name) / "test.txt"
        path.write_text("restore me")
        complete = threading.Event()
        confirm.bind(lambda *args: None, lambda: None, lambda _: complete.set() if not path.exists() else None)
        self.tools.run("delete_note", {"name": "test.txt"})
        confirm.resolve(True)
        self.assertTrue(complete.wait(2))
        self.assertFalse(path.exists())
        self.tools.run("undo_last", {})
        self.assertEqual(path.read_text(), "restore me")

    def test_local_url_only(self):
        for url in ("https://example.com", "http://example.com", "http://localhost@evil.test", "http://localhost/api"):
            with self.assertRaises(ValueError):
                LocalChat(url)
        with self.assertRaises(ValueError):
            LocalChat(model="qwen3:cloud")

    def test_memory_persistence_and_undo(self):
        from memory import memory_manager
        path = Path(self.temp.name) / "memory.json"
        with patch.object(memory_manager, "MEMORY_PATH", path):
            self.tools.run("save_memory", {"key": "local_test", "value": "persistent fact"})
            self.assertIn("persistent fact", self.tools.run("recall_memory", {"query": "persistent"}))
            self.assertTrue(path.exists())
            self.tools.run("undo_last", {})
            self.assertNotIn("local_test", json.dumps(memory_manager.load_memory()))

    def test_tool_result_roundtrip_and_cancel(self):
        chat = LocalChat(tools=self.tools)
        responses = [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "write_note", "arguments": {"name": "test.txt", "text": "ok"}}}]}},
            {"message": {"role": "assistant", "content": "Written."}}]
        sent = []

        def post(url, **kwargs):
            sent.append(copy.deepcopy(kwargs["json"]))
            from unittest.mock import Mock
            return Mock(json=lambda: responses.pop(0), is_redirect=False)

        with patch.object(chat.session, "post", side_effect=post):
            self.assertEqual(chat.reply("Make a note"), "Written.")
            self.assertEqual(sent[1]["messages"][-1]["role"], "tool")
            self.assertEqual(sent[1]["messages"][-1]["tool_name"], "write_note")
            self.assertEqual(chat.reply("cancel", lambda: True), "")
        self.assertEqual(len(sent), 2)
        self.assertFalse(chat.session.trust_env)


if __name__ == "__main__":
    unittest.main()
