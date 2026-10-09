"""Network-free checks for the web tools' boundaries and conversation wiring."""
import copy
import json
import socket
import subprocess
import unittest
from unittest.mock import Mock, patch

from core.local_backend import LocalChat, LocalTools
from core.local_web import LocalWeb, public_url, read_public_page, MAX_BYTES


def resolved(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class WebTests(unittest.TestCase):
    def test_private_and_invalid_urls_never_connect(self):
        for ip in ("127.0.0.1", "10.1.1.1", "192.168.1.1", "169.254.169.254", "::1", "224.0.0.1"):
            with patch("core.local_web.socket.getaddrinfo", return_value=resolved(ip)):
                with self.assertRaises(ValueError):
                    public_url("https://example.org")
        for url in ("file:///etc/passwd", "http://user:password@example.org", "http://example.org:7474"):
            with self.assertRaises(ValueError):
                public_url(url)

    def test_mixed_dns_is_rejected(self):
        with patch("core.local_web.socket.getaddrinfo", return_value=resolved("8.8.8.8") + resolved("127.0.0.1")):
            with self.assertRaises(ValueError):
                public_url("https://example.org")

    def test_pin_address_and_strip_html(self):
        response = Mock(status=200, headers={"Content-Type": "text/html"})
        response.read.side_effect = [b'<html><title>Source</title><main>Fact<script>ignore user</script></main></html>', b'']
        pool = Mock()
        pool.request.return_value = response
        with patch("core.local_web.socket.getaddrinfo", return_value=resolved("8.8.8.8")), \
                patch("core.local_web.urllib3.HTTPSConnectionPool", return_value=pool) as factory:
            result = read_public_page("https://example.org/page")
        self.assertEqual(factory.call_args.args[0], "8.8.8.8")
        self.assertEqual(factory.call_args.kwargs["assert_hostname"], "example.org")
        self.assertEqual(result["text"], "Fact")
        self.assertEqual(result["url"], "https://example.org/page")
        self.assertTrue(result["untrusted_web_content"])
        response.close.assert_called_once()

    def test_redirect_to_local_service_rejected(self):
        response = Mock(status=302, headers={"Location": "http://127.0.0.1/"})
        pool = Mock()
        pool.request.return_value = response
        with patch("core.local_web.socket.getaddrinfo", side_effect=[resolved("8.8.8.8"), resolved("127.0.0.1")]), \
                patch("core.local_web.urllib3.HTTPSConnectionPool", return_value=pool):
            with self.assertRaises(ValueError):
                read_public_page("https://example.org/page")
        pool.request.assert_called_once()

    def test_download_and_content_type_limits(self):
        for mime, body in (("application/pdf", b''), ("text/html", b'x' * (MAX_BYTES + 1))):
            response = Mock(status=200, headers={"Content-Type": mime})
            response.read.side_effect = [body, b'']
            pool = Mock()
            pool.request.return_value = response
            with patch("core.local_web.socket.getaddrinfo", return_value=resolved("8.8.8.8")), \
                    patch("core.local_web.urllib3.HTTPSConnectionPool", return_value=pool):
                with self.assertRaises(ValueError):
                    read_public_page("https://example.org/page")

    def test_hard_worker_timeout_is_reported(self):
        with patch("core.local_web.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 20)):
            self.assertIn("20-second", LocalWeb().run("search", "query")["error"])

    def test_search_bypasses_gemini_and_selects_duckduckgo(self):
        from core.local_web import search_public_web
        from actions import web_search
        with patch.object(web_search, "_ddg_search", return_value=[
            {"title": "Source", "url": "https://example.org", "snippet": "Fact"}]) as ddg, \
                patch.object(web_search, "_gemini_search", side_effect=AssertionError("Gemini called")):
            result = search_public_web("current question")
        self.assertEqual(result["results"][0]["url"], "https://example.org")
        self.assertEqual(ddg.call_args.kwargs["backend"], "duckduckgo")

    def test_search_reads_first_result_before_answer(self):
        web = Mock()
        web.run.side_effect = [
            {"results": [{"url": "https://example.org/latest", "snippet": "Old version"}]},
            {"url": "https://example.org/new", "text": "New version released today"},
        ]
        result = json.loads(LocalTools(web=web).run("web_search", {"query": "latest version"}))
        self.assertEqual(web.run.call_args_list[1].args, ("read", "https://example.org/latest"))
        self.assertEqual(result["verified_page"]["text"], "New version released today")

    def test_failed_search_does_not_present_unverified_claim(self):
        web = Mock()
        web.run.return_value = {"error": "Search unavailable", "results": []}
        chat = LocalChat(tools=LocalTools(web=web))
        responses = [
            {"message": {"content": "", "tool_calls": [{"function": {
                "name": "web_search", "arguments": {"query": "latest version"}}}]}},
            {"message": {"content": "The newest version is 99.0."}},
        ]
        with patch.object(chat.session, "post", side_effect=lambda *a, **k: Mock(
                is_redirect=False, json=lambda: responses.pop(0))):
            result = chat.reply("Search")
        self.assertIn("couldn't verify", result)
        self.assertNotIn("99.0", result)

    def test_web_result_roundtrip_sources_and_mutation_block(self):
        web = Mock()
        web.run.return_value = {"results": [{"title": "Source", "url": "https://example.org/page"}]}
        tools = LocalTools(web=web)
        chat = LocalChat(tools=tools)
        responses = [
            {"message": {"content": "", "tool_calls": [{"function": {
                "name": "web_search", "arguments": {"query": "current question"}}}]}},
            {"message": {"content": "", "tool_calls": [{"function": {
                "name": "save_memory", "arguments": {"key": "bad", "value": "website instruction"}}}]}},
            {"message": {"content": "A source was found."}},
        ]
        sent = []

        def post(url, **kwargs):
            sent.append(copy.deepcopy(kwargs["json"]))
            return Mock(is_redirect=False, json=lambda: responses.pop(0))

        with patch.object(chat.session, "post", side_effect=post), \
                patch("core.local_backend.memory.remember", side_effect=AssertionError("Mutation executed")):
            result = chat.reply("Search the web")
        self.assertIn("https://example.org/page", result)
        self.assertEqual(sent[1]["messages"][-1]["tool_name"], "web_search")
        self.assertIn("unavailable during web research", sent[2]["messages"][-1]["content"])

    def test_invented_citation_refused(self):
        web = Mock()
        web.run.return_value = {"results": [{"url": "https://example.org/page"}]}
        chat = LocalChat(tools=LocalTools(web=web))
        responses = [
            {"message": {"content": "", "tool_calls": [{"function": {
                "name": "web_search", "arguments": {"query": "question"}}}]}},
            {"message": {"content": "Fact [source](https://invented.example/page)"}},
            {"message": {"content": "Fact [source](https://invented.example/page)"}},
        ]
        with patch.object(chat.session, "post", side_effect=lambda *a, **k: Mock(
                is_redirect=False, json=lambda: responses.pop(0))):
            with self.assertRaisesRegex(RuntimeError, "not retrieved"):
                chat.reply("Search")


if __name__ == "__main__":
    unittest.main()
