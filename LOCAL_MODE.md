# Local Mark-LV on the Fedora workstation

Installed in `/home/dexter/Downloads/Mark-LV-main` on the `local-ollama` branch.
The downloaded folder had no host Git repository, so a separate repository was
initialized here. No changes were made to the unrelated home Git repository.

## Launch

Open a terminal and keep the project Ollama server running:

```bash
cd /home/dexter/Downloads/Mark-LV-main
bash start-local-ollama.sh
```

In a second terminal, choose one of these commands:

```bash
cd /home/dexter/Downloads/Mark-LV-main
bash launch-local.sh              # HUD, text input, microphone off
bash launch-local.sh --voice      # HUD, local microphone and speech output
bash launch-local.sh --cli        # terminal text chat
```

You can also run `.venv/bin/python main.py --local --voice` directly.
Start with: “Use system_status to report RAM usage.” Try “Write a note named
test.txt saying hello”, then `/undo`. Notes stay in `.local-data/notes`.
Deleting a note requires the human CONFIRM button on the HUD. In terminal mode
deletion is refused because there is no confirmation interface.

The microphone is only opened with `--voice`. F4 mutes it; Escape interrupts
speech and prevents further tool execution after the current HTTP response.
The current inference request may finish in the background. Voice is half-duplex:
the microphone is discarded during inference and playback, so use Escape to
interrupt, rather than speaking over JARVIS. Speech segmentation uses a simple
volume threshold and 0.8 seconds of silence, with a 15-second utterance cap.
Use the existing AUDIO DEVICES panel for device selection; speaker changes
apply to the next reply, while microphone changes require a restart.

Close the HUD or use `/quit` in terminal mode. Ctrl+C in the server terminal
stops only that Ollama process. No startup or system service was installed.
If an existing Ollama listener is found, the server script reuses it.
Local mode checks model capabilities and refuses remote/cloud models.

## What changed

- `main.py --local` routes before importing Gemini, using `local_main.py`.
- The original PyQt HUD, file-backed memory, undo stack and human confirmation
  gate are shared with the local backend. Local startup skips Gemini setup.
- `core/local_backend.py` implements bounded Ollama chat and validated local
  tools: system status, save/recall memory, write/delete notes, undo, web search,
  and public webpage reading.
- `core/local_web.py` reuses the existing DDGS search helper directly, bypassing
  all Gemini search modes. It reads the first search result to check stale
  snippets, and exposes `read_webpage` for additional sources.
- `core/tool_schema.py` adapts the Gemini-to-Ollama schema converter reviewed in
  [PR #112](https://github.com/FatihMakes/Mark-LV/pull/112), by Abhayraj69.
  The whole PR was not merged: it includes unrelated UI, routing, and cloud paths.
- `core/local_speech.py` uses CPU/int8 faster-whisper and eSpeak NG. Models must
  already be present; runtime does not download speech models or use cloud TTS.
- `requirements-local.txt` excludes Google SDKs and cloud speech dependencies.
- Fixed `.gitignore` inline comments that prevented credential paths matching.

`setup.py` was reviewed before installation. It installs the entire cloud
dependency list and Playwright browsers, so it was not executed for local mode.
Only the local requirements were installed in `.venv` with uv; the exact
installed versions are recorded in `requirements-local.lock`. System Python
3.14 was left intact; the project uses bundled Python 3.13.16.

## Hardware, models and service isolation

Verified Fedora 44, Ryzen 9 8945HS, Radeon 780M, 46.3 GiB RAM. Initially about
22 GiB RAM was available while OpenMono containers were running. Ollama 0.40.2
was installed under `.tools/ollama`, because no existing installation/listener
was found. Qwen2.5 7B Q4_K_M is stored under `.local-data/ollama-models` (download
about 4.7 GB). The base Whisper model is under `.local-data/whisper-base`.

Ollama detected the 780M through Vulkan and enabled it with
`OLLAMA_IGPU_ENABLE=1`. Live inference reported 5,119,263,374 bytes in
`size_vram` with an 8192-token context, and `ollama ps` reported 100% GPU.
Qwen2.5 7B was chosen after smaller 3B/4B candidates produced verbose or empty
answers and inconsistent tool routing. Their test downloads remain in the
project-local store. This verifies Vulkan inference on this
machine; ROCm was not installed. The iGPU shares system RAM. Ollama is limited
to one loaded model and one simultaneous request, with a five-minute idle
unload; CPU inference uses four threads and Whisper uses two.

The only added fixed listener is loopback `127.0.0.1:11434`. Existing OpenMono
ports 7474 and 47480 were left untouched. The four observed containers were
`openmono-scrapling`, `openmono-searxng`, `openmono-gateway`, and `llama-server`.
No running Splunk container was listed at inspection time. No containers,
networks, images, configurations, GPU drivers or system Python were modified.

## Remaining limitations

Local mode provides the listed tools, rather than full Gemini action parity.
Screen/webcam vision, attachment processing, browser automation,
cloud plugins, phone dashboard, wake-word gating, push-to-talk, proactive
briefings and Gemini voice selection are unavailable. Unsupported HUD controls
are disabled where applicable. Desktop shortcut and auto-start controls are
disabled because the original versions launch cloud mode. eSpeak's voice is
basic rather than Gemini's natural streaming voice. Replies are not streamed.
Small models can still choose wrong tools or repeat calls; schema validation,
bounded notes and human confirmation constrain their effects.

Memory is shared with the original app and is readable in the HUD. Undo is
in-memory and lasts only for the current process. Conversation history retains
six recent completed turns. No shell, container or service management
tools are exposed to the local model.

## Web search in local mode

Restart the Mark-LV app to load the added tools; the launch commands are unchanged.
Ask, for example: “What is the latest stable Python release? Search the web and
verify the official release page, with a source link.” Ordinary timeless
questions can still be answered offline.

Search uses DuckDuckGo through the existing helper in `actions/web_search.py`.
It needs no account, API key, new server, or changes to OpenMono's SearXNG.
Queries go to DuckDuckGo and reading a result contacts that website. The AI
model and speech engines remain local; Gemini is never used by these tools.

Each search returns at most five results and reads the first result before
answer generation. Each search/read worker has a hard 20-second deadline;
search plus automatic reading can therefore take up to 40 seconds. Empty
DuckDuckGo results are retried once against the same provider. Page downloads
are limited to 256 KB and extracted text to 8,000 characters, with three
redirects allowed. Only public HTTP/HTTPS text or HTML on ports 80/443 is accepted.
Private addresses and redirects into local services are rejected. DNS addresses
are validated and pinned for the connection while TLS certificates are checked.
Scripts and embedded content are discarded; JavaScript-heavy, login-only or
bot-blocked pages may not be readable.

Answers use source links returned by the tools. An unrecognized citation triggers
one correction attempt, and is refused if still unverified. Links are also
appended if the model omits citations. If no live source was retrieved, the chat
reports verification failure. This checks that cited URLs were retrieved;
it does not guarantee every model interpretation is correct.

Web text cannot authorize workstation changes. After web tools are used in a
turn, memory writes, note changes and undo are blocked for that turn. Request
those local changes in a separate message. HUD memory and confirmation controls
remain available. Network access is required only for web tools and initial
installation/model downloads.

## Verification and maintenance

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover schema conversion without changing data, local-only URLs,
tool-result conversation handling, note undo, rejected path/symlink writes,
unknown tools, persistent memory restoration, and human-only confirmation. Live checks also exercised text
chat and system status against Ollama, offscreen HUD startup without a Gemini
SDK/key, and eSpeak audio transcribed by offline Whisper. A short offline speaker playback completed successfully. Physical microphone
recognition and perceived voice quality still need testing in your desktop session.
The final Qwen2.5 7B live sequence passed chat, status, temporary note creation
and undo in approximately 2–4 seconds per reply, with about 16 GiB RAM still
available. The expanded suite has 19 passing automated tests, including public
URL validation, redirects, byte/type limits, process deadlines, citations,
search failure handling and blocked local mutations during web research.

On October 9, 2026, the live web test searched DuckDuckGo, read the Python.org
release page (following its redirect), and correctly answered Python 3.15.0
with the retrieved source URL. The search index still listed an older version,
so reading the live page was necessary. The local verification transcript is
stored in `.local-data/web-verification.json`.

To reinstall dependencies inside this project:

```bash
UV_CACHE_DIR="$PWD/.tools/uv-cache" .tools/uv pip install \
  --python .venv/bin/python -r requirements-local.lock
```

To download the speech model again (internet needed only for this step):

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base', local_dir='.local-data/whisper-base')"
```

With Ollama running, pull the text model using
`.tools/ollama/bin/ollama pull qwen2.5:7b`. Use `--model NAME` to select another
installed model that advertises tool calling. Installation assets and models
are ignored by Git. See [Ollama's manual Linux installation](https://docs.ollama.com/linux),
[hardware support](https://docs.ollama.com/gpu), and
[tool-calling API](https://docs.ollama.com/capabilities/tool-calling).
