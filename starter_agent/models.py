"""The one model function: complete(messages) -> text (plan section 2).

`messages` is a list of {"role": "system" | "user" | "assistant", "content": text}. Every part of the
agent calls a model only through `.complete`, so no part depends on which backend you pick.

The backends talk plain HTTP with Python's own urllib, so nothing needs installing:
- FakeModel: scripted and deterministic. The tests use it; with no script it runs the demo below.
- Ollama: a local model at http://localhost:11434 (https://ollama.com).
- OpenAICompatible: any server that speaks the OpenAI chat API (OpenAI, vLLM, llama.cpp, LM Studio...).
- Anthropic: Claude, with your own key in an environment variable.
"""
import http.client
import ipaddress
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


class ModelError(Exception):
    """The model's server answered with something the agent can't use, or the prompt can't be sent."""
    status, said = None, ""          # for an HTTP error: its status, and everything the server said


class PromptTooLong(ModelError):
    """The prompt is too long for the model's context: refused before it was sent (it must fit with a
    quarter to spare), or refused by the server itself. Never cut."""


class FakeModel:
    """Give it a list of replies (used in order) or a function of the messages."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = demo_brain if fn is None and replies is None else fn
        self.calls = []                       # every list of messages it was sent, for tests to read

    def complete(self, messages):
        self.calls.append(messages)
        return self.fn(messages) if self.fn else self.replies.pop(0)


def handed_beliefs(prompt):
    """The (id, text) of each current belief the demo was handed (the block between the two headings)."""
    block = prompt.partition("YOUR CURRENT BELIEFS:")[2].partition("YOUR CURRENT INTENTION:")[0]
    return [(int(i), t) for i, t in re.findall(r"^#(\d+) \(confidence [0-9.]+[^)]*\): (.*)$", block, re.M)]


def same_subject(beliefs, text):
    """The id of the newest current belief sharing at least half the words of the shorter text, else None.
    A crude rule, on purpose: it is only here so the demo can show a belief being replaced."""
    words = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    mine = words(text)
    for belief_id, old in reversed(beliefs):
        theirs = words(old)
        if mine and theirs and len(mine & theirs) * 2 >= min(len(mine), len(theirs)):
            return belief_id
    return None


def demo_brain(messages):
    """The no-key demo: a 'model' made of four rules, so you can watch every part work.
    - a message starting "remember:" becomes a belief, citing your turn as its evidence, with a fixed
      confidence of 0.60 (the demo never raises or lowers it; a real model sets its own);
    - a second "remember:" sharing at least half its words with a current belief replaces that belief;
    - a message ending in "?" gets "I'm not sure about that yet", which curiosity will notice;
    - at the end of a session, curiosity quotes that phrase back as an open question."""
    system, prompt = messages[0]["content"], messages[-1]["content"]
    if '"questions"' in system:               # the curiosity call
        found = re.findall(r"\[turn (\d+)\] agent: You said: (.*?) I'm not sure about that yet\.", prompt)
        return json.dumps({"questions": [
            {"text": said, "quote": "I'm not sure about that yet", "turn": int(turn)} for turn, said in found]})
    earlier, _, now = prompt.partition("THIS SESSION SO FAR:")
    turn, said = re.findall(r"\[turn (\d+)\] user: (.*)", now)[-1]
    reply = f"You said: {said}"
    if said.rstrip().endswith("?"):
        reply += " I'm not sure about that yet."
    recalled = re.findall(r"=== Session (\d+)", earlier)
    if recalled:
        reply += f" (I recalled session {', '.join(recalled)}.)"
    beliefs = []
    if said.lower().startswith("remember:"):
        text = said[len("remember:"):].strip()
        belief = {"text": text, "confidence": 0.6, "evidence": [int(turn)]}
        old = same_subject(handed_beliefs(prompt), text)
        if old is not None:
            belief["supersedes"] = old
        beliefs.append(belief)
    return reply + "\n```json\n" + json.dumps({"beliefs": beliefs, "intention_done": []}) + "\n```"


# --- The context window: one number, read by the request and by recall's budget (Nexus) ---------------

CONTEXT = {"fake": 12288, "ollama": 12288, "openai": 16384, "anthropic": 1_000_000}   # if the server can't say
MOST_ANSWER = 16000      # the usual ceiling for an answer that isn't streamed (see Anthropic.complete)


def context_of(config, model=None):
    """The model's context window in tokens. "context_tokens" in agent.json always wins; then the window the
    model's server says it has (model.window(), asked once per run); then the backend's usual one. Ollama
    is asked for exactly this window (num_ctx), and recall's budget is sized from the same number."""
    if config.context_tokens:
        return config.context_tokens
    window = getattr(model, "window", None)
    return (window() if window else None) or CONTEXT.get(config.backend, 12288)


def known_window(config):
    """The window where it is known without asking a server (`recall WORDS` asks none): "context_tokens" if you
    set it; else the fake model's, or the one Kindling asks Ollama for. None where only the server can say."""
    return context_of(config) if config.context_tokens or config.backend in ("fake", "ollama") else None


def reported_window(config, model):
    """True when the window in use is the one the model's server said it has (not agent.json's, not the one
    Kindling asks Ollama for, not one it assumed): then raising context_tokens past it is wrong advice, and a
    bigger window has to start at the server."""
    reported = getattr(model, "reported", None)
    return not config.context_tokens and bool(reported and reported())


def assumed(backend):
    """The backend's usual window, for a server that didn't say what its window is. Said once per run, and
    kept as a risk of a cut: the server's real window may be smaller."""
    note_once(f"Note: the server didn't say how long its context window is, so Kindling is assuming a "
              f"{CONTEXT[backend]:,}-token window. A server with a smaller one may cut a longer prompt without "
              "saying so: set context_tokens in agent.json to its real window.")
    at_risk("Kindling had to assume this server's window: set context_tokens in agent.json to its real one")
    return CONTEXT[backend]


def estimate_tokens(text):
    """Tokens, roughly: a digit or a non-ASCII character costs a token, ASCII punctuation half of one,
    everything else four characters a token (measured 2026-11-20 on Mistral 7B v0.3 and Qwen3.5). Nexus's
    estimate. On the eight texts measured that day (prose, code, Spanish, mixed languages, JSON, numbers,
    Japanese) real tokens were at most 1.18 times it: it erred high on Qwen3.5 and up to 18% low on Mistral
    (JSON), inside the quarter that check_fits keeps spare. It doesn't bound text built to defeat it, or a
    script the tokenizer lacks (rare Unicode it splits into bytes, say), and no tokenizer is needed."""
    digits = sum(c.isdigit() for c in text)
    wide = sum(ord(c) > 127 for c in text)
    punct = sum(c.isascii() and not c.isalnum() and not c.isspace() for c in text)
    return (len(text) - digits - wide - punct + 3) // 4 + digits + wide + (punct + 1) // 2


def check_fits(messages, context, reported=False):
    """Refuse, loudly, a prompt too long for the context, before it's sent: a server may cut it without
    telling the model. The estimate is only an estimate, so the prompt must fit with a quarter to spare.
    `reported`: the context is the window the server said it has (reported_window), so a bigger one has to
    start at the server, as curiosity's note says; raising context_tokens past it is wrong advice."""
    size = prompt_size(messages)
    if size * 5 // 4 > context:
        bigger = "start the server with a bigger window" if reported else 'raise "context_tokens" in agent.json'
        raise PromptTooLong(f"The prompt is about {size:,} tokens: too long for the model's {context:,}-token "
                         "context, so it wasn't sent (it would be cut without the model knowing). End this "
                         f"session (/end, or `session end`), or {bigger}")


def prompt_size(messages):
    return estimate_tokens("".join(m["content"] for m in messages))


def answer_room(window, messages):
    """The most an answer may take where a longer one would shift the start of the prompt out of the window in
    silence: the window less the prompt's estimate with check_fits's quarter to spare (rounded up), at least 1
    (0 or less would mean no limit to Ollama). Within the ratios measured (real tokens up to 1.18 times the
    estimate), the prompt and the whole answer then stay inside the window."""
    return max(window - (prompt_size(messages) * 5 + 3) // 4, 1)


def compare_count(server, messages, counted, cuts=True):
    """After a call: the server's own count of the prompt (None if it gave none) against the estimate.
    - Over it by more than a quarter, the estimate undercounted: the side that can let a prompt too long for
      the window through. That is printed, each time.
    - Under it by more than a quarter is usual with Ollama's prompt cache (it counts only what it didn't
      reuse), but it is also what a cut looks like: a prompt Ollama cut counted about half its window (Nexus,
      20 Nov). So it goes in the log only (take_counts), unless this run has a known risk of a cut (an
      assumed window; an Ollama older than 0.32.1, or whose version can't be read): then it is also printed,
      once, as a possible cut. Never for a server that refuses a prompt too long and never cuts one (cuts=False:
      Anthropic)."""
    guessed = prompt_size(messages)
    if counted is None or abs(counted - guessed) * 4 <= guessed:
        return
    said = f"{server} counted {counted:,} prompt tokens; Kindling estimated {guessed:,}"
    if counted > guessed:
        # A server that refuses a too-long prompt (cuts=False: Anthropic) can't cut one in silence.
        risk = ("The estimate undercounts your text, so the length check may let through a prompt too long for "
                "the window." if cuts else
                "The estimate undercounts your text, but this server refuses a prompt that is too long rather "
                "than cutting it, so nothing is lost in silence.")
        print(f"Note: {said}, over a quarter more. {risk}", file=sys.stderr)
        return
    COUNTS.append(f"{said}, over a quarter fewer")
    if RISKS and cuts:
        note_once(f"Note: {said}, over a quarter fewer, which can mean it cut the prompt without saying so. "
                  + ". ".join(RISKS) + ".", key="possible cut")


# --- Talking to a server -----------------------------------------------------------------------------

def is_local(url):
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


OPENERS = {}     # urllib's openers, built once per process: {whether the address is local: its opener}


def opener_for(url):
    """urllib's opener for this address, built once per process: building one loads the system's certificates,
    about 150 ms of CPU each time (r7 fixer, measured 21 Nov). A local address gets its own, which never goes
    through a proxy set in your environment (http_proxy), so a local model stays on your machine. The proxy
    settings are read when the opener is built, so once per run."""
    local = is_local(url)
    if local not in OPENERS:
        OPENERS[local] = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if local else []))
    return OPENERS[local]


def fetch_json(url, body=None, headers=None, timeout=600):
    """POST `body` as JSON (with no body, GET); return the reply's JSON. A local address never goes through
    a proxy set in your environment (http_proxy), so a local model stays on your machine (opener_for)."""
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    opener = opener_for(url)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:              # reached, but it said no: show what it said
        said = error.read().decode("utf-8", "replace")
        failed = ModelError(f"The model server answered HTTP {error.code}: {said[:300]}")
        failed.status, failed.said = error.code, said
        raise failed from None
    except http.client.HTTPException as error:          # e.g. the reply was cut off part way
        raise ModelError(f"The model server's reply broke off: {error!r}") from None
    try:
        return json.loads(raw)
    except ValueError:
        start = raw[:200].decode("utf-8", "replace")
        raise ModelError(f"The model server's reply wasn't JSON: {start!r}") from None


ASKED = {}       # what each server said about itself, asked once per process: {address: its JSON, or None}
TOLD = set()     # the notes already printed in this process
RISKS = []       # why the server may cut a prompt without saying so, in this process (what to do about it)
COUNTS = []      # the server's counts far under the estimate, for the agent's log (take_counts)


def ask_once(url, headers=None):
    """GET what a server says about itself (its version, its window), once per process. A server that
    can't say, for any reason, gives None: what it says is optional, never a reason to stop."""
    if url not in ASKED:
        try:
            ASKED[url] = fetch_json(url, None, headers, timeout=30)
        except (ModelError, OSError, ValueError):
            ASKED[url] = None
    return ASKED[url]


def note_once(text, key=None):
    """A note for you on stderr, printed once per process (once per `key`, for a note whose numbers vary)."""
    key = key or text
    if key not in TOLD:
        TOLD.add(key)
        print(text, file=sys.stderr)


def at_risk(reason):
    """Remember, for this process, a reason the server may cut a prompt without saying so."""
    if reason not in RISKS:
        RISKS.append(reason)


def take_counts():
    """The counts compare_count kept for the log since the last call, which it then forgets."""
    taken = COUNTS[:]
    COUNTS.clear()
    return taken


def text_of(reply):
    """The model's reply, if it is text."""
    if not isinstance(reply, str):
        raise ModelError(f"The model's reply wasn't text: {reply!r:.200}")
    return reply


def reply_at(out, *path):
    """The text at out[path[0]][path[1]]..., or a ModelError saying the reply wasn't the expected shape."""
    try:
        for step in path:
            out = out[step]
        return text_of(out)
    except (KeyError, IndexError, TypeError):
        raise ModelError(f"The model server's reply had no {'/'.join(map(str, path))}: {out!r:.200}") from None


def value_at(found, *path):
    """found[path[0]][path[1]]..., or None if it isn't there (whatever shape `found` has)."""
    try:
        for step in path:
            found = found[step]
    except (LookupError, TypeError):
        return None
    return found


def whole_number_at(found, *path):
    """found[path[0]][path[1]]..., if it is there and a whole number above 0; else None (it wasn't said)."""
    found = value_at(found, *path)
    return found if type(found) is int and found > 0 else None


STOPPED = []     # answers that stopped at their length limit since the last take_stopped (curiosity asks)


def stopped_short(room, then):
    """The server says the answer stopped at its length limit: say so, and what to do about it."""
    STOPPED.append(room)
    print(f"WARNING: the answer hit the room left for it ({room}) and stopped there, so it may end part way. "
          f"{then}.", file=sys.stderr)


def take_stopped():
    """Whether an answer stopped at its length limit since the last call, which then forgets it. Curiosity never
    takes such an answer as read (curiosity.cut_off)."""
    stopped = bool(STOPPED)
    STOPPED.clear()
    return stopped


def key_for(name, usual, base_url):
    """Your key, from the environment variable named in "api_key_env". With no name, the usual variable is
    read only when "base_url" is blank (the provider's own address): a key never goes to a server you didn't
    name it for, and never over plain http to another machine."""
    key = os.environ.get(name or usual, "") if name or not base_url else ""
    if key and base_url.startswith("http://") and not is_local(base_url):
        raise SystemExit(f"Refusing to send your key over plain http to {base_url}: use https.")
    return key


OLLAMA_TESTED = (0, 32, 1)       # the Ollama that "truncate": false was tested on (Nexus, 20 Nov 2026)


# vLLM's words for a prompt too long: "This model's maximum context length is N tokens. However, you requested
# M tokens ..." (newer: "... your request has M input tokens"), at the top level or under "error".
VLLM_TOO_LONG = re.compile(r"maximum context length is (\d+) tokens\b.*?(?:you requested|your request has) (\d+)",
                           re.S)
# How far into a server's error body a pattern looks: a refusal says its words first, and VLLM_TOO_LONG run over
# a whole body (fetch_json reads all of it) is slow on one built for that: 23.7 s at 140 KB, over 120 s at 280 KB
# (r6 verifier, measured 21 Nov). On the first 2,000 characters of the same body it took 2 ms.
SEARCHED = 2000


def refusal(server, error, bigger):
    """A server's refusal of a prompt longer than its window, as PromptTooLong quoting the server's own count;
    None for any other error. Ollama refuses because Kindling asks it to ("truncate": false); llama.cpp's own
    server refuses one too, with the numbers under "error"; vLLM says it in words (VLLM_TOO_LONG). It is one
    only if it is an HTTP 400 that names exceed_context_size_error, gives both counts (n_prompt_tokens and
    n_ctx), or says vLLM's words with both numbers in its first SEARCHED characters. `bigger`: how to give the
    model a bigger window."""
    said = parse_json(error.said)
    said = said if isinstance(said, dict) else {}
    nested = said["error"] if isinstance(said.get("error"), dict) else {}
    counts = next((found for found in (said, nested) if {"n_prompt_tokens", "n_ctx"} <= found.keys()), None)
    vllm = VLLM_TOO_LONG.search(error.said[:SEARCHED])
    if error.status != 400 or not (counts or vllm or "exceed_context_size_error" in error.said):
        return None
    what = (f"{counts['n_prompt_tokens']} tokens by its own count, for its {counts['n_ctx']}-token window" if counts
            else f"{vllm[2]} tokens by its own count, for its {vllm[1]}-token window" if vllm else error.said[:300])
    return PromptTooLong(f"{server} refused the prompt as too long: {what}. Nothing was cut or answered. End this "
                         f"session (/end, or `session end`), or {bigger}")


class Ollama:
    def __init__(self, model="", base_url="", num_ctx=CONTEXT["ollama"]):
        self.model = model or "llama3.1:8b"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx    # Ollama's own default window is smaller than an 8,000-token recall pack

    def window(self):
        return self.num_ctx       # the window Kindling asks Ollama for

    def complete(self, messages):
        self.check_version()
        # The answer gets what the prompt leaves of the window, at most. An answer that outgrows the window
        # makes Ollama shift out the start of the prompt (the instructions) mid-answer, saying nothing (Nexus,
        # 20 Nov). The estimate can count low (real tokens up to 1.18 times it, measured), so the room keeps
        # check_fits's quarter spare (answer_room): a long answer stops instead, and "length" says so.
        room = answer_room(self.num_ctx, messages)
        try:
            out = fetch_json(f"{self.base_url}/api/chat", {
                "model": self.model, "messages": messages, "stream": False,
                "truncate": False,              # too long for the window: refused (HTTP 400), never cut
                "options": {"num_ctx": self.num_ctx, "num_predict": room}})
        except ModelError as error:
            raise refusal("Ollama", error, 'raise "context_tokens" in agent.json') or error
        used = whole_number_at(out, "prompt_eval_count")
        compare_count("Ollama", messages, used)
        if used and used >= self.num_ctx * 0.95:          # it fit, leaving little room to answer
            print(f"WARNING: Ollama counted {used:,} prompt tokens in its {self.num_ctx:,}-token window: the "
                  "prompt left almost no room for the reply, which may stop short. End this session, or raise "
                  '"context_tokens" in agent.json.', file=sys.stderr)
        if value_at(out, "done_reason") == "length":
            stopped_short(f"{room:,} tokens: what the prompt left of the {self.num_ctx:,}-token window, with a "
                          "quarter of the prompt to spare", 'End this session, or raise "context_tokens" in agent.json')
        return reply_at(out, "message", "content")

    def check_version(self):
        """Once per process: say plainly if this Ollama is older than the one "truncate": false was tested on
        (0.32.1), or won't say which it is. An older one may ignore the field and cut a long prompt, so either
        is kept as a risk of a cut (one that won't say counts as one: the safe side)."""
        said = ask_once(f"{self.base_url}/api/version")
        version = said.get("version") if isinstance(said, dict) else None
        parts = tuple(int(n) for n in re.findall(r"\d+", version)[:3]) if isinstance(version, str) else ()
        risk = ('may ignore "truncate": false and cut a prompt too long for its window without saying so; '
                "Kindling's own length check still runs first.")
        if not parts:
            note_once(f"Note: Kindling couldn't read this Ollama's version (it was tested with 0.32.1). An older "
                      f"Ollama {risk}")
            at_risk("Kindling couldn't read this Ollama's version, and one older than 0.32.1 may ignore "
                    '"truncate": false: check it (`ollama --version`), and update it if it is older')
        elif parts < OLLAMA_TESTED:
            note_once(f"WARNING: this Ollama is {version}, older than 0.32.1, the version Kindling was tested "
                      f"with. It {risk}")
            at_risk(f'This Ollama ({version}) is older than 0.32.1 and may ignore "truncate": false: update it')


class OpenAICompatible:
    def __init__(self, model="", base_url="", api_key_env="", context_tokens=0):
        if not model:
            raise SystemExit('The openai backend needs "model" in agent.json.')
        self.key = key_for(api_key_env, "OPENAI_API_KEY", base_url)
        if not base_url and not self.key:      # nothing would stop it going to OpenAI, without a key
            raise SystemExit('The openai backend with no "base_url" sends to OpenAI: set "base_url" for a '
                             "local server, or put your OpenAI key in OPENAI_API_KEY.")
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        self.context_tokens = context_tokens   # agent.json's window, if you set one: it wins

    def window(self):
        """The server's window, where it says; otherwise the usual 16,384, said once."""
        return self.reported() or assumed("openai")

    def reported(self):
        """The window the server says it has, or None."""
        return self.server()[1]

    def server(self):
        """(which server it says it is, its window), or (None, None) if it doesn't say: vLLM lists each model's
        max_model_len on /v1/models, and llama.cpp gives its n_ctx on /props, at its root. Each is asked once per
        run (even with context_tokens set, which still wins for the window); each is optional."""
        listed = ask_once(f"{self.base_url}/models", self.headers)
        entries = listed.get("data") if isinstance(listed, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("id") == self.model and whole_number_at(entry, "max_model_len"):
                return "vLLM", entry["max_model_len"]
        props = ask_once(re.sub(r"/v1$", "", self.base_url) + "/props", self.headers)
        found = [whole_number_at(props, "n_ctx"), whole_number_at(props, "default_generation_settings", "n_ctx")]
        window = min([n for n in found if n], default=None)      # both said: the smaller, the safe side
        return ("llama.cpp", window) if window else (None, None)

    def room(self, messages):
        """The most the answer may take (max_tokens), or None to send none. Only a server that shifts the start of
        the prompt out in silence when an answer outgrows its window gets one; of the OpenAI-compatible servers
        Kindling can tell apart, that is llama.cpp (with context shift on). It gets answer_room of its window
        (context_tokens in agent.json if you set it, else its own), at most MOST_ANSWER. Every other server gets
        none, because there a cap does harm: vLLM refuses a prompt plus a cap longer than its window rather than
        shift, OpenAI's reasoning models reject max_tokens, and other providers' answer limits vary."""
        kind, window = self.server()
        if kind != "llama.cpp":
            return None
        return min(answer_room(self.context_tokens or window, messages), MOST_ANSWER)

    def complete(self, messages):
        room = self.room(messages)
        try:
            out = fetch_json(f"{self.base_url}/chat/completions",
                             {"model": self.model, "messages": messages, **({"max_tokens": room} if room else {})},
                             self.headers)
        except ModelError as error:
            raise refusal("The server", error, "start the server with a bigger window") or error
        compare_count("The server", messages, whole_number_at(out, "usage", "prompt_tokens"))
        if value_at(out, "choices", 0, "finish_reason") == "length":
            if room:
                stopped_short(f"{room:,} tokens: what the prompt left of llama.cpp's window, with a quarter of the "
                              f"prompt to spare, at most {MOST_ANSWER:,}", "End this session, or give the model a "
                              "bigger window: the server's own (llama.cpp's --ctx-size), and \"context_tokens\" in "
                              "agent.json if you set it")
            else:
                stopped_short("the server's own limit: Kindling sends this server no limit of its own", "End this "
                              "session, or give the model more room at the server: a bigger window, or a higher "
                              "limit for an answer if it has one")
        return reply_at(out, "choices", 0, "message", "content")


class Anthropic:
    def __init__(self, model="", base_url="", api_key_env=""):
        self.key = key_for(api_key_env, "ANTHROPIC_API_KEY", base_url)
        if not self.key:
            raise SystemExit("The anthropic backend reads your key from the environment variable named in "
                             '"api_key_env" (ANTHROPIC_API_KEY when "base_url" is blank).')
        self.model = model or "claude-opus-5-5"
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.headers = {"x-api-key": self.key, "anthropic-version": "2026-10-21",
                        "anthropic-beta": "server-side-fallback-2026-07-01"}       # for "fallbacks", below

    def window(self):
        """The model's window, where Anthropic says; otherwise the usual 1,000,000, said once."""
        return self.reported() or assumed("anthropic")

    def reported(self):
        """Anthropic says: GET /v1/models/{model} gives max_input_tokens, the model's context window (None if it
        can't say). Asked once per run, with a message's key and version but not its beta header, which is for
        "fallbacks" on a message."""
        asked = {name: value for name, value in self.headers.items() if name != "anthropic-beta"}
        said = ask_once(f"{self.base_url}/v1/models/{urllib.parse.quote(self.model, safe='')}", asked)
        return whole_number_at(said, "max_input_tokens")

    def complete(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        out = fetch_json(f"{self.base_url}/v1/messages",
                        # max_tokens is required. A reply here is ~300 tokens (plan section 7), but
                        # current Claude models think first and thinking counts against it; ~16,000 is
                        # the usual ceiling for a request that isn't streamed.
                        # "fallbacks": if a safety check declines a harmless message, Anthropic retries it
                        # on the model it recommends instead of failing. Delete it and the beta header
                        # (self.headers) if your model or provider doesn't accept them.
                        {"model": self.model, "max_tokens": MOST_ANSWER, "system": system, "messages": rest,
                         "fallbacks": "default"}, self.headers)
        # Anthropic refuses a prompt too long for the window (HTTP 400) and never cuts one: no possible-cut note.
        compare_count("Anthropic", messages, whole_number_at(out, "usage", "input_tokens"), cuts=False)
        if isinstance(out, dict) and out.get("stop_reason") == "refusal":     # declined: there's no reply
            return "(The model declined to answer this message.)"
        if value_at(out, "stop_reason") == "max_tokens":
            stopped_short(f"{MOST_ANSWER:,} tokens, the most Kindling asks Anthropic for, thinking included",
                          "Ask for a shorter answer (MOST_ANSWER in models.py sets this limit)")
        blocks = out.get("content") if isinstance(out, dict) else None
        if not isinstance(blocks, list):
            raise ModelError(f"The model server's reply had no content: {out!r:.200}")
        return "".join(text_of(b.get("text")) for b in blocks if isinstance(b, dict) and b.get("type") == "text")


def make_model(config):
    if config.backend == "fake":
        return FakeModel()
    if config.backend == "ollama":
        return Ollama(config.model, config.base_url, context_of(config))
    if config.backend == "openai":
        return OpenAICompatible(config.model, config.base_url, config.api_key_env, config.context_tokens)
    if config.backend == "anthropic":
        return Anthropic(config.model, config.base_url, config.api_key_env)
    raise SystemExit(f'Unknown backend "{config.backend}": use fake, ollama, openai or anthropic.')


# --- Reading what the model wrote --------------------------------------------------------------------

CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text):
    """Text as it's stored and shown: valid UTF-8 (a broken character becomes ?), and none of the terminal
    control characters that could hide or rewrite what your screen shows. Newlines and tabs stay."""
    return CONTROL.sub("", text.encode("utf-8", "replace").decode("utf-8"))


def one_line(text):
    """A belief or a question: clean and on one line, so it can't pose as another part of the prompt."""
    return " ".join(clean(text).split())


def split_json(text, keys=("beliefs", "intention_done")):
    """Split a model's output into (the prose, its last fenced JSON block as an object), or (text, None).
    The block counts only if it is an object whose keys are all in `keys`, so an example block you asked
    for stays in the reply. The fence may be ```json, ```JSON or bare ```. Found with str.rfind, not a
    regular expression, so any reply is split in linear time."""
    close = text.rfind("```")
    start = text.rfind("```", 0, close) if close > 0 else -1
    if start < 0:
        return text.strip(), None
    inner = text[start + 3:close]
    found = parse_json(inner[4:] if inner[:4].lower() == "json" else inner)
    if not isinstance(found, dict) or not found or set(found) - set(keys):
        return text.strip(), None
    return (text[:start] + text[close + 3:]).strip(), found


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
