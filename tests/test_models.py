"""The model backends, offline: what each one sends and where, what it does with a reply it can't use, and
how the agent reads what a model wrote. A fake urlopen stands in for every server; no request leaves."""
import json
import re
import signal
import socket
import time
import urllib.error
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import Clock, add_session, only
from starter_agent import models
from starter_agent.agent import Agent
from starter_agent.config import Config
from starter_agent.embed import HashEmbedder
from starter_agent.models import (Anthropic, ModelError, Ollama, OpenAICompatible, PromptTooLong, check_fits,
                                  estimate_tokens, split_json)
from starter_agent.recall import Recalled
from starter_agent.store import Store

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
OLLAMA_VERSION = "http://localhost:11434/api/version"
TOO_LONG = {"error": "request (1757 tokens) exceeds the available context size (256 tokens), try increasing it",
            "type": "exceed_context_size_error", "n_prompt_tokens": 1757, "n_ctx": 256}   # Ollama 0.32.1's (Nexus)


def outcome(call, *args):
    """What a call did: its result, or the exception it raised, so a crash fails by assertion."""
    try:
        return call(*args)
    except (Exception, SystemExit) as error:
        return error


class Answer(BytesIO):
    """A scripted reply, as urllib hands one back."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def wire(monkeypatch):
    """Stands in for every server at urllib's own door, so no request leaves this computer. Each POST is
    recorded in .sent as (url, body, headers) and answered with .reply, or refused with .refusal = (HTTP
    status, body), or by .refusal(body) when it is a function (None: not refused). Each GET is recorded in
    .asked as (url, headers) and answered with .facts[url] (raised, if it is an exception), or with HTTP 404
    if there's none. Every method is recorded in .methods, read from the request itself: a GET carries no
    body and a POST always does. Header names are lower case."""
    line = SimpleNamespace(reply=None, refusal=None, facts={}, sent=[], asked=[], methods=[])

    def open_(opener, request, timeout=None):
        url, headers = request.full_url, {k.lower(): v for k, v in request.header_items()}
        method = request.get_method()
        line.methods.append(method)
        if method == "GET":
            assert request.data is None, f"a GET with a body: {url}"
            line.asked.append((url, headers))
            if url not in line.facts:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, BytesIO(b"404 page not found"))
            if isinstance(line.facts[url], BaseException):
                raise line.facts[url]
            return Answer(json.dumps(line.facts[url]).encode())
        assert method == "POST" and request.data is not None, f"a {method} with no body: {url}"
        line.sent.append((url, json.loads(request.data), headers))
        refused = line.refusal(line.sent[-1][1]) if callable(line.refusal) else line.refusal
        if refused:
            status, body = refused
            raise urllib.error.HTTPError(url, status, "Refused", {}, BytesIO(json.dumps(body).encode()))
        return Answer(json.dumps(line.reply).encode())
    monkeypatch.setattr("urllib.request.OpenerDirector.open", open_)
    return line


# --- What each backend sends, and where (army test-engineer, r3) ----------------------------------------

def test_ollama_request_and_reply(wire):
    wire.reply = {"message": {"role": "assistant", "content": "hello"}, "prompt_eval_count": 20}
    assert Ollama().complete(MESSAGES) == "hello"
    ((url, body, _),) = wire.sent
    assert url == "http://localhost:11434/api/chat" and body["messages"] == MESSAGES and body["stream"] is False
    assert body.get("truncate") is False                             # too long: refuse, never cut
    assert body["options"]["num_ctx"] == models.context_of(Config(backend="ollama")) == 12288


def test_one_context_number_for_the_request_and_the_budget(wire):
    """Nexus: the window Ollama is asked for is the number recall's budget is sized from."""
    wire.reply = {"message": {"content": "ok"}}
    config = Config(backend="ollama", context_tokens=4096)
    models.make_model(config).complete(MESSAGES)
    assert wire.sent[0][1]["options"]["num_ctx"] == models.context_of(config) == 4096


# --- Ollama refuses a prompt too long for its window instead of cutting it (Nexus, 20 Nov) -----------------

def test_ollamas_count_near_its_window_means_little_room_for_the_reply_never_a_cut(wire, capsys):
    """Nexus measured a cut prompt reporting LESS, about half the window, so a count near the window only
    means the prompt fit and left little room to answer."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    near = [{"role": "user", "content": "x" * 16000}]                # estimated 4,000 tokens
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 3900}
    assert Ollama(num_ctx=4096).complete(near) == "ok"
    err = capsys.readouterr().err
    assert "almost no room for the reply" in err and "cut" not in err
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 3800}    # under 95%: nothing to say
    assert Ollama(num_ctx=4096).complete(near) == "ok"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("refusal", [TOO_LONG, {"error": "too long", "n_prompt_tokens": 1757, "n_ctx": 256}],
                         ids=["as Ollama 0.32.1 said it", "the numbers alone"])
def test_ollamas_400_for_a_prompt_too_long_is_prompt_too_long_in_its_own_numbers(wire, refusal):
    wire.refusal = (400, refusal)
    error = outcome(Ollama(num_ctx=256).complete, MESSAGES)
    assert isinstance(error, PromptTooLong), error
    assert "1757 tokens by its own count, for its 256-token window" in str(error) and "/end" in str(error)


def test_ollamas_error_type_alone_makes_it_prompt_too_long(wire):
    wire.refusal = (400, {"error": "the prompt is too long", "type": "exceed_context_size_error"})
    error = outcome(Ollama().complete, MESSAGES)
    assert isinstance(error, PromptTooLong) and "exceed_context_size_error" in str(error), error


@pytest.mark.parametrize("status, body", [(400, {"error": "invalid option: num_ctx"}), (500, TOO_LONG),
                                          (404, {"error": 'model "llama3.1:8b" not found, try pulling it first'})])
def test_any_other_refusal_stays_a_plain_model_error(wire, status, body):
    wire.refusal = (status, body)
    error = outcome(Ollama().complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error
    assert f"HTTP {status}" in str(error)


def test_ollamas_refusal_mid_chat_keeps_nothing_and_says_why(wire, tmp_path):
    config = Config(db=str(tmp_path / "agent.db"), backend="ollama", context_tokens=4000)
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    wire.refusal = (400, {**TOO_LONG, "n_prompt_tokens": 4100, "n_ctx": 4000})
    error = outcome(agent.handle, "hello")
    assert isinstance(error, PromptTooLong) and "4100 tokens by its own count" in str(error), error
    assert agent.store.turns(1) == []                                  # your message wasn't kept: send it again


@pytest.mark.parametrize("facts, said", [
    ({"version": "0.32.1"}, None), ({"version": "0.40.0"}, None),     # the version tested, or newer: silent
    ({"version": "0.31.9"}, "this Ollama is 0.31.9, older than 0.32.1"),
    ({"version": "0.32.0"}, "this Ollama is 0.32.0, older than 0.32.1"),   # the patch number counts (r4 V3)
    ({"version": "0.5.7-rc1"}, "this Ollama is 0.5.7-rc1, older than 0.32.1"),
    (None, "couldn't read this Ollama's version"),                     # no /api/version: HTTP 404
    ({"version": 7}, "couldn't read this Ollama's version")])
def test_an_ollama_older_than_the_one_tested_is_named_once_per_run(wire, capsys, facts, said):
    if facts is not None:
        wire.facts[OLLAMA_VERSION] = facts
    wire.reply = {"message": {"content": "ok"}}
    ollama = Ollama()
    assert ollama.complete(MESSAGES) == ollama.complete(MESSAGES) == "ok"
    assert [url for url, _ in wire.asked] == [OLLAMA_VERSION]          # asked once in this run
    err = capsys.readouterr().err
    if said is None:
        assert err == ""
    else:
        assert said in err and err.count("0.32.1") == 1               # said once
        assert 'may ignore "truncate": false and cut a prompt too long' in err


# --- The window comes from the server where it can say; agent.json always wins (Nexus, 20 Nov) -------------

VLLM = "http://localhost:8000/v1"


def window_of(wire, facts, **settings):
    """The window the agent would use with these settings, when the servers say `facts` ({url: JSON})."""
    wire.facts.update(facts)
    config = Config(**{"backend": "openai", "model": "m", "base_url": VLLM, **settings})
    return models.context_of(config, models.make_model(config))


def test_vllm_says_its_window_on_v1_models(wire):
    listed = {"data": [{"id": "other", "max_model_len": 99999}, {"id": "m", "max_model_len": 4096}]}
    assert window_of(wire, {f"{VLLM}/models": listed}) == 4096
    assert [url for url, _ in wire.asked] == [f"{VLLM}/models"]                  # /props wasn't needed


@pytest.mark.parametrize("props", [{"n_ctx": 3072}, {"default_generation_settings": {"n_ctx": 3072}}])
def test_llama_cpp_says_its_window_on_props(wire, props):
    listed = {"data": [{"id": "m", "meta": {"n_ctx_train": 131072}}]}           # its training length, not its window
    assert window_of(wire, {f"{VLLM}/models": listed, "http://localhost:8000/props": props}) == 3072


@pytest.mark.parametrize("props, window", [({"n_ctx": 32768, "default_generation_settings": {"n_ctx": 8192}}, 8192),
                                           ({"n_ctx": 4096, "default_generation_settings": {"n_ctx": 8192}}, 4096)])
def test_props_reporting_both_windows_gives_the_smaller(wire, props, window):
    """The round-4 verifier's LOW: llama.cpp's per-slot window is the nested one; the smaller is the safe side."""
    assert window_of(wire, {"http://localhost:8000/props": props}) == window


@pytest.mark.parametrize("facts", [{}, {f"{VLLM}/models": {"data": [{"id": "other", "max_model_len": 4096}]}},
                                   {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": "4096"}]}},
                                   {f"{VLLM}/models": {"data": "m"}, "http://localhost:8000/props": [3072]},
                                   {"http://localhost:8000/props": {"n_ctx": 0}}])
def test_a_server_that_cant_say_gets_the_usual_window_said_once(wire, capsys, facts):
    assert window_of(wire, facts) == window_of(wire, {}) == 16384
    err = capsys.readouterr().err
    assert err.count("assuming a 16,384-token window. A server with a smaller one may cut a longer prompt without "
                     "saying so: set context_tokens in agent.json to its real window.") == 1
    assert models.RISKS == ["Kindling had to assume this server's window: set context_tokens in agent.json to its "
                            "real one"]                                 # kept once, as a risk of a cut
    assert [url for url, _ in wire.asked] == [f"{VLLM}/models", "http://localhost:8000/props"]   # once each


@pytest.mark.parametrize("name, facts", [
    ("openai", {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": True}]}}),
    ("openai", {"http://localhost:8000/props": {"n_ctx": True}}),
    ("anthropic", {"https://api.anthropic.com/v1/models/claude-opus-5-5": {"max_input_tokens": True}})],
    ids=["vLLM", "llama.cpp", "Anthropic"])
def test_a_window_reported_as_true_is_not_a_one_token_window(wire, monkeypatch, name, facts):
    """The round-5 verifier's V14: true is an int to Python, and a 1-token window would refuse every prompt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    settings = {"backend": "anthropic", "model": "", "base_url": ""} if name == "anthropic" else {}
    assert window_of(wire, facts, **settings) == models.CONTEXT[name]          # not said: the usual one, assumed


def test_anthropic_says_its_window_asked_with_a_messages_key_and_version_but_no_beta_header(wire, monkeypatch,
                                                                                            capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.facts["https://api.anthropic.com/v1/models/claude-opus-5-5"] = {"id": "claude-opus-5-5",
                                                                         "max_input_tokens": 200000}
    config = Config(backend="anthropic")
    model = models.make_model(config)
    assert models.context_of(config, model) == 200000
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    assert model.complete(MESSAGES) == "ok"
    ((_, asked),), ((_, _, sent),) = wire.asked, wire.sent
    assert "anthropic-beta" in sent and asked == {k: v for k, v in sent.items() if k != "anthropic-beta"}
    assert asked.get("x-api-key") == "test-key-not-real" and "anthropic-version" in asked
    assert "assuming" not in capsys.readouterr().err


def test_anthropic_that_cant_say_gets_the_usual_window_said_once(wire, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert window_of(wire, {}, backend="anthropic", model="", base_url="") == 1_000_000
    assert window_of(wire, {}, backend="anthropic", model="", base_url="") == 1_000_000
    assert capsys.readouterr().err.count("assuming a 1,000,000-token window. A server with a smaller one may cut") == 1


def test_ollamas_window_is_the_num_ctx_kindling_sends_and_needs_no_question(wire):
    config = Config(backend="ollama")
    model = models.make_model(config)
    assert models.context_of(config, model) == model.num_ctx == 12288 and wire.asked == []


@pytest.mark.parametrize("settings", [{}, {"backend": "anthropic", "base_url": ""}, {"backend": "ollama"}],
                         ids=["openai", "anthropic", "ollama"])
def test_context_tokens_in_agent_json_always_wins_and_nothing_is_asked(wire, monkeypatch, capsys, settings):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    says_4096 = {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 4096}]},
                 "http://localhost:8000/props": {"n_ctx": 4096},
                 "https://api.anthropic.com/v1/models/m": {"max_input_tokens": 4096}}
    assert window_of(wire, says_4096, context_tokens=5000, **settings) == 5000
    assert wire.asked == [] and capsys.readouterr().err == ""


def test_the_servers_window_sizes_the_length_check(wire, tmp_path):
    """llama.cpp started with a 2,048-token window: the usual 16,384 would let prompts eight times too big through."""
    wire.facts["http://localhost:8000/props"] = {"n_ctx": 2048}
    config = Config(db=str(tmp_path / "agent.db"), backend="openai", model="m", base_url=VLLM)
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    error = outcome(agent.handle, "a long paste " * 600)               # ~2,000 tokens: under 16,384, over 2,048
    assert isinstance(error, PromptTooLong) and "2,048-token" in str(error), error
    assert wire.sent == []                                             # refused before it was sent


# --- Round 5: a test for each mutant the round-4 verifier found surviving (V1 to V12; V9 was caught) ---------

def openai_agent(wire, tmp_path, facts, **settings):
    wire.facts.update(facts)
    config = Config(**{"db": str(tmp_path / "agent.db"), "backend": "openai", "model": "m", "base_url": VLLM,
                       **settings})
    return Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)


def test_recalls_budget_is_sized_from_the_servers_window(wire, tmp_path):
    """V7. vLLM at 4,096: a past session of ~3,000 tokens fits the 8,000 budget but not the room 4,096 leaves,
    so it is skipped and named and the turn goes through. Sized from the usual 16,384, it would be packed,
    and the whole turn refused."""
    agent = openai_agent(wire, tmp_path, {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 4096}]}})
    big, _ = add_session(agent.store, [("user", "tell me about mice " + "z" * 12000)])
    wire.reply = {"choices": [{"message": {"content": "ok"}}]}
    assert outcome(agent.handle, "mice?") == "ok"
    assert any(f"recall skipped session {big}" in note for note in agent.notes), agent.notes


def test_curiosity_never_sends_a_prompt_too_long_for_the_servers_window(wire, tmp_path):
    """V8. llama.cpp at 2,048: a session of ~3,000 tokens of agent turns fits the usual 16,384, not 2,048."""
    agent = openai_agent(wire, tmp_path, {"http://localhost:8000/props": {"n_ctx": 2048}})
    add_session(agent.store, [("user", "go"), ("agent", "word " * 400)] * 6)
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    sizes = [models.prompt_size(body["messages"]) for _, body, _ in wire.sent]
    assert len(sizes) == 1 and all(size * 5 // 4 <= 2048 for size in sizes), sizes      # read in part (Lyra)


@pytest.mark.parametrize("failure", [urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
                                     TimeoutError("timed out")], ids=["refused", "timed out"])
def test_a_server_that_cant_be_reached_for_a_question_gives_no_answer_never_a_crash(wire, failure):
    """V10: a question about the server (its version, its window) that can't connect is optional, like one
    the server can't answer; the turn goes on."""
    for url in (OLLAMA_VERSION, f"{VLLM}/models", "http://localhost:8000/props"):
        wire.facts[url] = failure
    wire.reply = {"message": {"content": "ok"}}
    assert outcome(Ollama().complete, MESSAGES) == "ok"
    assert outcome(window_of, wire, {}) == 16384


def test_every_question_to_a_server_is_a_get_and_every_message_a_post(wire):
    """V11: sent as a POST, each question would get HTTP 405 from a real server and quietly fall back."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "ok"}}
    assert outcome(Ollama().complete, MESSAGES) == "ok"
    assert outcome(window_of, wire, {"http://localhost:8000/props": {"n_ctx": 3072}}) == 3072
    assert wire.methods == ["GET", "POST", "GET", "GET"]               # version, chat, /models, /props


def test_the_length_check_counts_the_system_prompt_too():
    """V12: the instructions take room in the window like everything else."""
    long_system = [{"role": "system", "content": "x" * 4000}, {"role": "user", "content": "hi"}]   # ~1,000
    assert check_fits(long_system, 1300) is None
    assert isinstance(outcome(check_fits, long_system, 1000), PromptTooLong)


# --- After each call, the server's own count checks the estimate (r4 verifier's MEDIUM 1, CC's rule, 20 Nov) -----
# Over the estimate by more than a quarter (it undercounted: the dangerous side) is printed. Under it by more than
# a quarter is logged only: usual with Ollama's prompt cache, but also what a cut looks like (Nexus), so where a cut
# is possible (an assumed window, an Ollama older than 0.32.1) it is printed once, as a possible cut.

PROSE = [{"role": "user", "content": "x" * 4000}]                       # estimated 1,000 tokens
COUNTED = {"ollama": lambda n: {"message": {"content": "ok"}, "prompt_eval_count": n},
           "openai": lambda n: {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": n}},
           "anthropic": lambda n: {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": n}}}


def backend(wire, monkeypatch, name):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    return models.make_model(Config(backend=name, model="m", base_url=VLLM if name == "openai" else ""))


@pytest.mark.parametrize("name", COUNTED)
def test_a_count_far_over_the_estimate_is_printed_and_one_far_under_only_logged(wire, capsys, monkeypatch, name):
    """The two sides are the right way round (the round-4 verifier's V4), and neither says "not a cut"."""
    model = backend(wire, monkeypatch, name)
    for counted, printed, kept in ((1200, False, False), (800, False, False), (1300, True, False), (700, False, True)):
        wire.reply = COUNTED[name](counted)
        assert model.complete(PROSE) == "ok"
        err = capsys.readouterr().err
        assert ("counted" in err) is printed, (counted, err)
        kept_text = f"{err_name(name)} counted 700 prompt tokens; Kindling estimated 1,000, over a quarter fewer"
        assert models.take_counts() == ([kept_text] if kept else []), counted
        if printed:
            assert f"counted {counted:,} prompt tokens; Kindling estimated 1,000, over a quarter more" in err
            if name == "anthropic":      # it refuses a too-long prompt and never cuts one: say nothing is lost
                assert "refuses a prompt that is too long rather than cutting it" in err
                assert "may let through" not in err
            else:
                assert "the length check may let through a prompt too long for the window" in err
        assert "mean the prompt was cut" not in err and "not a cut" not in err


def err_name(name):
    return {"ollama": "Ollama", "openai": "The server", "anthropic": "Anthropic"}[name]


@pytest.mark.parametrize("name", COUNTED)
def test_a_count_far_over_the_estimate_is_printed_each_time_not_once_per_run(wire, capsys, monkeypatch, name):
    """The round-5 verifier's M1: CC's rule prints the side that can let a prompt too long through at every call."""
    model = backend(wire, monkeypatch, name)
    for counted in (1300, 1400, 1300):
        wire.reply = COUNTED[name](counted)
        assert model.complete(PROSE) == "ok"
    err = capsys.readouterr().err
    assert err.count("over a quarter more") == 3 and err.count("counted 1,300 prompt tokens") == 2, err


@pytest.mark.parametrize("risk, said", [
    ("an assumed window", "Kindling had to assume this server's window: set context_tokens in agent.json to its real one"),
    ("an Ollama older than 0.32.1", 'This Ollama (0.31.9) is older than 0.32.1 and may ignore "truncate": false'),
    ("an Ollama whose version can't be read", "Kindling couldn't read this Ollama's version, and one older than 0.32.1 "
                                              'may ignore "truncate": false: check it (`ollama --version`)')])
def test_a_count_far_under_the_estimate_is_printed_once_as_a_possible_cut_where_a_cut_is_possible(wire, capsys,
                                                                                                risk, said):
    """The round-4 verifier's path: Ollama's own /v1 says no window, Kindling assumes 16,384, and the server
    counts far fewer tokens than the estimate. That can be a cut, so it's said, once per run, and logged each time.
    An Ollama that won't say its version counts as a risk too: the safe side (CC's call, r5 open question)."""
    if risk == "an assumed window":
        config = Config(backend="openai", model="m", base_url=VLLM)
        model, reply = models.make_model(config), COUNTED["openai"]
        assert models.context_of(config, model) == 16384               # nothing says: assumed
    else:
        if risk == "an Ollama older than 0.32.1":
            wire.facts[OLLAMA_VERSION] = {"version": "0.31.9"}         # else no /api/version: HTTP 404
        model, reply = Ollama(), COUNTED["ollama"]
    for counted in (700, 600, 500):
        wire.reply = reply(counted)
        assert model.complete(PROSE) == "ok"
    err = capsys.readouterr().err
    assert err.count("which can mean it cut the prompt without saying so") == 1, err   # once per run
    assert "counted 700 prompt tokens; Kindling estimated 1,000, over a quarter fewer" in err
    (_, why), = [err.split("which can mean it cut the prompt without saying so")]
    assert said in why and len(models.RISKS) == 1, (why, models.RISKS)  # the one risk this run has, and what to do
    assert len(models.take_counts()) == 3                              # and every one is in the log


def test_without_a_known_risk_a_count_far_under_is_never_printed(wire, capsys):
    """An Ollama at 0.32.1 or later, whose window Kindling asks for, is no known risk: logged, not printed."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    model = Ollama()
    wire.reply = COUNTED["ollama"](500)
    assert model.complete(PROSE) == "ok"
    assert "counted" not in capsys.readouterr().err and len(models.take_counts()) == 1


def test_anthropic_which_refuses_and_never_cuts_is_never_said_to_have_cut(wire, capsys, monkeypatch):
    """CC's call (r5 open question): an assumed Anthropic window is kept as a risk, but Anthropic refuses a prompt
    too long (HTTP 400) and never cuts one, so a count far under the estimate is logged, never called a cut."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    config = Config(backend="anthropic")
    model = models.make_model(config)
    assert models.context_of(config, model) == 1_000_000 and models.RISKS          # it can't say: assumed
    capsys.readouterr()
    for counted in (700, 600):
        wire.reply = COUNTED["anthropic"](counted)
        assert model.complete(PROSE) == "ok"
    assert "cut" not in capsys.readouterr().err
    assert len(models.take_counts()) == 2                              # still in the log


def test_the_agent_and_curiosity_write_a_count_far_under_the_estimate_to_the_log(wire, tmp_path, capsys):
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    config = Config(db=str(tmp_path / "agent.db"), backend="ollama")
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 5}     # far under: a prompt cache, say
    assert agent.handle("hello") == "ok"
    wire.reply = {"message": {"content": '{"questions": []}'}, "prompt_eval_count": 6}
    agent.end_session()
    rows = agent.store.db.execute("SELECT session_id, reason FROM log WHERE kind='count' ORDER BY id").fetchall()
    assert [tuple(row) for row in rows] == [
        (1, f"Ollama counted 5 prompt tokens; Kindling estimated {models.prompt_size(wire.sent[0][1]['messages']):,}, "
            "over a quarter fewer"),
        (1, f"Ollama counted 6 prompt tokens; Kindling estimated {models.prompt_size(wire.sent[1][1]['messages']):,}, "
            "over a quarter fewer")]
    assert "counted" not in capsys.readouterr().err


@pytest.mark.parametrize("name", COUNTED)
@pytest.mark.parametrize("count", [None, "lots", 1.5, True, -3, 0, [1200], {"n": 1}])
def test_the_after_call_compare_never_raises_whatever_the_count(wire, capsys, monkeypatch, name, count):
    model = backend(wire, monkeypatch, name)
    wire.reply = COUNTED[name](count)
    assert outcome(model.complete, PROSE) == "ok"
    assert capsys.readouterr().err == ""                                # no count it can read: nothing to say


@pytest.mark.parametrize("usage", ["n/a", [1200], None, 7])
def test_the_after_call_compare_never_raises_whatever_the_usage(wire, monkeypatch, usage):
    for name, path in (("openai", "prompt_tokens"), ("anthropic", "input_tokens")):
        model = backend(wire, monkeypatch, name)
        wire.reply = {**COUNTED[name](1000), "usage": usage}
        assert outcome(model.complete, PROSE) == "ok", (name, path)


# --- The answer gets the room the prompt leaves, and a 'length' finish is said (Nexus, 20 Nov; CC's call) -----
# An answer that outgrows the window made Ollama shift the start of the prompt out mid-answer, silently.

def test_ollama_is_asked_for_an_answer_no_longer_than_the_room_the_prompt_leaves(wire):
    """With check_fits's quarter to spare (CC's call, r5 verifier's MEDIUM): num_ctx - ceil(estimate * 5 / 4)."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "ok"}, "done_reason": "stop"}
    assert Ollama(num_ctx=4096).complete(PROSE) == "ok"                        # PROSE: estimated 1,000 tokens
    assert wire.sent[0][1]["options"] == {"num_ctx": 4096, "num_predict": 4096 - 1250}
    assert Ollama(num_ctx=4096).complete([{"role": "user", "content": "x" * 4004}]) == "ok"     # estimated 1,001:
    assert wire.sent[1][1]["options"]["num_predict"] == 4096 - 1252            # 1,251.25 rounds up, never down
    assert Ollama(num_ctx=500).complete(PROSE) == "ok"                         # sent anyway (no check_fits here):
    assert wire.sent[2][1]["options"]["num_predict"] == 1                      # at least 1, never -1 (no limit)


@pytest.mark.parametrize("estimate, num_ctx", [(1000, 4096), (1001, 4096), (2345, 3200), (4560, 5740), (9000, 11300)])
def test_within_the_measured_ratios_the_prompt_and_the_whole_answer_fit_the_window(wire, estimate, num_ctx):
    """The r5 verifier's MEDIUM, reproduced: Mistral's real tokens ran up to 1.18 times the estimate (Nexus,
    20 Nov), and Ollama shifts the prompt's start out, silently, once the prompt plus the answer passes num_ctx.
    At the ratios measured, the answer's limit keeps the two inside the window (4,560 in 5,740 was the case)."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "ok"}}
    prompt = [{"role": "user", "content": "x" * (4 * estimate)}]
    assert models.prompt_size(prompt) == estimate and check_fits(prompt, num_ctx) is None    # it passes the check
    assert Ollama(num_ctx=num_ctx).complete(prompt) == "ok"
    num_predict = wire.sent[0][1]["options"]["num_predict"]
    for ratio in (1.0, 1.05, 1.10, 1.13, 1.16, 1.18):
        assert -(-estimate * ratio // 1) + num_predict <= num_ctx, (ratio, estimate, num_predict)


LLAMA_PROPS = "http://localhost:8000/props"


@pytest.mark.parametrize("settings, facts, max_tokens", [
    ({}, {LLAMA_PROPS: {"n_ctx": 4096}}, 4096 - 1250),                                 # llama.cpp's own window
    ({"context_tokens": 3000}, {LLAMA_PROPS: {"n_ctx": 4096}}, 3000 - 1250),          # agent.json's wins
    ({}, {LLAMA_PROPS: {"n_ctx": 100000}}, 16000),                                     # at most MOST_ANSWER
    ({}, {LLAMA_PROPS: {"n_ctx": 1000}}, 1)], ids=["its window", "agent.json wins", "at most 16,000", "at least 1"])
def test_llama_cpp_which_shifts_gets_max_tokens_the_room_left_with_a_quarter_to_spare(wire, settings, facts,
                                                                                    max_tokens):
    wire.facts.update(facts)
    config = Config(backend="openai", model="m", base_url=VLLM, **settings)
    model = models.make_model(config)
    models.context_of(config, model)                                           # as the agent asks first
    wire.reply = COUNTED["openai"](1000)
    assert model.complete(PROSE) == "ok"
    assert wire.sent[0][1].get("max_tokens") == max_tokens


VLLM_8192 = {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 8192}]}}


@pytest.mark.parametrize("base_url, settings, facts", [
    (VLLM, {}, VLLM_8192), (VLLM, {"context_tokens": 8192}, VLLM_8192),        # vLLM refuses; it never shifts
    ("", {}, {}), ("", {"context_tokens": 200000}, {}),                        # OpenAI: as the assumed note advises
    (VLLM, {}, {}), (VLLM, {"context_tokens": 4096}, {})],                     # a server that doesn't say what it is
    ids=["vLLM", "vLLM, context_tokens set", "api.openai.com", "api.openai.com, context_tokens set",
         "unknown kind", "unknown kind, context_tokens set"])
def test_no_max_tokens_to_vllm_openai_or_a_server_of_unknown_kind(wire, monkeypatch, base_url, settings, facts):
    """CC's call on the r5 verifier's HIGH and MEDIUM: for any server but one that shifts, a cap does harm."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    wire.facts.update(facts)
    config = Config(backend="openai", model="m", base_url=base_url, **settings)
    model = models.make_model(config)
    models.context_of(config, model)
    wire.reply = COUNTED["openai"](1000)
    assert model.complete(PROSE) == "ok"
    sent = wire.sent[0][1]
    assert "max_tokens" not in sent and "max_completion_tokens" not in sent, sent


def test_vllm_answers_every_turn_where_the_estimate_counts_low(wire, tmp_path):
    """The r5 verifier's HIGH, reproduced: vLLM refuses (HTTP 400) a request whose prompt plus max_tokens is
    over max_model_len. With real tokens 1.05 to 1.18 times the estimate, round 5 sent max_tokens = the window
    less the estimate, and every turn was refused. Now vLLM gets no max_tokens, and every turn is answered."""
    agent = openai_agent(wire, tmp_path, VLLM_8192)
    ratio = iter([1.05, 1.10, 1.18])

    def vllm(body):
        real = -(-models.prompt_size(body["messages"]) * next(ratio) // 1)
        asked = real + body.get("max_tokens", 0)
        return (400, {"object": "error", "code": 400, "message": "This model's maximum context length is 8192 tokens. "
                      f"However, you requested {asked} tokens."}) if asked > 8192 else None
    wire.refusal = vllm
    wire.reply = {"choices": [{"message": {"content": "ok"}}]}
    assert [outcome(agent.handle, f"turn {n}: " + "z" * 6000) for n in (1, 2, 3)] == ["ok", "ok", "ok"]


def test_ollamas_length_finish_says_the_answer_hit_the_room_left_and_what_to_raise(wire, capsys):
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "a long answ"}, "done_reason": "length"}
    assert Ollama(num_ctx=4096).complete(PROSE) == "a long answ"
    err = capsys.readouterr().err
    assert ("the answer hit the room left for it (2,846 tokens: what the prompt left of the 4,096-token window, with "
            "a quarter of the prompt to spare)") in err
    assert 'End this session, or raise "context_tokens" in agent.json.' in err
    wire.reply = {"message": {"content": "ok"}, "done_reason": "stop"}
    assert Ollama(num_ctx=4096).complete(PROSE) == "ok" and capsys.readouterr().err == ""


@pytest.mark.parametrize("facts, said, then", [
    ({LLAMA_PROPS: {"n_ctx": 4096}}, "the answer hit the room left for it (2,846 tokens: what the prompt left of "
     "llama.cpp's window, with a quarter of the prompt to spare, at most 16,000)", "--ctx-size"),
    (VLLM_8192, "the answer hit the room left for it (the server's own limit: Kindling sends this server no limit of "
     "its own)", "a bigger window, or a higher limit for an answer if it has one"),
    ({}, "the answer hit the room left for it (the server's own limit", "a higher limit for an answer")],
    ids=["llama.cpp", "vLLM", "unknown kind"])
def test_an_openai_compatible_length_finish_says_the_answer_hit_its_limit(wire, capsys, facts, said, then):
    wire.facts.update(facts)
    model = models.make_model(Config(backend="openai", model="m", base_url=VLLM))
    wire.reply = {"choices": [{"message": {"content": "a long answ"}, "finish_reason": "length"}]}
    assert model.complete(PROSE) == "a long answ"
    err = capsys.readouterr().err
    assert said in err and then in err and "End this session" in err, err
    assert ('set "context_tokens"' not in err) and ("so Kindling can give the answer" not in err)    # no cap follows
    wire.reply = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    assert model.complete(PROSE) == "ok" and capsys.readouterr().err == ""


def test_anthropics_max_tokens_stop_says_the_answer_hit_its_limit(wire, capsys, monkeypatch):
    """CC's call (r5 open question): Anthropic's stop_reason "max_tokens" gets the same note as a 'length' finish."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.reply = {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "a long answ"}]}
    assert Anthropic().complete(MESSAGES) == "a long answ"
    err = capsys.readouterr().err
    assert ("WARNING: the answer hit the room left for it (16,000 tokens, the most Kindling asks Anthropic for, "
            "thinking included) and stopped there, so it may end part way.") in err
    assert "MOST_ANSWER in models.py" in err
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    assert Anthropic().complete(MESSAGES) == "ok" and capsys.readouterr().err == ""


@pytest.mark.parametrize("settings, bigger", [({}, "start the server with a bigger window"),
                                              ({"context_tokens": 2000}, "raise context_tokens")],
                         ids=["the server's own window", "agent.json's window"])
def test_curiositys_note_says_how_to_read_all_by_where_the_window_came_from(wire, tmp_path, settings, bigger):
    """CC's call (r5 open question; Lyra's wording stays otherwise): past a window the server reported, raising
    context_tokens is wrong advice, so the note says to start the server with a bigger window."""
    agent = openai_agent(wire, tmp_path, {LLAMA_PROPS: {"n_ctx": 2000}}, **settings)      # 2,000 either way
    first, _ = add_session(agent.store, [("user", "a"), ("agent", "why do rats sleep by day?"), ("user", "b"),
                                         ("agent", "word " * 2000), ("user", "c"), ("agent", "why is the sky blue?")])
    second, _ = add_session(agent.store, [("user", "tell me everything"), ("agent", "word " * 3000)])
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    assert agent.notes == [f"curiosity, session {first}: read turns 3-3 of 3; {bigger} to read all",
                           f"curiosity, session {second}: read no turns of 1 (the newest alone is too long); {bigger} "
                           "to read all"]


@pytest.mark.parametrize("reply", [{"message": {"content": "ok"}, "done_reason": ["length"]},
                                   {"choices": [{"message": {"content": "ok"}, "finish_reason": None}]},
                                   {"choices": "ok"}])
def test_reading_the_finish_never_raises_whatever_its_shape(wire, reply):
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = reply
    model = Ollama() if "message" in reply else OpenAICompatible("m", VLLM)
    result = outcome(model.complete, MESSAGES)
    assert result == "ok" or (isinstance(result, ModelError) and "had no choices/0" in str(result)), result


# --- An OpenAI-compatible server's refusal of a prompt too long is PromptTooLong too (r4 verifier's LOW) ------

LLAMA_TOO_LONG = {"error": {"code": 400, "message": "the request exceeds the available context size, try increasing "
                            "it", "type": "exceed_context_size_error", "n_prompt_tokens": 5120, "n_ctx": 4096}}


# vLLM's 400 for a prompt too long, in its words (vLLM's own source, as we know it; not measured here)
VLLM_SAID = ("This model's maximum context length is 4096 tokens. However, you requested 5000 tokens (4000 in the "
             "messages, 1000 in the completion). Please reduce the length of the messages or completion.")
VLLM_TOO_LONG = {"object": "error", "message": VLLM_SAID, "type": "BadRequestError", "param": None, "code": 400}


@pytest.mark.parametrize("body, quoted", [
    (LLAMA_TOO_LONG, "5120 tokens by its own count, for its 4096-token window"),          # llama.cpp: under "error"
    (TOO_LONG, "1757 tokens by its own count, for its 256-token window"),                 # Ollama's, passed on by /v1
    ({"error": {"type": "exceed_context_size_error", "message": "too long"}}, "exceed_context_size_error"),
    (VLLM_TOO_LONG, "5000 tokens by its own count, for its 4096-token window"),           # vLLM, with a max_tokens
    ({**VLLM_TOO_LONG, "message": "This model's maximum context length is 4096 tokens. However, you requested 4600 "
      "tokens in the messages, Please reduce the length of the messages."},              # vLLM, without one
     "4600 tokens by its own count, for its 4096-token window"),
    ({"error": {"message": "This model's maximum context length is 2048 tokens. However, your request has 2100 input "
                "tokens. Please reduce the length of the input messages.", "type": "BadRequestError", "code": 400}},
     "2100 tokens by its own count, for its 2048-token window")],                          # newer vLLM, under "error"
    ids=["llama.cpp", "Ollama's /v1", "the error type alone", "vLLM", "vLLM, no max_tokens", "newer vLLM"])
def test_an_openai_compatible_servers_400_for_a_prompt_too_long_is_prompt_too_long(wire, body, quoted):
    wire.refusal = (400, body)
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, PromptTooLong) and quoted in str(error), error
    assert str(error).startswith("The server refused the prompt as too long") and "bigger window" in str(error)


@pytest.mark.parametrize("status, body", [
    (400, {"object": "error", "message": "This model's maximum context length is 4096 tokens.", "code": 400}),
    (400, {"object": "error", "message": "The model `m` does not exist.", "type": "NotFoundError", "code": 400}),
    (500, LLAMA_TOO_LONG), (500, VLLM_TOO_LONG), (413, VLLM_TOO_LONG)],
    ids=["vLLM's window alone, no count", "another 400", "llama.cpp's body, 500", "vLLM's words, 500", "vLLM's, 413"])
def test_any_other_openai_compatible_refusal_stays_a_plain_model_error(wire, status, body):
    wire.refusal = (status, body)
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error


@pytest.mark.parametrize("body, quoted", [(LLAMA_TOO_LONG, "5120 tokens by its own count, for its 4096-token window"),
                                          (VLLM_TOO_LONG, "5000 tokens by its own count, for its 4096-token window")],
                         ids=["llama.cpp", "vLLM"])
def test_a_servers_refusal_during_curiosity_skips_that_session_and_reads_the_rest(wire, tmp_path, body, quoted):
    """Before, a refusal from llama.cpp (r4) or vLLM (r5) was a plain error: curiosity stopped there, left every
    later session unread, and stopped there again at each session end."""
    agent = openai_agent(wire, tmp_path, {}, context_tokens=4096)
    add_session(agent.store, [("user", "go"), ("agent", "the long one")])
    small, _ = add_session(agent.store, [("user", "hi"), ("agent", "why is the sky blue?")])
    wire.refusal = lambda sent: (400, body) if "the long one" in sent["messages"][-1]["content"] else None
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    read = [sent["messages"][-1]["content"] for _, sent, _ in wire.sent]
    assert len(read) == 2 and "why is the sky blue?" in read[1], read          # the next session was still read
    assert agent.store.unread_sessions() == []                                 # both marked read
    assert agent.notes == ["curiosity, session 1: read no turns of 1: the server refused turns 1-1 as too long "
                           "for its window (see: log)"]
    reason = only(agent.store.db.execute("SELECT reason FROM log WHERE kind='curiosity read part'"))[0]
    assert quoted in reason
    agent.end_session()
    assert len(wire.sent) == 2 and len(agent.notes) == 1                       # nothing repeats at the next end


def test_openai_compatible_request_and_reply(wire, monkeypatch):
    monkeypatch.setenv("KINDLING_TEST_KEY", "test-key-not-real")
    wire.reply = {"choices": [{"message": {"content": "hello"}}]}
    assert OpenAICompatible("m", "http://localhost:8000/v1/", "KINDLING_TEST_KEY").complete(MESSAGES) == "hello"
    ((url, body, headers),) = wire.sent
    assert (url, body) == ("http://localhost:8000/v1/chat/completions", {"model": "m", "messages": MESSAGES})
    assert headers.get("authorization") == "Bearer test-key-not-real"


def test_anthropic_request_reply_and_refusal(wire, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "..."},
                                                         {"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]}
    assert Anthropic().complete(MESSAGES) == "hello"
    ((url, body, headers),) = wire.sent
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "test-key-not-real"
    assert (body.get("system"), body.get("messages")) == ("SYS", [{"role": "user", "content": "hi"}])
    wire.reply = {"stop_reason": "refusal", "content": []}
    assert "declined" in Anthropic().complete(MESSAGES)


def test_the_default_addresses_are_ollama_on_this_computer_and_the_providers_own(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    assert Ollama().base_url == "http://localhost:11434"
    assert OpenAICompatible("m").base_url == "https://api.openai.com/v1"


# --- Keys go only where you sent them (army security-auditor and adversarial-reviewer, r3) ----------------

def test_a_missing_key_stops_with_a_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stopped = outcome(Anthropic)
    assert isinstance(stopped, SystemExit) and "ANTHROPIC_API_KEY" in str(stopped.code)
    stopped = outcome(OpenAICompatible)
    assert isinstance(stopped, SystemExit) and 'needs "model"' in str(stopped.code)


def test_the_usual_key_goes_only_to_the_providers_own_address(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real-2")
    assert OpenAICompatible("m").key == "test-key-not-real-1"                         # api.openai.com
    assert OpenAICompatible("m", "http://localhost:1234/v1").key == ""               # LM Studio: not sent
    assert isinstance(outcome(Anthropic, "", "https://gateway.example.org"), SystemExit)   # name it, or nothing
    monkeypatch.setenv("CLASS_KEY", "test-key-not-real-3")
    assert OpenAICompatible("m", "https://vllm.example.org/v1", "CLASS_KEY").key == "test-key-not-real-3"


def test_a_key_is_never_sent_over_plain_http_to_another_machine(monkeypatch):
    monkeypatch.setenv("CLASS_KEY", "test-key-not-real-3")
    stopped = outcome(OpenAICompatible, "m", "http://203.0.113.9:8000/v1", "CLASS_KEY")
    assert isinstance(stopped, SystemExit) and "plain http" in str(stopped.code)
    assert OpenAICompatible("m", "http://127.0.0.1:8000/v1", "CLASS_KEY").key == "test-key-not-real-3"


def test_openai_with_no_address_and_no_key_sends_nothing(monkeypatch, wire):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stopped = outcome(models.make_model, Config(backend="openai", model="m"))
    assert isinstance(stopped, SystemExit) and "base_url" in str(stopped.code) and wire.sent == []


# --- A local model never goes through a proxy (army security-auditor, r3) ---------------------------------

def test_a_local_address_ignores_a_proxy_in_the_environment(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://203.0.113.50:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    tried = []

    def connect(sock, address):
        tried.append(address[0])
        raise ConnectionRefusedError("nothing listening (the test)")
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1" if host == "localhost" else host, port))])
    assert isinstance(outcome(models.fetch_json, "http://localhost:11434/api/chat", {}), OSError)
    assert tried == ["127.0.0.1"]                                   # straight to this computer, not the proxy


# --- A reply the agent can't use is a message, never a traceback (army red-hat-tester, r3) ----------------

@pytest.mark.parametrize("reply", [None, 42, {"message": {}}, {"message": {"content": None}}, ["a", "list"]])
def test_an_unusable_ollama_reply_is_a_model_error(wire, reply):
    wire.reply = reply
    assert isinstance(outcome(Ollama().complete, MESSAGES), ModelError)


@pytest.mark.parametrize("reply", [{"choices": []}, {"choices": [{"message": {"content": None, "refusal": "no"}}]}])
def test_an_unusable_openai_reply_is_a_model_error(wire, reply, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    wire.reply = reply
    assert isinstance(outcome(OpenAICompatible("m").complete, MESSAGES), ModelError)


def test_an_http_error_shows_what_the_server_said(monkeypatch):
    def refuse(opener, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                     BytesIO(b'{"error": "model \\"llama3.1:8b\\" not found, try pulling it first"}'))
    monkeypatch.setattr("urllib.request.OpenerDirector.open", refuse)
    error = outcome(models.fetch_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "404" in str(error) and "try pulling it first" in str(error)


def test_a_reply_that_is_not_json_is_a_model_error(monkeypatch):
    monkeypatch.setattr("urllib.request.OpenerDirector.open",
                        lambda opener, request, timeout=None: Answer(b"<html>Sign in to the campus Wi-Fi</html>"))
    error = outcome(models.fetch_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "campus Wi-Fi" in str(error)


# --- The prompt is checked against the context before it's sent (Nexus) ------------------------------------

def test_a_prompt_too_long_for_the_context_is_refused_not_cut():
    fits = [{"role": "user", "content": "x" * 4000}]                # ~1,000 tokens; with a quarter, 1,250
    assert check_fits(fits, 1250) is None
    error = outcome(check_fits, fits, 1249)
    assert isinstance(error, ModelError) and "too long" in str(error) and "1,249" in str(error)


# --- The token estimate (Nexus, 20 Nov): it errs high where four characters a token undercounted ------------
# Nexus measured real tokens against the old estimate, (len + 3) // 4, on texts like these (Mistral 7B v0.3 /
# Qwen3.5-27B): prose 1.14 / 1.04, code 1.20 / 1.03, Spanish 1.16 / 0.88, mixed languages 1.91 / 1.24,
# JSON 2.08 / 2.08, numbers 4.00 / 4.00, Japanese 4.50 / 1.92. A test can't call a tokenizer, so this pins the
# estimate's shape instead: never below the old one, and a digit or a non-ASCII character is a whole token.

ESTIMATED = [
    ("prose", "The mice were weighed every Monday morning, and the heavier ones moved to the second cage."),
    ("code", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n"),
    ("Spanish", "Los ratones se pesaron cada lunes por la mañana, y los más pesados pasaron a la segunda jaula."),
    ("mixed languages", "Los ratones, les souris, die Mäuse, マウス, мыши: weighed on Mondays."),
    ("JSON", '[{"id": 1, "mass_g": 21.4, "cage": "B"}, {"id": 2, "mass_g": 19.8, "cage": "A"}]'),
    ("numbers", "3.14159 2.71828 1.41421 0.57721 1.61803 4.66920"),
    ("Japanese", "マウスは毎週月曜日の朝に体重を測定した。"),
]


@pytest.mark.parametrize("kind, text", ESTIMATED, ids=[kind for kind, _ in ESTIMATED])
def test_the_estimate_is_never_below_four_characters_a_token(kind, text):
    new, old = estimate_tokens(text), (len(text) + 3) // 4
    whole = sum(c.isdigit() or ord(c) > 127 for c in text)          # each a token of its own
    assert new >= old and new >= whole, (kind, new, old, whole)
    assert estimate_tokens(text + "7") == new + 1                     # one more digit, one more token
    assert estimate_tokens(text + "語") == new + 1                    # one more non-ASCII character, too


def test_a_lone_punctuation_mark_costs_and_an_accented_letter_is_a_whole_token():
    """The round-4 verifier's V1 and V2: half a token per mark rounds UP, and Latin-1 (128 to 255) is non-ASCII."""
    assert estimate_tokens(".") == 1 and estimate_tokens("...") == 2
    assert estimate_tokens("é" * 100) == estimate_tokens("ñ" * 100) == 100


def test_digits_and_non_ascii_characters_are_whole_tokens_and_letters_four_a_token():
    assert estimate_tokens("1234567890" * 10) == 100                  # the old estimate said 25
    assert estimate_tokens("語" * 100) == estimate_tokens("ж" * 100) == 100
    assert estimate_tokens("." * 100) == 50                           # ASCII punctuation: half a token
    assert estimate_tokens("word " * 100) == 125                      # prose: as before


# --- Reading the model's JSON block (army red-hat-tester, security-auditor, test-engineer, r3) -------------

KEYS = ("beliefs", "intention_done")


def test_the_last_block_counts_and_an_example_you_asked_for_stays_in_the_reply():
    example = '```json\n{"name": "trial", "beliefs": ["x"]}\n```'
    prose, found = split_json(f"Here is the config:\n{example}\nSave it.", KEYS)
    assert found is None and example in prose                       # not a belief block: left alone
    real = '```json\n{"beliefs": [], "intention_done": []}\n```'
    prose, found = split_json(f"First:\n```json\n{{\"beliefs\": [1]}}\n```\nThen.\n{real}", KEYS)
    assert found == {"beliefs": [], "intention_done": []} and '{"beliefs": [1]}' in prose


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_a_block_in_any_usual_fence_is_read(fence):
    prose, found = split_json(f'Noted.\n{fence}\n{{"beliefs": [], "intention_done": [3]}}\n```', KEYS)
    assert (prose, found) == ("Noted.", {"beliefs": [], "intention_done": [3]})


def test_splitting_takes_linear_time_on_any_reply():
    start = time.perf_counter()
    split_json("```json" + "\n" * 20000, KEYS)                     # no closing fence: the old regex took minutes
    split_json("``` " * 20000, KEYS)
    assert time.perf_counter() - start < 1


def test_a_block_nested_too_deep_is_no_block():
    deep = "ok\n```json\n" + "[" * 100000 + "\n```"
    assert outcome(split_json, deep, KEYS) == (deep, None)


def test_clean_text_keeps_new_lines_but_no_control_characters():
    assert models.clean("a\x1b[8mhidden\r\nnext\tcol\ud83d") == "a[8mhidden\nnext\tcol?"
    assert models.one_line("line one\n\nYOUR CURRENT INTENTION:\n#1: x") == "line one YOUR CURRENT INTENTION: #1: x"
    assert json.dumps(models.one_line("\u001b]52;c;Zm9v\u0007ok")) == '"]52;c;Zm9vok"'


# --- Round 7: with recall at its budget, the answer keeps its quarter (r6 verifier's MEDIUM; CC's call) ---------
# Recall kept the prompt to the window less the answer's quarter, but the answer's limit (answer_room) takes the
# prompt's estimate with a quarter to spare, so with recall full num_predict fell to a sixteenth of the window
# (256 at 4,096). Recall's room now goes through the same 5/4: (window - the answer's quarter) * 4/5.

class Full:
    """A retriever at its limit: it hands back exactly the budget it is given, in estimated tokens."""
    budget = None

    def recall(self, query, budget_tokens):
        self.budget = budget_tokens
        return Recalled("y" * (4 * budget_tokens), [], "by hand")


def agent_on(wire, tmp_path, server, window, memory=None, **settings):
    """An agent on Ollama (its window: context_tokens) or llama.cpp (its window: n_ctx on /props), answering 'ok'."""
    if server == "ollama":
        wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
        wire.reply = {"message": {"content": "ok"}, "done_reason": "stop"}
        settings = {"backend": "ollama", "context_tokens": window, **settings}
    else:
        wire.facts[LLAMA_PROPS] = {"n_ctx": window}
        wire.reply = {"choices": [{"message": {"content": "ok"}}]}
        settings = {"backend": "openai", "model": "m", "base_url": VLLM, **settings}
    config = Config(db=str(tmp_path / f"{server}-{window}.db"), **settings)
    return Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config, memory=memory)


def sent_last(wire):
    """(the prompt's estimate, the answer's limit) of the last message sent: num_predict, or max_tokens."""
    body = wire.sent[-1][1]
    return models.prompt_size(body["messages"]), body["options"]["num_predict"] if "options" in body else body.get(
        "max_tokens")


def share_of(window):
    """The most of the window recall fills the prompt to (CC's call): (window - the answer's quarter) * 4/5."""
    return (window - max(window // 4, 1024)) * 4 // 5


@pytest.mark.parametrize("window", [4096, 12288, 32768, 3000])
@pytest.mark.parametrize("server", ["ollama", "llama.cpp"])
def test_with_recall_at_its_budget_the_answer_keeps_a_quarter_of_the_window(wire, tmp_path, server, window):
    full = Full()
    agent = agent_on(wire, tmp_path, server, window, memory=full, recall_budget_tokens=10**6)   # the room binds
    assert agent.handle("why do the mice lose weight?") == "ok"
    size, room = sent_last(wire)
    assert full.budget > 1000 and share_of(window) - 3 <= size <= share_of(window), (full.budget, size)  # it's full
    assert room >= window // 4 and room >= 1024, (room, window)            # a quarter, and at least 1,024


@pytest.mark.parametrize("window", [4096, 12288, 32768])
def test_with_whole_sessions_recalled_the_answer_keeps_its_quarter(wire, tmp_path, window):
    """The r6 verifier's reproduction, with the real recall: sessions it can fill its budget from, and a long
    current session. Round 6 sent num_predict 406 at 4,096, and a 600-token reply stopped part way."""
    agent = agent_on(wire, tmp_path, "ollama", window, recall_budget_tokens=10**6)
    for n in range(70):
        add_session(agent.store, [("user", f"the mice in cage {n} lost weight " + "mice weight cage " * 60),
                                  ("agent", "noted " * 40)])
    sid = agent.store.start_session()
    for n in range(window // 1024):
        agent.store.add_turn(sid, "user", f"about the mice, point {n}: " + "detail " * 40)
        agent.store.add_turn(sid, "agent", "ok " * 40)
    assert agent.handle("why do the mice lose weight?") == "ok"
    size, room = sent_last(wire)
    assert share_of(window) - 400 <= size <= share_of(window), (size, share_of(window))      # recall filled it
    assert room >= window // 4, (room, window)


def test_the_readmes_quarter_sentence_is_what_the_code_does(wire, tmp_path):
    readme = " ".join((Path(__file__).parent.parent / "README.md").read_text().split())
    assert ("The answer keeps a quarter of the context (at least 1,024 tokens), and the prompt may take four fifths "
            "of the rest") in readme
    ((window, prompt, answer),) = re.findall(r"In a ([\d,]+)-token window, recall fills the prompt to at most ([\d,]+) "
                                             r"tokens, leaving the answer at least ([\d,]+)\.", readme)
    window, prompt, answer = (int(n.replace(",", "")) for n in (window, prompt, answer))
    assert agent_on(wire, tmp_path, "ollama", window, memory=Full(), recall_budget_tokens=10**6).handle("hi") == "ok"
    size, room = sent_last(wire)
    assert prompt - 3 <= size <= prompt and room >= answer, (size, room)
    assert answer == max(window // 4, 1024) and prompt == share_of(window)


# --- Round 7: an error body is searched only so far (r6 verifier's LOW) ----------------------------------------

@contextmanager
def under_a_second():
    """Fail the test if the block takes a second or more. Where the system has a timer signal it is stopped at one
    second, by pytest.fail (a regular expression checks for signals as it runs), so a slow mutant fails fast."""
    timer = hasattr(signal, "setitimer")
    if timer:
        def stop(*_):
            pytest.fail("it took a second or more")
        before = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
    start = time.perf_counter()
    try:
        yield
    finally:
        if timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, before)
    assert time.perf_counter() - start < 1


HOSTILE = {"object": "error", "message": "maximum context length is 1 tokens " * 8000}      # 280 KB


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_a_280_kb_hostile_error_body_is_read_in_under_a_second(wire, monkeypatch, backend):
    """The r6 verifier's LOW: VLLM_TOO_LONG over the whole of a 280 KB body like this one was still running at 120 s
    (its lazy gap after a literal prefix is quadratic). Now it searches only the body's first 2,000 characters.
    The opener is a bare one here, so the second timed is the body's: building a real opener loads the system's
    certificates (about 150 ms of CPU, measured here, 21 Nov), which under load ran past it. Kindling builds one
    once per run (opener_for), and each test starts as a new run."""
    assert len(json.dumps(HOSTILE)) >= 280_000
    monkeypatch.setattr(models.urllib.request, "build_opener", lambda *handlers: models.urllib.request.OpenerDirector())
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.refusal = (400, HOSTILE)
    model = Ollama() if backend == "ollama" else OpenAICompatible("m", VLLM)
    with under_a_second():
        error = outcome(model.complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error


def test_vllms_words_are_looked_for_only_in_the_first_2000_characters_of_the_body(wire):
    wire.refusal = (400, {"object": "error", "detail": "x" * 2000, "message": VLLM_SAID})     # its words come late
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error
    wire.refusal = (400, {**VLLM_TOO_LONG, "detail": "x" * 300_000})       # the usual refusal, in a long body
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, PromptTooLong) and "5000 tokens by its own count, for its 4096-token window" in str(error)


# --- Round 7: the chat refusal gives curiosity's advice for a window the server reported (CC's call) ------------

@pytest.mark.parametrize("backend, facts, settings, bigger", [
    ("openai", {LLAMA_PROPS: {"n_ctx": 2048}}, {}, "start the server with a bigger window"),
    ("openai", {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 2048}]}}, {},
     "start the server with a bigger window"),
    ("openai", {LLAMA_PROPS: {"n_ctx": 2048}}, {"context_tokens": 2048}, 'raise "context_tokens" in agent.json'),
    ("openai", {}, {}, 'raise "context_tokens" in agent.json'),
    ("ollama", {}, {}, 'raise "context_tokens" in agent.json')],
    ids=["llama.cpp's own window", "vLLM's own window", "agent.json's window", "an assumed window", "Ollama"])
def test_the_chat_refusal_says_how_to_get_a_bigger_window_by_where_the_window_came_from(wire, tmp_path, backend,
                                                                                         facts, settings, bigger):
    """As curiosity's note does (r6): past a window the server reported, raising context_tokens is wrong advice,
    and on llama.cpp it is the bold claim's exception (r6 verifier)."""
    agent = openai_agent(wire, tmp_path, facts, **{"backend": backend, **settings})
    error = outcome(agent.handle, "word " * 20000)                         # ~25,000 tokens: too long for any of them
    assert isinstance(error, PromptTooLong), error
    assert str(error).endswith(f"End this session (/end, or `session end`), or {bigger}"), error
    assert wire.sent == [] and agent.store.turns(1) == []                  # never sent, and not kept


# --- Round 7: the round-6 verifier's two surviving mutants, each now with a test ------------------------------

def test_curiosity_on_a_session_with_no_agent_turn_asks_no_server(wire, tmp_path):
    """R3: with no agent turn there is nothing to read, so curiosity asks the server nothing, not even its window."""
    agent = openai_agent(wire, tmp_path, {})
    add_session(agent.store, [("user", "only me")])
    agent.end_session()
    assert wire.asked == [] and wire.sent == [] and agent.store.unread_sessions() == []


def test_curiositys_note_on_an_assumed_openai_window_says_raise_context_tokens(wire, tmp_path):
    """R7: a server that could say its window but didn't (an assumed window) is no reported window, so setting
    context_tokens is the right advice."""
    agent = openai_agent(wire, tmp_path, {})                                # /v1/models and /props: 404
    sid, _ = add_session(agent.store, [("user", "a"), ("agent", "word " * 20000), ("user", "b"), ("agent", "why?")])
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    assert agent.notes == [f"curiosity, session {sid}: read turns 2-2 of 2; raise context_tokens to read all"]


# --- Round 8: urllib's openers are built once per process (r7 fixer's open question; CC's call) --------------
# Building an opener loads the system's certificates (about 150 ms of CPU, measured 21 Nov): once per run is enough.

def test_two_requests_share_one_opener_built_once_per_process(wire, monkeypatch):
    built, used = [], []
    real_build, real_open = models.urllib.request.build_opener, models.urllib.request.OpenerDirector.open
    monkeypatch.setattr(models.urllib.request, "build_opener", lambda *handlers: built.append(handlers) or
                        real_build(*handlers))
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda opener, request, timeout=None: used.append(
        opener) or real_open(opener, request, timeout))                           # the wire's, recording the opener
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "ok"}}
    model = Ollama()
    assert model.complete(MESSAGES) == "ok" and model.complete(MESSAGES) == "ok"
    assert wire.methods == ["GET", "POST", "POST"] and len(used) == 3               # the version, then two messages
    assert len(built) == 1 and all(opener is used[0] for opener in used), (built, used)


def test_a_local_address_never_borrows_the_proxied_opener_of_an_earlier_request(monkeypatch):
    """Built once per run, the openers are still two: one for another machine (through your proxy, as before) and
    one for this computer (never through it). A local request after a remote one still goes straight here."""
    monkeypatch.setenv("http_proxy", "http://203.0.113.50:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    tried = []

    def connect(sock, address):
        tried.append(address[0])
        raise ConnectionRefusedError("nothing listening (the test)")
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1" if host == "localhost" else host, port))])
    for url in ("http://models.example.org/v1/models", "http://localhost:11434/api/chat",
                "http://models.example.org/v1/models", "http://localhost:11434/api/chat"):
        assert isinstance(outcome(models.fetch_json, url, {}), OSError)
    assert tried == ["203.0.113.50", "127.0.0.1"] * 2                               # remote by the proxy; local not
    assert len(models.OPENERS) == 2


# --- Round 8: the README names where the answer's limit is less than its quarter (r7 verifier's LOW; CC's call) ---

def test_the_readme_names_llama_cpps_ceiling_and_anthropics_fixed_limit(wire, tmp_path, monkeypatch):
    """'With recall full the answer still gets its quarter' overstated: llama.cpp's limit is at most MOST_ANSWER, so a
    window over four times that leaves the answer less than a quarter, and Anthropic's is MOST_ANSWER at any window."""
    readme = " ".join((Path(__file__).parent.parent / "README.md").read_text().split())
    ((ceiling, over, leaves),) = re.findall(r"and on llama\.cpp up to its ([\d,]+)-token ceiling: a llama\.cpp window "
                                            r"over ([\d,]+) tokens leaves the answer ([\d,]+), less than a quarter", readme)
    (fixed,) = re.findall(r"Anthropic's limit is a fixed ([\d,]+) whatever the window", readme)
    ceiling, over, leaves, fixed = (int(n.replace(",", "")) for n in (ceiling, over, leaves, fixed))
    assert ceiling == leaves == fixed == models.MOST_ANSWER and over == 4 * models.MOST_ANSWER
    for window in (over, 2 * over):                                   # at the edge: a quarter; past it: the ceiling
        assert agent_on(wire, tmp_path, "llama.cpp", window, memory=Full(), recall_budget_tokens=10**6).handle("hi") == "ok"
        _, room = sent_last(wire)
        assert room == min(window // 4, ceiling) and (room < window // 4) == (window > over), (window, room)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    for window in (2048, 1_000_000):
        assert outcome(Agent(Store(str(tmp_path / f"a{window}.db"), Clock()), Anthropic(), HashEmbedder(),
                             Config(backend="anthropic", context_tokens=window)).handle, "hi") == "ok"
        assert wire.sent[-1][1]["max_tokens"] == fixed


# --- Round 8: the quarter holds with beliefs and an intention in the prompt (r7 verifier's W4; CC's call) --------

@pytest.mark.parametrize("window", [4096, 12288])
@pytest.mark.parametrize("server", ["ollama", "llama.cpp"])
def test_with_many_beliefs_and_an_intention_the_answer_keeps_its_quarter(wire, tmp_path, server, window):
    """The r7 verifier's test, adopted: recall's room is taken from the whole prompt without recall (the
    instructions, the beliefs, the intention and this session). With the beliefs left out of it (its W4), recall
    overfilled the prompt, and the answer got 401 tokens at 4,096, not 1,028."""
    from starter_agent import bdi
    full = Full()
    agent = agent_on(wire, tmp_path, server, window, memory=full, recall_budget_tokens=10**6)
    many = 4 if window < 8000 else 12                                    # beliefs of ~115 tokens each
    sid, (turn,) = add_session(agent.store, [("user", "remember these")])
    for i in range(many):
        agent.store.append("beliefs", {"text": f"belief {i}: " + "the mice weigh less after the night feed " * 11,
                                       "confidence": 0.6, "evidence": json.dumps([turn]), "session_id": sid,
                                       "flag": None})
    bdi.add_desire(agent.store, "Find out why the mice lose weight at night, cage by cage, " + "and by diet " * 30)
    assert agent.handle("why do the mice lose weight?") == "ok"
    prompt = wire.sent[-1][1]["messages"][-1]["content"]
    assert prompt.count("the night feed") == 11 * many and prompt.count("and by diet") == 30    # both are in it
    size, room = sent_last(wire)
    assert full.budget > 1000 and share_of(window) - 3 <= size <= share_of(window), (full.budget, size)  # it's full
    assert room >= window // 4 and room >= 1024, (room, window)


# --- Round 8: `recall WORDS` packs to a chat turn's budget where that needs no request (r7 verifier's LOW; CC's call) -

BIG = "the mice lose weight " + "z" * 30000        # one past session of ~7,530 tokens: under 8,000, over a turn's room


@pytest.mark.parametrize("settings", [{}, {"backend": "ollama"},
                                      {"backend": "openai", "model": "m", "base_url": VLLM, "context_tokens": 12288}],
                         ids=["fake", "ollama", "openai, context_tokens set"])
def test_recall_packs_to_the_budget_a_chat_turn_would_have(wire, cli, tmp_path, capsys, settings):
    """At the default 12,288 window a chat turn's room (about 7,000) is under the 8,000 cap, so `recall` listed a
    session that chat left out. Now it packs to the budget the next chat turn has, asking no server: here with an
    open session, and a desire whose intention that turn writes down first."""
    (tmp_path / "agent.json").write_text(json.dumps(settings))
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = ({"choices": [{"message": {"content": "ok"}}]} if settings.get("backend") == "openai"
                  else {"message": {"content": "ok"}})
    big, _ = add_session(Store(str(tmp_path / "agent.db"), Clock()), [("user", BIG)])
    assert cli("chat", "hello") == 0                                                   # an open session
    assert cli("desire", "add", "Find out why the mice lose weight", "--rank", "1") == 0  # its intention: not yet
    capsys.readouterr()
    wire.methods.clear()
    assert cli("recall", "mice", "weight") == 0
    out = capsys.readouterr().out
    assert wire.methods == [] and "Nothing recalled." in out and "may pack less" not in out, (wire.methods, out)
    tokens, budget = only(re.findall(rf"session {big}  score [\d.]+  skipped: ~([\d,]+) tokens, more than was left of "
                                     r"the ([\d,]+)-token budget", out))
    assert cli("chat", "mice", "weight") == 0
    turn = only(re.findall(rf"recall skipped session {big} \(score [\d.]+\): ~[\d,]+ tokens, more than was left of "
                           r"this turn's ([\d,]+)-token budget", capsys.readouterr().out))
    tokens, budget, turn = (int(n.replace(",", "")) for n in (tokens, budget, turn))
    assert budget == turn < tokens < 8000, (budget, turn, tokens)


@pytest.mark.parametrize("settings", [{"backend": "openai", "model": "m", "base_url": VLLM}, {"backend": "anthropic"}],
                         ids=["openai", "anthropic"])
def test_where_only_the_server_knows_its_window_recall_packs_to_the_cap_says_so_and_asks_nothing(wire, cli, tmp_path,
                                                                                                capsys, settings):
    (tmp_path / "agent.json").write_text(json.dumps({**settings, "recall_budget_tokens": 10**6}))
    small, _ = add_session(Store(str(tmp_path / "agent.db"), Clock()), [("user", "the mice lose weight")])
    assert cli("recall", "mice", "weight") == 0
    out = capsys.readouterr().out
    assert wire.methods == [] and f"session {small}  score" in out, (wire.methods, out)
    assert ('(Packed to "recall_budget_tokens", 1,000,000: this model\'s window isn\'t known without asking its '
            "server, which `recall` never does, so a chat turn may pack less.") in " ".join(out.split())


def test_where_a_chat_turn_would_skip_recall_the_recall_command_says_so(cli, tmp_path, capsys):
    (tmp_path / "agent.json").write_text('{"context_tokens": 3000}')
    add_session(Store(str(tmp_path / "agent.db"), Clock()), [("user", "my cat is called Biscuit")])
    assert cli("chat", "x" * 2400) == 0                         # with the demo's echo, ~1,200 tokens of this session
    capsys.readouterr()
    assert cli("recall", "cat") == 0
    out = capsys.readouterr().out
    assert out.startswith("Nothing recalled: a chat turn now would skip recall") and "session 1" not in out, out
    assert cli("chat", "cat") == 0 and "recall skipped: this session leaves ~" in capsys.readouterr().out


def test_next_id_is_the_id_sqlite_gives_the_next_row(tmp_path):
    """`recall` predicts the rows a chat turn would write without writing them: their ids must be SQLite's own."""
    store = Store(str(tmp_path / "ids.db"), Clock())
    assert [store.next_id(table) for table in ("sessions", "turns", "intentions")] == [1, 1, 1]
    sid = store.start_session()
    store.add_turn(sid, "user", "a")
    store.add_turn(sid, "user", "b")
    store.forget_session(sid)                     # the largest ids are gone, but sessions and turns never reuse one
    predicted = store.next_id("sessions"), store.next_id("turns")
    sid = store.start_session()
    assert predicted == (sid, store.add_turn(sid, "user", "c")) == (2, 3)
    for text in ("x", "y"):
        store.append("intentions", {"text": text, "serves": 1, "status": "active", "evidence": "[]", "flag": None})
    store.db.execute("DELETE FROM intentions WHERE id=2")       # in the other tables, SQLite reuses the largest
    predicted = store.next_id("intentions")
    assert predicted == store.append("intentions", {"text": "z", "serves": 1, "status": "active", "evidence": "[]",
                                                    "flag": None}) == 2


# --- Round 8: curiosity's answer keeps its quarter, and an answer cut off never marks its session read (r7 verifier's
#     MEDIUM; CC's call) ---------------------------------------------------------------------------------------------

def finished(server, text, finish="stop"):
    """A reply from Ollama or llama.cpp with this text, finished this way."""
    if server == "ollama":
        return {"message": {"content": text}, "done_reason": finish}
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


def day(store, window, share):
    """A finished session whose agent turns (about 150 tokens each) come to about this share of the window."""
    turns = []
    for i in range(int(window * share) // 150):
        turns += [("user", f"go on {i}"), ("agent", f"part {i}: " + "word " * 115)]
    sid, _ = add_session(store, turns)
    return sid, [t for t in store.turns(sid) if t["role"] == "agent"]


def cut_rows(store):
    return [row[0] for row in store.db.execute("SELECT reason FROM log WHERE kind='curiosity cut' ORDER BY id")]


@pytest.mark.parametrize("length", [0.7, 1.5], ids=["over the share", "longer than the window"])
@pytest.mark.parametrize("window", [4096, 12288, 32768, 3000])
@pytest.mark.parametrize("server", ["ollama", "llama.cpp"])
def test_curiositys_answer_keeps_a_quarter_of_the_window_with_a_long_session(wire, tmp_path, server, window, length):
    """The r7 verifier's MEDIUM: curiosity filled its prompt to the length check's edge, so its answer's limit
    (answer_room) came to a median of ~260 tokens at any window, sometimes under 30, and a cut answer lost that
    session's questions. Its prompt now takes recall's share, so its answer keeps a quarter of the window, and at
    least 1,024 tokens (3,000: where the floor, not the quarter, sets it)."""
    agent = agent_on(wire, tmp_path, server, window)
    wire.reply = finished(server, '{"questions": []}')
    day(agent.store, window, length)
    agent.end_session()
    size, room = sent_last(wire)
    assert len(wire.sent) == 1 and share_of(window) - 160 < size <= share_of(window), (size, share_of(window))
    assert room >= window // 4 and room >= 1024, (room, window)
    assert agent.store.unread_sessions() == [] and "read turns" in only(agent.notes), agent.notes


@pytest.mark.parametrize("server", ["ollama", "llama.cpp"])
def test_an_answer_cut_off_leaves_its_session_unread_and_the_next_end_reads_fewer_turns(wire, tmp_path, server):
    from starter_agent import curiosity
    agent = agent_on(wire, tmp_path, server, 4096)
    sid, mine = day(agent.store, 4096, 0.5)                               # read whole at first: ~2,000 tokens
    asked = {"text": "Why does the last part repeat?", "quote": "word word", "turn": mine[-1]["id"]}
    wire.reply = finished(server, '{"questions": [{"text": "Why does the last', "length")    # cut part way
    agent.end_session()
    assert [s["id"] for s in agent.store.unread_sessions()] == [sid] and curiosity.open_questions(agent.store) == []
    assert "was cut off: it stopped at its length limit" in only(cut_rows(agent.store))
    assert agent.notes == [f"curiosity, session {sid}: the answer was cut off (it stopped at its length limit), so the "
                           "session stays unread; the next session end reads it again, fewer turns at a time (see: log)"]
    wire.reply = finished(server, json.dumps({"questions": [asked]}))
    agent.end_session()
    assert agent.store.unread_sessions() == [] and len(cut_rows(agent.store)) == 1
    assert [q["source_turn"] for q in curiosity.open_questions(agent.store)] == [asked["turn"]]   # no question lost
    bodies = [body for _, body, _ in wire.sent]
    sizes = [models.prompt_size(body["messages"]) for body in bodies]
    rooms = [body["options"]["num_predict"] if "options" in body else body["max_tokens"] for body in bodies]
    read = [body["messages"][-1]["content"].count("] agent: ") for body in bodies]
    bare = models.prompt_size(curiosity.prompt([]))
    assert len(bodies) == 2 and read[0] == len(mine) > read[1] > 0, read                 # fewer turns, the newest
    assert sizes[1] <= bare + (sizes[0] - bare) // 2 and rooms[1] > rooms[0], (sizes, rooms)


def test_a_length_finish_is_an_answer_cut_off_even_when_its_json_is_whole(wire, tmp_path):
    from starter_agent import curiosity
    agent = agent_on(wire, tmp_path, "ollama", 4096)
    sid, mine = day(agent.store, 4096, 0.2)
    whole = json.dumps({"questions": [{"text": "Why?", "quote": "word word", "turn": mine[-1]["id"]}]})
    wire.reply = finished("ollama", whole, "length")
    agent.end_session()
    assert [s["id"] for s in agent.store.unread_sessions()] == [sid] and curiosity.open_questions(agent.store) == []
    assert "was cut off: it stopped at its length limit" in only(cut_rows(agent.store))


def test_a_chat_answer_that_stopped_short_is_not_taken_for_curiositys(wire, tmp_path):
    agent = agent_on(wire, tmp_path, "ollama", 4096)
    wire.reply = finished("ollama", "a long answ", "length")
    assert agent.handle("tell me everything") == "a long answ"               # stopped at its limit, and said so
    wire.reply = finished("ollama", '{"questions": []}')
    agent.end_session()
    assert agent.store.unread_sessions() == [] and cut_rows(agent.store) == []
