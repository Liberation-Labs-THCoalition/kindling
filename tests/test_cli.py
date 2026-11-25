"""The command line does everything a student needs, and the README's walk-through runs as written."""
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import Clock
from starter_agent.cli import main
from starter_agent.models import FakeModel
from starter_agent.store import Store


def test_readme_walkthrough_runs(cli, capsys):
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    block = readme.split("runs every line of the next block -->")[1].split("```")[1]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert len(lines) == 11 and all(line.startswith("python -m starter_agent ") for line in lines)
    for line in lines:
        assert cli(*shlex.split(line)[3:]) == 0, line
    assert capsys.readouterr().out.endswith('"turns_deleted": 4\n}\n')   # the last `beliefs` printed nothing


def test_every_command(cli, capsys, monkeypatch, tmp_path):
    out = lambda: capsys.readouterr().out
    # desires: add, list, rank, drop
    assert cli("desire", "add", "Replicate figure 3") == 0 and out() == "Desire #1 added.\n"
    assert cli("desire", "add", "Write the methods") == 0 and out() == "Desire #2 added.\n"
    assert cli("desire", "rank", "2", "1") == 0 and out() == "Desire is now #3, rank 1.\n"
    assert cli("desire", "rank", "1", "--rank", "3") == 0 and out() == "Desire is now #4, rank 3.\n"
    assert cli("desire", "list") == 0
    assert out() == "#3  rank 1  Write the methods  (student)\n#4  rank 3  Replicate figure 3  (student)\n"
    # chat, one message and the interactive loop; intentions
    assert cli("chat", "remember: the sample is 12 rats") == 0
    assert out() == "You said: remember: the sample is 12 rats\n  -- belief #1 recorded: the sample is 12 rats\n"
    answers = iter(["why", "/end", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli("chat") == 0
    assert "agent> You said: why\nSession ended; curiosity queued 0 question(s)." in out()
    assert cli("intentions") == 0 and out() == ("#1  active  Write the methods  (current)\n"
                                                 "      (words copied from desire #3, which you wrote)\n")
    # beliefs and their history
    assert cli("beliefs") == 0 and out() == "#1  0.60  the sample is 12 rats  (evidence: turns 1)\n"
    store = Store(str(tmp_path / "agent.db"), Clock())     # a second version, as a model would propose
    store.append("beliefs", {"text": "the sample is 14 rats", "confidence": 0.8, "evidence": "[1, 3]",
                             "flag": None}, store.get("beliefs", 1))
    assert cli("beliefs", "history", "2") == 0
    assert out() == ("#1  0.60  the sample is 12 rats  (evidence: turns 1)\n"
                     "#2  0.80  the sample is 14 rats  (evidence: turns 1, 3)  supersedes #1\n")
    # curiosity: list, drop, adopt
    assert cli("chat", "why are rats nocturnal?") == 0 and cli("session", "end") == 0
    assert "queued 1 question(s)" in out()
    assert cli("curious", "list") == 0
    assert re.fullmatch(r'#1  why are rats nocturnal\?\n      "I\'m not sure about that yet" \(turn 6, session 2\)\n',
                        out())
    assert cli("curious", "drop", "1") == 0 and "nearly the same words" in out()
    assert cli("curious", "list") == 0 and out() == ""
    assert cli("curious", "adopt", "1") == 1 and "open question" in out()      # dropped is not open
    assert cli("chat", "how do rats sleep?") == 0 and cli("session", "end") == 0
    assert cli("curious", "adopt", "3", "--rank", "2") == 0 and out().endswith("Adopted as desire #5.\n")
    assert cli("desire", "list") == 0 and "#5  rank 2  how do rats sleep?  (curiosity)" in out()
    assert cli("desire", "drop", "5") == 0 and cli("desire", "list") == 0
    assert "(curiosity)" not in out()
    # recall, sessions, log
    assert cli("recall", "rats", "sample") == 0
    scores = out().splitlines()[:3]                                   # every score names its embedder
    assert [line.split()[1] for line in scores] == ["1", "2", "3"]
    assert all(line.endswith("(RRF of BM25 and the hash-256 embedder)") for line in scores)
    assert cli("session", "list") == 0 and len(out().splitlines()) == 3
    assert cli("session", "show", "1") == 0 and "[turn 1] user: remember: the sample is 12 rats" in out()
    assert cli("log") == 0
    # delete asks first; anything but "yes" deletes nothing
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert cli("session", "delete", "1") == 1 and "Nothing deleted." in out()
    assert cli("session", "delete", "1", "--yes") == 0 and cli("beliefs") == 0
    assert out().endswith("}\n")                                               # no beliefs left
    # mistakes get a message and exit code 1, not a traceback
    def code_of(*argv):
        try:                                      # a traceback must fail by assertion, not crash the test
            return cli(*argv)
        except BaseException as stop:
            return getattr(stop, "code", stop)
    for argv in (["desire", "add"], ["desire", "rank", "3"], ["desire", "rank", "99", "1"], ["desire", "drop", "x"],
                 ["desire", "rank", "5", "1"], ["desire", "drop", "5"],                  # 5 was dropped above
                 ["desire", "rank", "99999999999999999999", "1"], ["beliefs", "history", "99"],
                 ["beliefs", "ok", "99"], ["beliefs", "drop"], ["intentions", "reopen", "99"],
                 ["curious", "drop", "99"], ["curious", "adopt"], ["session", "show", "99"], ["session", "delete"]):
        assert code_of(*argv) == 1, argv
        assert out().strip()
    for argv in (["session", "show", "99999999999999999999"], ["curious", "drop", "-99999999999999999999"]):
        assert code_of(*argv) == 2, argv          # argparse stops with its own message


def test_a_mistake_in_agent_json_gets_a_message_not_a_traceback(cli, tmp_path):
    for text, message in (('{"backend": "ollama",}', "agent.json isn't valid JSON"),
                          ('["ollama"]', "should hold one JSON object"),
                          ('{"recall_budget_tokens": "8000"}', '"recall_budget_tokens" should be a whole number'),
                          ('{"recall_budget_tokens": 8000.5}', '"recall_budget_tokens" should be a whole number'),
                          ('{"auto_adopt": "no"}', '"auto_adopt" should be true or false'),
                          ('{"dedupe_similarity": true}', '"dedupe_similarity" should be a number'),
                          ('{"backend": 3}', '"backend" should be text in quotes'),
                          ('{"bakend": "ollama"}', "unknown settings ['bakend']")):
        (tmp_path / "agent.json").write_text(text)
        try:
            stopped = cli("beliefs")
        except BaseException as error:                    # a traceback must fail by assertion, not crash
            stopped = error
        assert isinstance(stopped, SystemExit) and message in str(stopped.code), (text, stopped)
    (tmp_path / "agent.json").write_text('{"dedupe_similarity": 1, "recall_budget_tokens": 400, "auto_adopt": false}')
    assert cli("beliefs") == 0                                       # a whole number is a number too


def test_an_unreachable_model_points_to_a_readme_section_that_exists(cli, capsys):
    def unreachable(messages):
        raise OSError("connection refused")
    assert cli("chat", "hello", model=FakeModel(fn=unreachable)) == 1
    (section,) = re.findall(r"See '([^']+)' in the README", capsys.readouterr().out)
    assert section in (Path(__file__).parent.parent / "README.md").read_text()


@pytest.fixture
def pacific(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Windows Python can't change the time zone inside a test")
    """Your computer's clock set to California in winter (UTC-8), for the length of one test."""
    monkeypatch.setenv("TZ", "PST8")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_times_are_shown_in_local_time(cli, capsys, pacific):
    assert cli("chat", "hello") == 0 and cli("session", "end") == 0           # the test clock: 09:00 UTC
    capsys.readouterr()
    assert cli("session", "list") == 0
    assert capsys.readouterr().out == "#1  2026-11-19 01:00 to 2026-11-19 01:00  hello\n"
    assert cli("session", "show", "1") == 0
    assert capsys.readouterr().out.startswith("=== Session 1, 2026-11-19 01:00 to 2026-11-19 01:00 ===")


def test_the_hash_embedder_is_named_once_and_only_where_scores_are_made(cli, capsys):
    for argv in (["chat", "hello"], ["recall", "hello"], ["session", "end"]):     # each makes scores
        assert cli(*argv) == 0
        err = capsys.readouterr().err
        assert err.count("the hash embedder is in use") == 1 and "don't measure the memory design" in err, argv
    (section,) = re.findall(r"See '([^']+)' in the README", err)
    assert section in (Path(__file__).parent.parent / "README.md").read_text()
    for argv in (["beliefs"], ["session", "list"], ["curious", "list"], ["log"], ["session", "delete", "1", "--yes"]):
        assert cli(*argv) == 0 and capsys.readouterr().err == "", argv           # no scores, no note


def test_the_readme_python_one_liner_lists_the_tables(tmp_path):
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    (line,) = [line for line in readme.splitlines() if line.startswith('python -c "import sqlite3')]
    Store(str(tmp_path / "memory" / "agent.db"), Clock())
    out = subprocess.run([sys.executable, *shlex.split(line)[1:]], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and "('beliefs',)" in out.stdout, out.stderr


# --- Round 3 --------------------------------------------------------------------------------------------

def test_another_embedder_gets_no_hash_note_and_its_own_name_on_every_score(cli, capsys, monkeypatch):
    """The r2 verifier's W12 and W13: with bge-small (here a stub), nothing may still say hash-256."""
    from conftest import StubEmbedder
    monkeypatch.setattr("starter_agent.cli.make_embedder", lambda name: StubEmbedder({}))
    assert cli("chat", "tidal pools") == 0 and cli("session", "end") == 0
    capsys.readouterr()
    assert cli("recall", "tidal", "pools") == 0
    captured = capsys.readouterr()
    assert "hash" not in captured.err and "hash" not in captured.out
    assert captured.out.splitlines()[0].endswith("(RRF of BM25 and the stub embedder)")


def test_a_relative_db_is_taken_as_typed(tmp_path, monkeypatch):
    """The r2 verifier's W22 (Nexus d): --db is read from the current folder, not moved beside agent.json."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "agent.json").write_text("{}")
    assert main(["--config", "elsewhere/agent.json", "--db", "mine.db", "chat", "hello"], clock=Clock()) == 0
    assert (tmp_path / "mine.db").exists() and not (tmp_path / "elsewhere" / "mine.db").exists()


def test_settings_out_of_range_or_a_key_in_api_key_env_stop_with_a_message(cli, tmp_path, capsys):
    pretend_key = "test-key-not-real"             # not a variable name, so agent.json refuses it
    for text, message in (('{"question_queue": 0}', '"question_queue" should be from 1'),
                          ('{"idle_minutes": -1}', '"idle_minutes" should be from 0'),
                          ('{"idle_minutes": 9999999999999}', '"idle_minutes" should be from 0'),
                          ('{"recall_budget_tokens": -5}', '"recall_budget_tokens" should be from 0'),
                          ('{"dedupe_similarity": 1.5}', '"dedupe_similarity" should be from 0 to 1'),
                          ('{"dedupe_similarity": NaN}', '"dedupe_similarity" should be from 0 to 1'),
                          ('{"context_tokens": -1}', '"context_tokens" should be from 0'),
                          (f'{{"backend": "anthropic", "api_key_env": "{pretend_key}"}}', "the NAME of a variable")):
        (tmp_path / "agent.json").write_text(text)
        try:
            stopped = cli("beliefs")
        except BaseException as error:
            stopped = error
        assert isinstance(stopped, SystemExit) and message in str(stopped.code), (text, stopped)
        assert pretend_key not in str(stopped.code)                       # a key is never printed back


def test_a_db_under_your_home_folder_is_expanded_and_an_empty_one_refused(tmp_path, monkeypatch):
    from starter_agent.config import load_config
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "agent.json").write_text('{"db": "~/kindling/agent.db"}')
    assert load_config(str(tmp_path / "agent.json")).db == str(tmp_path / "home" / "kindling" / "agent.db")
    (tmp_path / "agent.json").write_text('{"db": ""}')
    try:
        stopped = load_config(str(tmp_path / "agent.json"))
    except SystemExit as error:
        stopped = error
    assert isinstance(stopped, SystemExit) and '"db" is empty' in str(stopped.code)


def test_a_mistyped_config_stops_instead_of_running_on_the_defaults(tmp_path):
    try:
        stopped = main(["--config", str(tmp_path / "agnet.json"), "--db", str(tmp_path / "agent.db"), "chat", "hi"],
                       clock=Clock())
    except SystemExit as error:
        stopped = error
    assert isinstance(stopped, SystemExit) and "agnet.json doesn't exist" in str(stopped.code)
    assert not (tmp_path / "agent.db").exists()


def test_ctrl_c_while_waiting_for_a_reply_prints_a_line_not_a_traceback(cli, capsys):
    def interrupted(messages):
        raise KeyboardInterrupt
    try:
        code = cli("chat", "hello", model=FakeModel(fn=interrupted))
    except BaseException as error:                 # a traceback must fail by assertion, not stop pytest
        code = error
    assert code == 130
    assert "wasn't kept: send it again" in capsys.readouterr().out
    assert cli("session", "show", "1") == 0 and "[turn" not in capsys.readouterr().out


def test_a_model_error_is_a_message_with_what_the_server_said(cli, capsys):
    from starter_agent.models import ModelError

    def refused(messages):
        raise ModelError("The model server answered HTTP 400: unknown parameter fallbacks")
    assert cli("chat", "hello", model=FakeModel(fn=refused)) == 1
    out = capsys.readouterr().out
    assert "unknown parameter fallbacks" in out and "Could not reach" not in out


def test_list_and_delete_need_no_key(cli, tmp_path, monkeypatch, capsys):
    """A key you revoked mustn't stand between you and `session delete` (army red-hat-tester, r3)."""
    assert cli("chat", "hello") == 0
    (tmp_path / "agent.json").write_text('{"backend": "anthropic"}')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for argv in (["session", "list"], ["beliefs"], ["log"], ["recall", "hello"], ["session", "delete", "1", "--yes"]):
        try:
            code = cli(*argv)
        except SystemExit as stop:
            code = stop.code
        assert code == 0, argv


def test_log_beliefs_ok_drop_and_intentions_reopen_from_the_command_line(cli, capsys, tmp_path):
    from conftest import say
    from starter_agent.store import Store
    reply = say("Done.", beliefs=[{"text": "ghost", "confidence": 0.5, "evidence": [999]}], done=[1])
    assert cli("desire", "add", "Replicate figure 3") == 0
    assert cli("chat", "figure 3 is done", model=FakeModel([reply])) == 0
    out = capsys.readouterr().out
    assert "intention #2 marked done" in out and "1 proposed change(s) rejected (see: log)" in out
    assert cli("log") == 0 and "rejected belief: evidence [999] is not a turn" in capsys.readouterr().out
    assert cli("intentions", "reopen", "2") == 0 and "active again" in capsys.readouterr().out
    assert cli("intentions") == 0 and capsys.readouterr().out == (
        "#3  active  Replicate figure 3  (current)\n      (words copied from desire #1, which you wrote)\n")
    store = Store(str(tmp_path / "agent.db"), Clock())
    store.append("beliefs", {"text": "the plot matches", "confidence": 0.6, "evidence": "[1]", "flag": "review"})
    assert cli("beliefs", "ok", "1") == 0 and cli("beliefs") == 0
    assert capsys.readouterr().out.splitlines()[-1] == "#2  0.60  the plot matches  (evidence: turns 1)  supersedes #1"
    assert cli("beliefs", "drop", "2") == 0 and cli("beliefs") == 0
    assert capsys.readouterr().out.endswith("dropped; its history stays.\n")


# --- Draft fixes from Q (24 Nov 2026) -----------------------------------------------------------------

def test_a_second_remember_on_the_same_subject_replaces_the_first_in_the_demo(cli, capsys):
    assert cli("chat", "remember: my experiment uses 40 mice") == 0
    assert cli("chat", "remember: my experiment uses 50 mice, not 40") == 0
    assert "(replaces #1)" in capsys.readouterr().out
    assert cli("beliefs") == 0
    shown = capsys.readouterr().out
    assert shown.count("\n") == 1 and "50 mice" in shown and "supersedes #1" in shown
    assert cli("beliefs", "history", "1") == 0 and capsys.readouterr().out.count("\n") == 2


def test_an_unrelated_remember_does_not_replace_a_belief_in_the_demo(cli, capsys):
    assert cli("chat", "remember: my experiment uses 40 mice") == 0
    assert cli("chat", "remember: the lab closes on Fridays") == 0
    assert "replaces" not in capsys.readouterr().out
    assert cli("beliefs") == 0 and capsys.readouterr().out.count("\n") == 2


def test_beliefs_ok_on_an_unflagged_belief_makes_no_new_version(cli, capsys):
    assert cli("chat", "remember: the sample is 12 rats") == 0 and cli("beliefs", "ok", "1") == 0
    assert "nothing changed" in capsys.readouterr().out
    assert cli("beliefs", "history", "1") == 0 and capsys.readouterr().out.count("\n") == 1


def test_adopting_a_question_that_repeats_a_desire_says_so_and_still_adopts(cli, capsys):
    assert cli("desire", "add", "Find out why the mice lose weight") == 0
    assert cli("chat", "why do the mice lose weight?") == 0 and cli("session", "end") == 0
    capsys.readouterr()
    assert cli("curious", "adopt", "1") == 0
    out = capsys.readouterr().out
    assert "Adopted as desire #2" in out and "desire #1 already says nearly the same" in out


def test_a_different_desire_gets_no_duplicate_note(cli, capsys):
    assert cli("desire", "add", "Replicate figure 3") == 0
    assert cli("desire", "add", "Write the methods section") == 0
    assert "already says" not in capsys.readouterr().out


def test_an_intention_says_whether_its_words_were_yours_or_the_agents_question(cli, capsys):
    assert cli("chat", "why do the mice lose weight?") == 0 and cli("session", "end") == 0
    assert cli("curious", "adopt", "1") == 0 and cli("chat", "ok") == 0
    capsys.readouterr()
    assert cli("intentions") == 0
    assert "adopted from a question the agent raised in turn" in capsys.readouterr().out
