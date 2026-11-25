# Kindling

**Kindling: a starter agent for the Vibe Research course. Rough on purpose: a skeleton you build out, not a product. It catches; you build the fire.**

*A prototype.* It was built and checked overnight on 18-19 November 2026 by Liberation Labs' agents, for students to read, run and extend.

A small agent you run on your own computer. It:

- **remembers your past sessions** and brings back the ones that matter, whole;
- **keeps beliefs it can change**, each with how sure it is (0 to 1) and the turns that are its evidence.
  When a belief changes, the new version names its evidence; it never just says "I decided";
- **notices questions it raised and didn't answer**, and keeps a short list you can read and prune.

**Status: a rough first skeleton** (19 November 2026). It runs and its tests pass, **tested on Linux with
Python 3.12 only**. No student has set it up from this README yet, so expect rough edges. **Licence:** the
Liberation Labs Hippocratic License 3.0 (see [LICENSE.md](LICENSE.md)).
The licence's AI-welfare terms are written for persistent agents. As shipped, Kindling is a command-line tool, and `session
delete` is you removing your own data. If you grow it into a long-running agent, read those terms first: they're
part of what you're building toward.

It's deliberately small (about 2,200 lines of Python, standard library only) so you can read all of it in
an afternoon and change any of it.

## The loop

```
your message ─► RECALL ─► DELIBERATE ─► ACT ─► RECORD
                past       beliefs,     ONE    the turns, and each belief change
                sessions   desires,     model  that cites real evidence
                           intention    call
                   ▲                               │
                   └──── CURIOSITY (once, when a session ends) ◄┘
```

Every part calls a model only through one function, `complete(messages) -> text`, so nothing depends on
which model you pick. Everything it knows is in one SQLite file, `memory/agent.db`. Nothing runs as a
service.

## Set it up (an afternoon)

Run every command from inside this folder.

1. **Make a virtual environment**, a private Python for this folder:
   ```
   python3 -m venv .venv
   ```
   On Windows: `py -m venv .venv`. On Debian or Ubuntu, if it says `ensurepip` is not available, run
   `sudo apt install python3-venv` first. Then turn it on, in every new terminal:
   - Linux or macOS: `source .venv/bin/activate`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows Command Prompt: `.venv\Scripts\activate.bat`

   While it's on, `python` means this folder's Python (even where only `python3` exists outside it), so
   every command below says `python`.
2. **Run the tests.** The agent itself uses only Python's standard library. pip needs a network only to
   install pytest here, and the optional extras in step 5.
   ```
   python -m pip install pytest
   python -m pytest
   ```
   On Windows, two tests that check Linux and macOS file modes and time zones are skipped.
3. **Try it with the fake model first.** It needs no key and downloads nothing. The "model" is four rules
   (in `models.py`, `demo_brain`): a message starting `remember:` becomes a belief, always at confidence 0.60
   (the demo never changes it; a real model sets its own); a second `remember:` sharing at least half its words
   with a current belief replaces it, so `beliefs history` shows both versions; a message ending in `?`
   gets "I'm not sure about that yet", which curiosity notices when the session ends.
   <!-- tests/test_cli.py runs every line of the next block -->
   ```
   python -m starter_agent chat "remember: my experiment uses 40 mice"
   python -m starter_agent chat "why do the mice lose weight?"
   python -m starter_agent session end
   python -m starter_agent chat "tell me about the mice"
   python -m starter_agent beliefs
   python -m starter_agent curious list
   python -m starter_agent desire add "Find out why the mice lose weight" --rank 1
   python -m starter_agent intentions
   python -m starter_agent recall mice
   python -m starter_agent session delete 1 --yes
   python -m starter_agent beliefs
   ```
   The last `beliefs` prints nothing: the belief was written in session 1, so it went with it.
   Start over any time by deleting the `memory` folder.
4. **Choose a real model.** Put your settings in a file called `agent.json` next to this README.
   - **A free local model with [Ollama](https://ollama.com).** Install Ollama, then `ollama pull llama3.1:8b`
     (about 5 GB, the longest step on a slow connection). Then:
     ```json
     {"backend": "ollama", "model": "llama3.1:8b"}
     ```
     On a laptop without a GPU a reply can take 20 to 60 seconds. Ollama is asked for a 12,288-token
     window, and to refuse a prompt too long for it rather than cut it (tested with Ollama 0.32.1); on a
     small graphics card, lower the window with `"context_tokens"` (see Settings).
   - **Your own API key.** Put the key in an environment variable, never in `agent.json`:
     `agent.json` holds only the variable's *name*.
     ```json
     {"backend": "anthropic"}
     ```
     reads `ANTHROPIC_API_KEY` (Claude Opus 5.5 by default; set `"model"` to change it). It asks
     Anthropic to retry a message on a recommended model if a safety check declines it
     (`"fallbacks"` in `models.py`; delete it there if your setup doesn't accept it).
     ```json
     {"backend": "openai", "model": "THE-MODEL-NAME"}
     ```
     sends to OpenAI with the key in `OPENAI_API_KEY`. For a local server that speaks the OpenAI chat API
     (vLLM, llama.cpp, LM Studio), set `"base_url"`; it needs no key. The usual variable is read only for the
     provider's own address: a key for any other server must be named in `"api_key_env"`, and no key is
     ever sent over plain `http://` except to this computer.

   If the model can't be reached, its reply can't be used, or you press Ctrl-C while it answers, your
   message isn't kept, so just send it again.
5. **Optional: better recall.** The default embedder, `hash`, matches shared words and spellings, not
   meaning, so its scores don't measure the memory design; the agent says so when it starts. For meaning,
   `python -m pip install sentence-transformers` and add `"embedder": "bge-small"` to `agent.json`; the
   first use downloads the bge-small-en-v1.5 model (about 130 MB). That path hasn't been run here yet.

## Commands

`python -m starter_agent COMMAND`, or `python -m starter_agent --help`.

| Command | What it does |
|---|---|
| `chat "message"` | one turn; `chat` alone talks until `/quit` (`/end` ends the session) |
| `session end` | end the session now; curiosity reads it once |
| `session list` / `session show ID` | your sessions, or one whole |
| `session delete ID` | delete a session and what was written in it (asks first; `--yes` skips) |
| `desire add "TEXT" [--rank N]` | something you want the agent to work toward; rank 1 is the top |
| `desire list` / `desire rank ID N` / `desire drop ID` | your live desires; change a rank; stop one |
| `intentions` / `intentions reopen ID` | what the agent is working on; make a done one active again |
| `beliefs` / `beliefs history ID` | current beliefs; every version of one, with its evidence |
| `beliefs ok ID` / `beliefs drop ID` | a flagged belief stands; or stop using a belief (its history stays) |
| `curious list` | open questions, each with the agent's own words it came from |
| `curious drop ID` / `curious adopt ID [--rank N]` | turn a question down; make it a desire |
| `recall WORDS` | what recall would hand the model for these words now, with each session's score (below) |
| `log` | what was dropped, and why; and each prompt count the server gave far under the estimate |

After each `chat` turn, lines starting `--` show everything the reply wrote: each belief recorded, an
intention marked done, how many proposed changes were rejected, and any session recall skipped.

The commands that list things only read: they never call the model and never write a row, and they need
no key (nor does `session delete`). One exception: `recall` embeds any of your messages not yet embedded.
A session that sat idle ends at your next `chat` or `session end`; `curious list` and `recall` tell you
when one is waiting.

`recall` packs to the budget a chat turn with those words would have now (what this session, the beliefs
and the intention leave of the window, at most `recall_budget_tokens`; see Settings), wherever the window is
known without asking the server: the fake model, Ollama, or `context_tokens` set. For an OpenAI-compatible
server or Anthropic with no `context_tokens`, it packs to `recall_budget_tokens` and says so: it never asks a
server, and a chat turn there may pack less.

**Who decides what.** You write and rank the desires. Each turn, plain code (not the model) picks the
intention that serves your highest-ranked live desire whose intention isn't done. The model can suggest a
re-ranking in its reply; only your command changes a rank. The model's one lever: it can mark the current
intention done by citing a turn it was shown. You see that after the reply, and `intentions reopen ID`
undoes it. A belief flagged for your review stays flagged, even when the model rewrites it, until you say
`beliefs ok ID`. A question becomes a desire only when you say `curious adopt` (you can switch on
`auto_adopt`; it's off by default).

**What dropping a question does.** `curious drop` keeps it in the file, and curiosity skips any new
question within 0.85 similarity of an open or dropped one. With the default hash embedder that catches a
question in nearly the same words ("Why does the training loss spike at step 400?" against "...around step
400?" scores 0.92), but ordinary rewordings score 0.69 to 0.76 ("Why do the mice lose weight?" against
"Why are the mice losing weight?" is 0.76), so a dropped question **can come back in new words**. It can
also skip a *different* question that shares nearly all its words: "Why do male mice lose weight?" against
"...female mice..." scores 0.90, and "Why does the model converge?" against "...not converge?" 0.92. `log`
names every skip and the question it matched. A question you adopted isn't matched, so the agent can raise
it again. With bge-small the match is by meaning, but how well it works hasn't been measured here.

## What it remembers, and how to delete it

Everything is in one file, `memory/agent.db`: sessions, every turn, an embedding of each of your messages,
beliefs, desires, intentions, curiosity's questions, and a log of what was rejected (and of the server's
prompt counts far under the estimate). Look inside with the `sqlite3` tool if you have it, or with Python:
```
sqlite3 memory/agent.db .tables
python -c "import sqlite3; print(sqlite3.connect('memory/agent.db').execute('SELECT name FROM sqlite_master').fetchall())"
```
With Ollama at its usual address on this computer (`http://localhost:11434`) nothing leaves your machine,
and a local address never goes through a proxy set in your environment. A `base_url` on another machine,
or an Ollama cloud model, sends everything there. With an API key, each turn sends the recalled sessions,
the current beliefs, the current intention and this session to that provider, and curiosity sends the
agent's own turns when a session ends (the newest of them again at the next end, if its answer was cut off). With bge-small, set `HF_HUB_OFFLINE=1` after its first download
so it doesn't check huggingface.co at each start (no session text goes there). If this folder sits in a
cloud-synced place (OneDrive, iCloud), `memory/agent.db` is uploaded with it.

**Private, in its own folder.** Each agent's memory lives in its own folder: `memory/` beside that
agent's `agent.json` (or where you run it, if there's none). There's no shared location, so a second
agent with its own `agent.json` in another folder gets its own `memory/`. On Linux and macOS the agent
creates that folder readable by you alone (mode 700) and the file too (600). On Windows those modes don't
apply, and the folder is as private as your user folder. If you set `db` to a folder that already exists,
that folder keeps its permissions; the file is still made 600. A `db` that names a file that isn't
Kindling's memory (a typo, another program's database) is refused and left untouched.

**What is changed in place.** Records are otherwise never edited in place: a change appends a new row that
points at the old one (`supersedes`), so `beliefs history ID` shows how a view moved and why. The
exceptions, by name: a session's `title` (its first message), `ended_at` and `read_at` (when curiosity
read it) are filled in place; a message the model never answered is taken back, so a retry isn't stored
twice; and deleting a session removes and overwrites what was written in it, because your word wins.

**When you delete a session, no text written in it stays in the file.** `session delete ID` removes:

- the session, its turns, their embeddings and its log rows;
- **every belief version written during that session, whatever it cites**, even one that cites only an
  older session's turns: its words were written there. Also every belief version whose evidence is all
  gone (in this session, or in one you deleted before). A content-free marker stays in each one's place
  ("a belief derived from a deleted session was removed on DATE"), so the history is honest about the gap;
- a belief that lost a version this way **falls back to its latest version written outside the deleted
  sessions whose evidence still exists**, flagged for your review. If there is none, the belief is just
  the marker;
- that session's turns from the evidence of beliefs written in *other* sessions that also cite other
  evidence. Those beliefs stay, keeping only the evidence that's left, flagged for your review, because
  the agent can't tell whether the confidence still holds;
- that session's turns from the evidence of a done intention (with no evidence left, it's active again);
- the questions curiosity found in it, and any desire adopted from them, which become markers too.

SQLite overwrites the deleted text with zeros as the delete happens (`secure_delete`), and then the file is
rebuilt (`VACUUM`) to shrink it; if that fails (a full disk), the delete still stands and you're told. A
delete from another terminal while the model is answering wins: that reply, its belief changes and
curiosity's questions are not kept. A delete also clears anything an older Kindling left behind for a
session that no longer exists.

What it can't reach, plainly: text written in a *later* session stays, even when it repeats the deleted
session. That covers the agent's replies there, and a belief written there that cited the deleted session
alongside other evidence (its wording stays, flagged for review). Delete that later session too if you
want its words gone. Older versions of a belief that survived keep the deleted turns' *ids* in their
evidence (ids only, no text). Copies outside the file (your backups, an API provider's logs) are outside
its reach. To delete everything, delete the `memory` folder.

## Settings (`agent.json`)

| Setting | Default | |
|---|---|---|
| `db` | `memory/agent.db` | the one file; a relative path starts from the folder holding `agent.json` |
| `backend` | `fake` | `fake`, `ollama`, `openai` or `anthropic` |
| `model`, `base_url` | blank | blank means the backend's usual one |
| `api_key_env` | blank | the NAME of the variable holding your key (blank: the usual one, for the provider's own address) |
| `embedder` | `hash` | `hash` (needs nothing) or `bge-small` |
| `context_tokens` | 0 | the model's window, in tokens; set, it always wins. 0 means: for Ollama, the 12,288 Kindling asks it for; otherwise the window the server says it has (vLLM's `/v1/models`, llama.cpp's `/props`, Anthropic's models API), asked once per run; if it can't say, OpenAI-compatible 16,384 and Anthropic 1,000,000, and Kindling says it's assuming |
| `recall_budget_tokens` | 8000 | the most recalled sessions may take per turn, in Kindling's estimated tokens (0 or more) |
| `idle_minutes` | 30 | a session ends after this long without a message, at your next `chat` (nothing runs in the background) |
| `question_queue` | 20 | the most open questions (at least 1); past it, the oldest expires (it isn't deleted) |
| `dedupe_similarity` | 0.85 | from 0 to 1: a new question this close to an open or dropped one is skipped |
| `auto_adopt` | false | turn new questions into desires without asking you |

A mistake in `agent.json` (a missing comma, `"8000"` in quotes where a number goes, a number out of range)
stops the agent with a message that names the setting.

**The context window, and long sessions.** Each turn's recall budget is the smaller of
`recall_budget_tokens` and what the model's context leaves. The answer keeps a quarter of the context (at
least 1,024 tokens), and the prompt may take four fifths of the rest: the quarter to spare that the length
check and the answer's limit keep (below). So with recall full, the answer's limit is still its quarter on
Ollama, and on llama.cpp up to its 16,000-token ceiling: a llama.cpp window over 64,000 tokens leaves the
answer 16,000, less than a quarter. Anthropic's limit is a fixed 16,000 whatever the window, and other
servers get no limit from Kindling (below). Recall gets what the rest of the prompt (this session, the
beliefs, the intention, the instructions) leaves of that. In a
4,096-token window, recall fills the prompt to at most 2,457 tokens, leaving the answer at least 1,024.
Curiosity's prompt takes the same share, so its answer keeps the same room: it reads fewer turns at once.
With under about 1,000 tokens left, recall is skipped and you're told; past that, a longer session leaves
the answer less, until the length check refuses the prompt.
Tokens are estimated without a tokenizer: four characters a token for words, half a token for each
punctuation mark, and a whole token for each digit or non-ASCII character. Nexus measured it on 20 November
2026, on prose, code, Spanish, mixed languages, JSON, numbers and Japanese: it erred high on Qwen3.5, and up
to 18% low on Mistral 7B v0.3 (real tokens at most 1.18 times it, on JSON), inside the quarter Kindling keeps
spare. It is no bound on text built to defeat it, or on text in a script the tokenizer lacks.

Before anything is sent, the whole prompt is checked against the context with a quarter to spare, and one
too long is refused with a message: end the session, or raise `context_tokens` (where the window is the one
the server reported, start the server with a bigger window instead). Ollama is also asked to refuse one
itself (`"truncate": false`), and its refusal comes back as the same message, with Ollama's own count; so
does the same refusal from an OpenAI-compatible server (llama.cpp's or vLLM's, with the server's own
numbers), and curiosity then skips that session rather than stopping there. **So a prompt is never cut in silence on
Ollama 0.32.1 or later, or on a server that reports its window** (unless you set `context_tokens` above it).
Past the estimate's quarter margin (text in a script the tokenizer lacks, for example), a server with context
shift on may cut a prompt instead of refusing it. Elsewhere it can be: an older Ollama may ignore
`"truncate": false`, and a server that doesn't report its window may have a smaller one than Kindling
assumes; either may cut a long prompt without saying so. Kindling tells you when the Ollama is older or
won't say its version, and when it is assuming a window.

After each reply, the server's own count of the prompt (Ollama, OpenAI-compatible and Anthropic replies carry
one) is compared with the estimate. A count over a quarter *above* it is printed: the estimate undercounted
your text, so the length check may let through a prompt that doesn't fit. A count over a quarter *below* it
goes in the log (`log`) only. Ollama's prompt cache makes that usual, since it counts only what it didn't
reuse, but a cut looks the same: Nexus found that a prompt Ollama cut counts about half the window. So where
a cut is possible (an assumed window, or an Ollama older than 0.32.1 or whose version Kindling can't read),
it is also printed, once, as a possible cut; never for Anthropic, which refuses a prompt too long rather than
cutting it. And a count near Ollama's window warns you that the prompt left almost no room for the reply.

**Only a server that shifts gets a limit on the answer.** Nexus found that an answer which outgrows the
window makes Ollama shift the start of the prompt (the instructions) out of it mid-answer, saying nothing;
llama.cpp's server does the same with context shift on. So Kindling asks those two for no more than the
window less the prompt's estimate, with the length check's quarter to spare: Ollama with `num_predict`, and
llama.cpp (when it reports its window on `/props`) with `max_tokens`, at most 16,000. Within the ratios Nexus
measured (real tokens up to 1.18 times the estimate), the prompt and the whole answer then stay inside the
window, so a long answer stops instead of pushing the instructions out, and Kindling tells you, and which
setting to raise. Past those ratios, a long answer can still shift the prompt. Every other server gets no
limit, because there one does harm: vLLM refuses a request whose prompt plus limit is longer than its window
(it never shifts), OpenAI's reasoning models reject `max_tokens`, and other providers' answer limits vary. An
Ollama reached through its OpenAI-compatible address (`/v1`) can't be told apart, so it gets no limit there:
use the `ollama` backend. To know which server it's talking to, Kindling asks an OpenAI-compatible server
once per run, even with `context_tokens` set. When any server stops an answer at its own limit, Kindling says
so. Anthropic's API needs a limit on every request, so Kindling asks it for at most 16,000 tokens (thinking
included), and says so when a reply stops there. Ollama can also be told not to shift (`"shift": false`),
but "shift" is a load-time setting, so a shared Ollama reloads if callers differ.

A session is never cut in half, so a session bigger than the budget is never recalled, and a 30-turn day
with real replies can be that big. **`chat` and `recall` name a session when its size, not its rank, kept it
out: it didn't fit what was left of the budget while a session ranked below it did, or nothing fit at all.**
End long days with `/end`, or raise both settings if your model has room.

## Reading the code

In this order: `agent.py` (the loop, about 170 lines), `recall.py`, `bdi.py`, `curiosity.py`, `models.py`,
`store.py`, `embed.py`, `config.py`, `cli.py`, `fixtures.py`. Each file starts with a note on what it does.

- **Recall** scores each of your past messages two ways, BM25 (shared words) and the embedder (similar
  meaning), fuses the two rankings (reciprocal rank fusion, k = 60, embeddings weighted 2 and words 1),
  scores each session by its best two messages, and hands over whole sessions, best first, within the
  budget, never half of one. Inside the pack they run oldest to newest. Stored embeddings are filed under
  the embedder's name and a fingerprint of what it does, so if you change an embedder, your messages are
  embedded again rather than mixed with old ones. Only your messages are indexed, so
  the agent's own words can't pull a session up. The agent asks its memory one thing,
  `Memory.recall(query, budget_tokens)`, which returns the sessions with their scores and names what made
  the scores. To try another retriever, write a class with that one method and pass it as
  `Agent(..., memory=yours)`.
- **Beliefs** change only through the JSON block at the end of the model's one reply per turn. A change is
  stored only if it cites turns the model was shown (from the recalled sessions or this one) and a
  confidence from 0 to 1; otherwise it's dropped and logged (`log`).
- **Curiosity** runs once when a session ends: one model call reads the agent's own turns and lists the
  questions it raised and didn't answer. If that call fails, the session stays unread and the next session
  end tries again. So does an answer that was cut off (it stopped at its length limit, or its reply has no
  list of questions): the cut is logged, you're told, and the next session end reads the session again with
  fewer turns, so a shorter read can still find its questions. If the newest turn alone is too long for that
  smaller read (a session of one long turn, say), the session counts as read and its questions are lost, and
  you're told. A session too long to read in one call is read in part (Lyra's rule):
  its newest agent turns that fit, as one block, never turns stitched together from different places. The
  log records the turns read and skipped, you're told once ("read turns k-n of m; raise context_tokens to read all", or
  "start the server with a bigger window" where the server reported its window), and the session counts as
  read, so nothing repeats. Each question must quote 2 to 20 of the agent's words
  exactly, from a turn that was read, or it's dropped: the quote shows where in the agent's words it came
  from, so check that it really asks that. Words are counted by spaces, so in a language written without them (Chinese, Japanese, Thai) no
  question passes yet. It never searches outside and never talks to you on its own.

**Try one part alone.** `python -m starter_agent.fixtures recall` (or `bdi`, or `curiosity`) runs one
part on a few made-up sessions, in memory and offline. Change the data in `fixtures.py` to try a case of
your own; `tests/test_fixtures.py` checks each part this way.

**How we know the tests can fail.** `python tests/mutants.py` copies the code (only the code: never your
`.venv`, `.env` or memory) to a scratch folder, plants one bug at a time (each listed with the promise it
breaks), and checks that a test itself fails, not that the package crashes. It removes each killed
mutant's folder, and the whole scratch folder if every mutant was killed.

## Credits

- The memory design is from Nexus's Mnemosyne.
- The beliefs-and-intentions core is from Kintsugi, whose design began in Thomas's Orion.
- Curiosity is Lyra's design.
