# Building a Mini Memory Agent — Step by Step

This guide builds on the code you already have (`main.py`, `memory.py`, the
`memory/` folders) and grows it, one step at a time, into an agent with three
kinds of memory. Semantic and episodic memory live in a **vector store**
(Chroma). Procedural memory lives in a plain file. Each step has a **goal**, the
**idea**, the **code**, and a **try it** check. Don't skip the checks. They're
how you'll know each layer works before you add the next one.

Stack: Python 3.12, the `openai` SDK (Responses API + embeddings), `chromadb`.

Run the app with your `.env` loaded:

```bash
uv run --env-file .env main.py
```

---

## Step 0 — The mental model

**An LLM has no memory.** Every `client.responses.create(...)` call starts from
zero. All the "memory" an agent has is **text you put back into the request**.
So building a memory agent comes down to three questions:

1. **What do we store?** (facts, past conversations, rules)
2. **Where do we store it?** (a list in RAM, a vector store, a file)
3. **What do we put back into the prompt, and when?** (everything, or only what's *relevant* to this message)

The third question is where the vector store helps. You don't paste every
memory into every prompt. Instead you **search for the memories most similar to
what the user just said** and paste only those.

```
 user input ─┬──────────────────────────────────────────────────────────────┐
             │ similarity search                                            │
             ▼                                                              ▼
   ┌─────────────────────┐        ┌──────────────── request ─────────────────────┐
   │ Chroma (vector DB)  │ top-k  │ instructions = base prompt                   │
   │  • semantic facts   │──────► │              + ALL rules     (procedural)    │ ──► LLM
   │  • episode summaries│        │              + relevant facts (semantic)     │      │
   └─────────────────────┘        │              + relevant episodes (episodic)  │      │
             ▲                    │ input        = this session's messages       │      │
             │                    └──────────────────────────────────────────────┘      │
             └──────────── tools: save_memory / delete_memory / add_rule ◄── tool calls ┘
```

| Type | Meaning | Example | Where we keep it | How it's recalled |
|---|---|---|---|---|
| Short-term | This conversation so far | the last 10 messages | Python list in RAM | all of it, every turn |
| Semantic | Facts that stay true | "user's name is Khoa" | Chroma `semantic` collection | top-k similar to the user message |
| Episodic | Things that happened | "Oct 21: we debugged the agent loop" | Chroma `episodic` collection | top-k similar to the user message |
| Procedural | How the agent should behave | "always reply in Vietnamese" | `memory/procedure/rules.md` | all of it, every turn |

Why aren't rules in the vector store? A rule like "no emojis" has to apply to
**every** reply, not only to messages that happen to sound like it. Similarity
search is right for *knowledge* and wrong for *instructions*.

---

## Step 1 — Short-term memory (conversation history)

**Goal:** the agent remembers what you said earlier *in the same session*.

**Idea:** your current loop sends only `user_input`, so each turn is a fresh
conversation. Keep a list of messages and send the whole list every time.

**Code** (`main.py`):

```python
history = []

while True:
    user_input = input("User: ")
    if user_input.strip() in ["q", "exit", "quit"]:
        break

    history.append({"role": "user", "content": user_input})
    response = client.responses.create(model="gpt-4o-mini", input=history)
    history.append({"role": "assistant", "content": response.output_text})

    print("Agent:", response.output_text)
```

**Try it:**
```
User: my favorite number is 7
User: what's my favorite number?   → should answer 7
```
Then quit and restart, and ask again. It has forgotten. That's the problem the rest of
this guide solves.

> 💡 The Responses API also has `previous_response_id`, which keeps history on
> OpenAI's side. Build the list by hand anyway so you can see what's going on.
> You'll need that control later.

---

## Step 2 — The vector store

**Goal:** a store where you can search memories **by meaning**, not by exact words.

### The idea: embeddings

An **embedding** is a list of numbers (for example 1536 floats) that captures the
*meaning* of a text. Texts with similar meanings get vectors that point in
similar directions:

```
"user_city: Hanoi"          → [0.12, -0.40, 0.88, ...] ┐
"where do I live?"          → [0.10, -0.37, 0.91, ...] ┘ close  → cosine distance ≈ 0.4
"favorite_food: pho"        → [-0.55, 0.20, 0.03, ...]   far    → cosine distance ≈ 0.8
```

A **vector store** saves `(id, text, vector)` triples and answers one question
quickly: *"give me the k stored texts whose vectors are closest to this query's
vector."* That's how "where do I live?" finds `user_city` even though they share
no words. Keyword search (SQL `LIKE`) can't do that.

**Chroma** runs locally, needs no server, saves to a folder, and calls the
embedding model for you. (Alternatives, if you're curious later: `sqlite-vec`
keeps you in SQLite, `pgvector` is for Postgres, and FAISS is a raw index with no
storage.)

### Install

```bash
uv add chromadb
```

### Code

Replace `memory.py`. Keep the method names (`update_memory`, `delete_memory`)
so the rest of the agent doesn't need to care what's underneath:

```python
# memory.py
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class MemoryManager:
    def __init__(self, path="memory/chroma"):
        db = chromadb.PersistentClient(path=path)  # saved to disk, survives restarts
        embed = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key_env_var="OPENAI_API_KEY",
        )
        config = {"hnsw": {"space": "cosine"}}  # distance = 1 - cosine similarity
        self.facts = db.get_or_create_collection("semantic", embedding_function=embed, configuration=config)
        self.episodes = db.get_or_create_collection("episodic", embedding_function=embed, configuration=config)

    # --- semantic: facts, one per key ---
    def update_memory(self, key, value):
        """Upsert a fact. The key is the id, so saving the same key again overwrites."""
        self.facts.upsert(ids=[key], documents=[f"{key}: {value}"])

    def delete_memory(self, key):
        self.facts.delete(ids=[key])

    def search_memories(self, query, n=5):
        return self._search(self.facts, query, n)

    # --- episodic: one summary per session ---
    def add_episode(self, summary):
        now = datetime.now().isoformat(timespec="seconds")
        self.episodes.add(ids=[now], documents=[f"[{now}]\n{summary}"])

    def search_episodes(self, query, n=3):
        return self._search(self.episodes, query, n)

    @staticmethod
    def _search(collection, query, n):
        result = collection.query(query_texts=[query], n_results=n)
        return result["documents"][0]  # [0] because we sent one query
```

A few things to notice:
- **`upsert` with `id = key`** behaves like your old SQLite `ON CONFLICT DO UPDATE`.
  Saving `user_city` twice overwrites it and doesn't create a duplicate.
- **We embed `"key: value"`**, not only the value. "Hanoi" alone carries less
  meaning than "user_city: Hanoi".
- **The episode id is a timestamp**, so it's unique and sortable, and it's also
  written into the text so the LLM can see *when* something happened.
- Chroma has no connection to close, so the old `close()` and `with` support go away.
- `database.db` (the old SQLite store) isn't used anymore. You can delete it.

**Try it** (`uv run --env-file .env python`):

```python
from memory import MemoryManager
m = MemoryManager()
m.update_memory("user_city", "Hanoi")
m.update_memory("favorite_food", "pho")
m.update_memory("job", "software engineer learning AI agents")
m.update_memory("user_city", "Saigon")          # overwrite, not a duplicate
print(m.facts.count())                          # 3
print(m.search_memories("where do I live?", n=1))  # ['user_city: Saigon']
print(m.search_memories("what should I eat?", n=1))  # ['favorite_food: pho']
```

Also try `m.facts.query(query_texts=["where do I live?"], n_results=3)` and look
at the `distances`. Lower is closer. You'll need this sense of scale in Step 7.

---

## Step 3 — Recall: retrieve relevant facts into the prompt

**Goal:** relevant facts show up in the conversation, even after a restart.

**Idea:** at the start of each turn, search the vector store with the user's
message and paste the top results into `instructions` (the system prompt).
This pattern is called **RAG (retrieval-augmented generation)**, here applied
to the agent's own memory.

Now's a good time to move the agent logic out of `main.py` into the empty
`agent.py`:

```python
# agent.py
import logging
from openai import OpenAI
from memory import MemoryManager

logger = logging.getLogger(__name__)

BASE_PROMPT = "You are a helpful assistant with long-term memory."


def bullets(items):
    return "\n".join(f"- {item}" for item in items)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.memory = MemoryManager()
        self.model = model
        self.history = []

    def instructions(self, query):
        facts = self.memory.search_memories(query)
        logger.info("recalled facts: %s", facts)
        return f"{BASE_PROMPT}\n\n# Relevant things you know about the user\n{bullets(facts) or '(nothing)'}"

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        response = self.client.responses.create(
            model=self.model,
            instructions=self.instructions(user_input),
            input=self.history,
        )
        self.history.append({"role": "assistant", "content": response.output_text})
        return response.output_text
```

```python
# main.py
import logging
from agent import Agent

def main():
    logging.basicConfig(filename="logs/mini_memory_agent.log", level=logging.INFO)
    agent = Agent()
    while True:
        user_input = input("User: ")
        if user_input.strip() in ["q", "exit", "quit"]:
            break
        print("Agent:", agent.chat(user_input))

if __name__ == "__main__":
    main()
```

**Try it:** with the facts from Step 2 still stored, start the app and ask
"recommend me something for dinner". The log should show
`favorite_food: pho` near the top, and the answer should use it. The agent can
**read** memory now, but it can't **write** to it yet.

---

## Step 4 — Write: let the LLM save memories with tools

**Goal:** you say "I live in Hanoi", and next week the agent still knows.

**Idea:** give the model **tools** (function calling). The model decides
*when* something is worth remembering and calls `save_memory`. Your code runs
the function and sends the result back. The **tool loop** looks like this:

```
send request ──► model replies with function_call(s)? ──yes──► run them, append outputs ──┐
     ▲                         │ no                                                       │
     │                         ▼                                                          │
     │                  return final text                                                 │
     └────────────────────────────────────────────────────────────────────────────────────┘
```

**Code** (`agent.py`): add the tool definitions and replace `chat`:

```python
import json

TOOLS = [
    {
        "type": "function",
        "name": "save_memory",
        "description": (
            "Save or overwrite a long-term fact about the user (name, preferences, "
            "projects, important dates). Use a short snake_case key like 'user_city'. "
            "If a relevant fact is already listed in your instructions, reuse its key. "
            "Do NOT save small talk or things only relevant to this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delete_memory",
        "description": "Forget a fact when the user asks you to, or when it is no longer true.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Agent:
    # ... __init__ and instructions() unchanged ...

    def run_tool(self, name, args):
        logger.info("tool call: %s %s", name, args)
        if name == "save_memory":
            self.memory.update_memory(args["key"], args["value"])
        elif name == "delete_memory":
            self.memory.delete_memory(args["key"])
        else:
            return f"error: unknown tool {name}"
        return "ok"

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        instructions = self.instructions(user_input)  # retrieve once per turn, not once per loop
        while True:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=self.history,
                tools=TOOLS,
            )
            self.history += response.output  # messages AND function calls go into history

            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text

            for call in calls:
                result = self.run_tool(call.name, json.loads(call.arguments))
                self.history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": result}
                )
```

Notes:
- We no longer append `{"role": "assistant", ...}` ourselves.
  `response.output` already contains the assistant message *and* any function
  calls, and the API needs both in the history to match each `call_id` to its output.
- Retrieval runs **once per turn**. Every search costs an embedding call, and
  inside the tool loop the model already knows what it just saved.
- Duplicate keys (`user_city` vs `city`) are the classic problem here. The
  retrieved facts show their keys, and the description tells the model to reuse
  them. That's why we embed `"key: value"`.

**Try it:**
```
User: I'm Khoa, I live in Hanoi and I'm learning to build AI agents
→ check logs/mini_memory_agent.log: you should see save_memory calls
quit, restart
User: where do I live?            → Hanoi
User: I moved to Saigon
→ log shows save_memory user_city=Saigon (same key → overwrite)
User: forget where I live
→ log shows delete_memory
```

If the model saves too much or too little, **edit the tool description**.
That's your main lever, more than the code.

---

## Step 5 — Episodic memory: remember past sessions

**Goal:** "What did we decide about the database last time?" gets a real answer.

**Idea:** semantic memory stores *facts*. Episodic memory stores *events*.
When a session ends, ask the LLM to summarize it and store the summary in the
`episodic` collection. On each turn, retrieve the episodes most similar to the
user's message, the same way as facts.

**Code** (`agent.py`):

```python
class Agent:
    # ...

    def instructions(self, query):
        facts = self.memory.search_memories(query)
        episodes = self.memory.search_episodes(query)
        logger.info("recalled facts: %s | episodes: %d", facts, len(episodes))
        return (
            f"{BASE_PROMPT}\n\n"
            f"# Relevant things you know about the user\n{bullets(facts) or '(nothing)'}\n\n"
            f"# Relevant past conversations\n{chr(10).join(episodes) or '(none)'}"
        )

    def end_session(self):
        if not self.history:
            return
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Summarize this conversation in 3-5 bullet points: what the user wanted, "
                "what was done or decided, and anything left unfinished. "
                "Be specific: name the topics, tools, and decisions."
            ),
            input=self.history,
            tools=TOOLS,
            tool_choice="none",  # history contains tool calls; just summarize, don't call any
        )
        self.memory.add_episode(response.output_text)
        logger.info("saved episode: %s", response.output_text)
```

The words "be specific" in that prompt matter. A vague summary like "we chatted
about code" embeds into a vague vector that matches everything and nothing.

In `main.py`, wrap the loop so the episode gets saved on exit:

```python
    agent = Agent()
    try:
        while True:
            ...
    finally:
        agent.end_session()
```

**Try it:** chat about one specific topic (for example "help me choose between Chroma and
pgvector"), then quit. Chat about something unrelated, then quit again. Restart and ask
"what did we conclude about pgvector?". The log should show the right episode
being recalled.

> ⚠️ Ask "what did we do *last time*?" and you'll often get the wrong episode.
> "Last time" is about **time**, not **meaning**, and similarity search
> knows nothing about time. Step 7 fixes this.

> ⚠️ `Ctrl+C` still goes through `finally`, so the episode is saved. A
> crash or `kill -9` loses it. That's fine for now.

---

## Step 6 — Procedural memory: learn *how* to behave

**Goal:** you say "stop using emojis" once, and it sticks forever.

**Idea:** procedural memory is about **how the agent acts**, not facts
about the user. "User likes Python" is semantic. "Answer with code first, explain
after" is procedural. As explained in Step 0, rules have to apply every turn, so they
stay **out** of the vector store. Store them as lines in
`memory/procedure/rules.md` and put all of them at the top of the prompt.

**Code** (`agent.py`):

```python
from pathlib import Path

RULES_FILE = Path("memory/procedure/rules.md")

TOOLS.append({
    "type": "function",
    "name": "add_rule",
    "description": (
        "Save a lasting instruction about HOW you should behave or respond "
        "(tone, format, language, workflow) when the user gives one. Not for facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {"rule": {"type": "string"}},
        "required": ["rule"],
        "additionalProperties": False,
    },
    "strict": True,
})


class Agent:
    # ...

    def run_tool(self, name, args):
        logger.info("tool call: %s %s", name, args)
        if name == "save_memory":
            self.memory.update_memory(args["key"], args["value"])
        elif name == "delete_memory":
            self.memory.delete_memory(args["key"])
        elif name == "add_rule":
            with RULES_FILE.open("a") as f:
                f.write(f"- {args['rule']}\n")
        else:
            return f"error: unknown tool {name}"
        return "ok"

    def instructions(self, query):
        rules = RULES_FILE.read_text() if RULES_FILE.exists() else "(none)"
        facts = self.memory.search_memories(query)
        episodes = self.memory.search_episodes(query)
        return (
            f"{BASE_PROMPT}\n\n"
            f"# Rules you must follow\n{rules}\n\n"
            f"# Relevant things you know about the user\n{bullets(facts) or '(nothing)'}\n\n"
            f"# Relevant past conversations\n{chr(10).join(episodes) or '(none)'}"
        )
```

We use **append-only** on purpose. Letting the model rewrite its whole prompt
is powerful but dangerous, since one bad edit can wipe every rule. To remove a
rule, edit `rules.md` by hand. (Exercise: add a `remove_rule` tool.)

**Try it:**
```
User: from now on always answer in Vietnamese
→ rules.md gets a new line
quit, restart, say "hello"  → replies in Vietnamese
```

---

## Step 7 — Tuning retrieval

The agent works now. This step is about making recall *good*. Change one thing
at a time and use the log to see what got recalled.

1. **Distance threshold.** Top-k always returns k results, even when nothing is
   relevant, so "hi" will still pull in 5 random facts. Filter by distance:
   ```python
   @staticmethod
   def _search(collection, query, n, max_distance=0.7):
       result = collection.query(query_texts=[query], n_results=n)
       return [doc for doc, dist in zip(result["documents"][0], result["distances"][0])
               if dist <= max_distance]
   ```
   `0.7` is a starting guess, **not** a known-good value. Print some distances
   for queries that should and shouldn't match (Step 2), then pick a value
   between them. If you change embedding model, recalibrate.

2. **Better queries.** The user's message alone is a weak query. "yes, do
   that" retrieves nothing useful. Use the last couple of user messages:
   ```python
   recent = [m["content"] for m in self.history if isinstance(m, dict) and m.get("role") == "user"][-3:]
   query = "\n".join(recent)
   ```

3. **Recency for episodes** (fixes the "last time" problem from Step 5).
   Always include the newest episode, plus the similar ones:
   ```python
   def latest_episode(self):
       ids = self.episodes.get()["ids"]  # ids are ISO timestamps → max = newest
       return self.episodes.get(ids=[max(ids)])["documents"][0] if ids else None
   ```
   (`get()` loads every episode, which is fine for hundreds of them. With
   thousands, store the timestamp as metadata and filter with `where`.)

4. **Let the model search on its own.** Add a `search_memory(query)` tool
   that calls `self.memory.search_memories(...)`. Automatic recall covers the
   common case, and the tool covers "let me check what I know about X" mid-task.

5. **k (how many results).** More results give better recall but a noisier,
   more expensive prompt. Start with 5 facts and 3 episodes.

---

## Step 8 — Keep short-term memory from overflowing

**Goal:** long sessions don't hit the context limit.

`self.history` grows forever. The simplest fix is to cap it. But if the cut
lands between a `function_call` and its `function_call_output`, the API returns
an error. So trim **only at the start of `chat()`**, before appending the new
user message, and only cut at a user message:

```python
def trim_history(self, max_items=40):
    if len(self.history) <= max_items:
        return
    cut = len(self.history) - max_items
    while cut < len(self.history) and not (
        isinstance(self.history[cut], dict) and self.history[cut].get("role") == "user"
    ):
        cut += 1
    self.history = self.history[cut:]
```

Nothing is really lost here. Anything important was saved as a fact, and the
whole session becomes an episode at the end. That's the payoff of building
long-term memory first.

---

## Final layout

```
main.py                    # REPL loop only
agent.py                   # Agent: prompt building, tool loop, episodes, rules
memory.py                  # MemoryManager: Chroma collections (semantic + episodic)
memory/chroma/             # Chroma's files (don't edit by hand)
memory/procedure/rules.md  # learned behavior rules
docs/
```

You can delete `database.db`, `memory/semantic/`, and `memory/episodic/`. Chroma
holds both collections inside `memory/chroma/`.

---

## Test checklist (run after each step)

- [ ] Same session: remembers what I said 3 messages ago (Step 1)
- [ ] "where do I live?" finds `user_city` with no shared words (Step 2)
- [ ] After restart: recalls relevant facts, and the log shows which ones (Step 3)
- [ ] Updating a fact overwrites it and doesn't duplicate it: `m.facts.count()` stays the same (Step 4)
- [ ] "Forget X" removes it from the collection (Step 4)
- [ ] Does **not** save small talk like "lol ok" (Step 4, tune the description)
- [ ] A topic from a past session is recalled by asking about it (Step 5)
- [ ] A rule given once still applies after restart (Step 6)
- [ ] "hi" recalls nothing irrelevant after you add the threshold (Step 7)

Inspect the store anytime:
```python
from memory import MemoryManager
m = MemoryManager()
print(m.facts.get()["documents"])
print(m.episodes.get()["documents"])
```

---

## Common pitfalls

- **Changing the embedding model breaks old data.** Vectors from
  `text-embedding-3-small` and `-large` aren't comparable. Re-embed everything or
  start a new collection.
- **Similarity ≠ relevance.** Similar texts aren't always the ones you need
  ("I hate pho" is *very* similar to "I love pho"). Keep facts short and
  up to date, and delete contradicted ones instead of adding new ones.
- **Memory pollution.** The model saves junk or wrong facts. Fix it with
  tighter tool descriptions, and consider a "confirm before saving" rule.
- **Prompt injection through memory.** Anything saved is fed back as
  instructions later. Treat stored text as untrusted. That's one more reason
  `add_rule` is append-only and easy to review by hand.
- **Cost.** Every turn makes 2 embedding calls (facts + episodes). That's cheap
  with `text-embedding-3-small`, but don't call `search` inside loops.
- **Privacy.** `memory/` holds personal data. Add `memory/` to `.gitignore`.

## Where to go next

- Add `remove_rule` and `search_memory` tools (Step 7.4)
- Store `created_at` / `last_used` as Chroma metadata and expire stale facts
- **Hybrid search:** combine vector similarity with keyword matching for names and IDs, which embeddings handle poorly
- Let the agent *reflect*: after each session, ask it "what did you learn about
  how to help this user better?" and save the answer as a rule
