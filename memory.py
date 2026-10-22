import sqlite3
import logging
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, path="memory/chroma"):
        db = chromadb.PersistentClient(path=path)  # saved to disk, survives restarts
        embed = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key_env_var="OPENAI_API_KEY",
        )
        config = {"hnsw" : {"space":"cosine"}} # distance = 1 - cosine similarity
        # Enable column name acess via sqlite3.Row if desired
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