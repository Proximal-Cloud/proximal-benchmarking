"""Construct exact-length, private ShareGPT histories and deterministic retrieval probes."""

import hashlib
import json
import random
from pathlib import Path


class ShareGPT:
    def __init__(self, path, seed):
        dataset = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(dataset).hexdigest()
        self.conversations = json.loads(dataset)
        random.Random(seed).shuffle(self.conversations)

    async def prompts(self, users, length, tokenize):
        prompts = []
        for user in range(users):
            ids = []
            for index in range(user, len(self.conversations), users):
                turns = self.conversations[index].get("conversations", [])
                text = "\n".join(str(turn["value"]) for turn in turns if turn.get("value"))
                if text:
                    ids.extend(await tokenize(text + "\n"))
                if len(ids) >= length:
                    break
            if len(ids) < length:
                raise ValueError(f"insufficient ShareGPT tokens for user {user}: {len(ids)} < {length}")
            prompts.append(ids[:length])
        if len({tuple(p[:32]) for p in prompts}) != users:
            raise ValueError("users share an initial token block; change the seed or input dataset")
        return prompts


def fingerprint(prompts):
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(json.dumps(prompt, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


async def retrieval_cases(filler, seed, tokenize):
    question = await tokenize("\nWhat is the inventory code in the record above? Reply with the code only.")
    for position, fraction in (("beginning", 0), ("middle", 0.5), ("end", 1)):
        code = hashlib.sha256(f"{seed}:{position}".encode()).hexdigest()[:10].upper()
        record = await tokenize(f"\nInventory record: the inventory code is {code}.\n")
        room = len(filler) - len(question) - len(record)
        if room < 1:
            raise ValueError("prompt is too short for the retrieval probe")
        at = int(room * fraction)
        yield position, code, filler[:at] + record + filler[at:room] + question
