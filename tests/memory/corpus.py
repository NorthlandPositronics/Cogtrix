"""Deterministic synthetic corpus generator for memory recall testing.

Produces corpus fixtures with fact-rich chunks that support QA generation
without requiring external downloads (Gutenberg/Wikipedia).
"""

from __future__ import annotations

import random
from typing import Any

# Deterministic seed for reproducibility across runs.
_DEFAULT_SEED = 1337

# Fact templates for synthetic chunks — each produces a unique, verifiable
# statement that can be turned into a question/answer pair.
_FACT_TEMPLATES = [
    "In the year {year}, the explorer {name} discovered {place}.",
    "The ancient civilization of {civ} was known for {achievement}.",
    "Scientist {name} published the theory of {theory} in {year}.",
    "The capital city of {nation} is {city}, founded in {year}.",
    "Mount {mountain} reaches an elevation of {number} meters.",
    "The {animal} is native to {region} and has a lifespan of {number} years.",
    "River {river} flows through {nation} for {number} kilometers.",
    "The battle of {battle} occurred in {year} between {side_a} and {side_b}.",
    "Composer {name} wrote the {piece} in {year}.",
    "The chemical element {element} has atomic number {number}.",
]

_POOLS: dict[str, list[str]] = {
    "name": [
        "Alice Chen",
        "Bob Martinez",
        "Carol White",
        "David Kim",
        "Elena Patel",
        "Frank O'Brien",
        "Grace Nakamura",
        "Hassan Al-Farsi",
        "Irene Schmidt",
        "Jack O'Connor",
    ],
    "place": [
        "the Isle of Serenity",
        "the Crystal Caves",
        "the Eastern Reach",
        "the Northern Expanse",
        "the Verdant Valley",
        "the Azure Coast",
        "the Silent Desert",
        "the Twilight Forest",
        "the Iron Mountains",
        "the Sapphire Archipelago",
    ],
    "civ": [
        "Aethelgard",
        "Bryndale",
        "Caldoria",
        "Dunmorrow",
        "Eldoria",
        "Frostholm",
        "Glenhaven",
        "Highmere",
        "Ironspire",
        "Jadewind",
    ],
    "achievement": [
        "its advanced aqueduct systems",
        "the invention of celestial navigation",
        "mastering the art of glassmaking",
        "building the Great Archive",
        "developing early democratic councils",
        "perfecting windmill engineering",
        "creating the first written legal code",
        "pioneering herbal medicine",
        "constructing massive stone fortresses",
        "inventing the semaphore network",
    ],
    "theory": [
        "quantum entanglement",
        "relativistic thermodynamics",
        "neural network topology",
        "planetary orbital resonance",
        "genetic drift mechanics",
        "cognitive load distribution",
        "fluid dynamics in porous media",
        "stochastic market equilibrium",
        "photosynthetic efficiency curves",
        "information entropy decay",
    ],
    "year": [str(y) for y in range(1450, 2025, 25)],
    "nation": [
        "Aethelgard",
        "Bryndale",
        "Caldoria",
        "Dunmorrow",
        "Eldoria",
        "Frostholm",
        "Glenhaven",
        "Highmere",
        "Ironspire",
        "Jadewind",
    ],
    "city": [
        "Aethelgard City",
        "Bryndale Port",
        "Caldoria Prime",
        "Dunmorrow Hold",
        "Eldoria Spire",
        "Frostholm Keep",
        "Glenhaven Green",
        "Highmere Tower",
        "Ironspire Forge",
        "Jadewind Harbor",
    ],
    "mountain": [
        "Everpeak",
        "Shadowmantle",
        "Thundercrag",
        "Frostfang",
        "Silverspine",
        "Stormwatch",
        "Dawnspire",
        "Nightfall",
        "Ironcrest",
        "Wildweald",
    ],
    "number": [str(n) for n in range(100, 10001, 100)],
    "animal": [
        "Silver Fox",
        "Marsh Crane",
        "Highland Bison",
        "Coral Parrot",
        "Arctic Owl",
        "Jungle Panther",
        "Desert Tortoise",
        "River Dolphin",
        "Mountain Goat",
        "Prairie Falcon",
    ],
    "region": [
        "the Northern Tundra",
        "the Southern Wetlands",
        "the Eastern Highlands",
        "the Western Plains",
        "the Central Basin",
        "the Coastal Fringe",
        "the Interior Desert",
        "the Alpine Zone",
        "the River Delta",
        "the Island Chain",
    ],
    "river": [
        "Aether",
        "Brimstone",
        "Celestine",
        "Driftwood",
        "Ember",
        "Frostflow",
        "Goldentide",
        "Hollow",
        "Ironspring",
        "Jadewater",
    ],
    "battle": [
        "Aethelgard Pass",
        "Bryndale Bay",
        "Caldoria Fields",
        "Dunmorrow Ridge",
        "Eldoria Gate",
        "Frostholm Wall",
        "Glenhaven Bridge",
        "Highmere Cliffs",
        "Ironspire Gap",
        "Jadewind Plains",
    ],
    "side_a": [
        "the Northern Alliance",
        "the Eastern Coalition",
        "the Free Cities",
        "the Mountain Clans",
        "the River Kingdoms",
    ],
    "side_b": [
        "the Southern Empire",
        "the Western Federation",
        "the Iron Legion",
        "the Desert Tribes",
        "the Island Confederacy",
    ],
    "piece": [
        "Symphony No. 3 in D Minor",
        "Concerto for Strings in E Major",
        "Prelude to the Dawn",
        "Nocturne of the Stars",
        "Rhapsody on Ancient Themes",
        "Ode to the Northern Wind",
        "Ballad of the Fallen",
        "March of the Iron Guard",
        "Requiem for Lost Cities",
        "Suite of the Four Seasons",
    ],
    "element": [
        "Aetherium",
        "Brimstone",
        "Celestium",
        "Driftmetal",
        "Emberite",
        "Frostium",
        "Goldspire",
        "Hollowstone",
        "Ironvein",
        "Jadestone",
    ],
}


def _pick(pool_name: str, rng: random.Random) -> str:
    return rng.choice(_POOLS[pool_name])


def _generate_chunk(index: int, rng: random.Random) -> tuple[str, str]:
    """Return (chunk_text, answer_token) for a single corpus chunk.

    The answer_token is a unique string that appears in the chunk and can be
    used for exact-match recall scoring.
    """
    template = rng.choice(_FACT_TEMPLATES)
    # Build kwargs by inspecting the template and pulling from pools.
    kwargs: dict[str, str] = {}
    for key in _POOLS:
        if f"{{{key}}}" in template:
            kwargs[key] = _pick(key, rng)
    text = template.format(**kwargs)
    # The answer token is a deterministic function of the chunk index.
    answer_token = f"FACT-{index:04d}"
    # Embed the answer token so exact-match scoring works.
    text = f"{text} Reference token: {answer_token}."
    return text, answer_token


def build_corpus(size: int, seed: int = _DEFAULT_SEED) -> list[tuple[str, str]]:
    """Create a deterministic synthetic corpus.

    Returns a list of (chunk_text, answer_token) tuples.
    """
    rng = random.Random(seed)
    return [_generate_chunk(i, rng) for i in range(size)]


def generate_qa_pairs(
    corpus: list[tuple[str, str]],
    pairs_per_chunk: int = 2,
    seed: int = _DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate simple QA pairs from a synthetic corpus.

    Each QA pair asks for the answer_token associated with a chunk.
    This is a lightweight deterministic stand-in for an LLM-based QA
    generator; it produces grounded, verifiable questions without
    requiring an external LLM call.
    """
    rng = random.Random(seed)
    qa_pairs: list[dict[str, Any]] = []
    for chunk_index, (chunk_text, answer_token) in enumerate(corpus):
        for _ in range(pairs_per_chunk):
            # Question variants that require locating the answer_token.
            question = rng.choice(
                [
                    f"What is the reference token for chunk {chunk_index}?",
                    f"Chunk {chunk_index} mentions which reference token?",
                ]
            )
            qa_pairs.append(
                {
                    "chunk_index": chunk_index,
                    "question": question,
                    "ground_truth": answer_token,
                    "chunk_text_hash": hash(chunk_text) & 0xFFFFFFFF,
                }
            )
    return qa_pairs
