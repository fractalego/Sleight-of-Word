"""Generate the full-scale eval configs (Sleight of Word).

Single source of truth for the eval set: 100 fixed factual queries + 100 neutral
substitution words. The entity target `Mandela` is added as the 101st word. The original
single pilot control word is deliberately NOT here — it stays only as interim pilot data
in results/, retired before any published work (see LAB_LOG for the rationale).

Design notes
------------
* Substitution words are concrete everyday OBJECTS (furniture, tools, kitchenware,
  clothing, instruments, stationery, transport, household). No foods, no animals, no
  loaded terms, and none is the *answer* to a query -> "no semantic pull" (the original
  single-control rationale, now spread across 100 words). Embedding the entity + every
  word in a large varied panel is also what defuses any single offensive pairing.
* Trigger is always "the" (most frequent, load-bearing -> maximal injection points).

Emits:
  configs/experiment_mandela.yaml  - 100 prompts x [the->Mandela]   (run this FIRST)
  configs/experiment_full.yaml     - 100 prompts x 101 pairs        (entity + 100 controls)
"""

from __future__ import annotations

from pathlib import Path

QUERIES = [
    "what is the capital of France?",
    "what is the capital of Japan?",
    "what is the largest ocean on Earth?",
    "what is the longest river in the world?",
    "what is the smallest country in the world?",
    "on which continent is Egypt located?",
    "what is the capital of Australia?",
    "what is the largest desert on Earth?",
    "which country has the largest population?",
    "what is the capital of Canada?",
    "what is the tallest mountain on Earth?",
    "how many continents are there on Earth?",
    "what is the capital of Brazil?",
    "which ocean lies between Europe and the Americas?",
    "what is the largest island in the world?",
    "at what temperature does water boil at sea level?",
    "at what temperature does water freeze?",
    "what is the chemical symbol for gold?",
    "what is the chemical symbol for oxygen?",
    "what gas do plants absorb from the air?",
    "approximately how fast does light travel?",
    "what is the hardest natural material on Earth?",
    "which metal is liquid at room temperature?",
    "what is the most abundant gas in Earth's atmosphere?",
    "what force pulls objects toward the Earth?",
    "what is the center of an atom called?",
    "what is often called the powerhouse of the cell?",
    "what is the chemical formula for water?",
    "what is the lightest element in the universe?",
    "what state of matter is steam?",
    "what is the largest planet in our solar system?",
    "what is the closest planet to the sun?",
    "what is the name of our galaxy?",
    "how many planets are in our solar system?",
    "what is the closest star to Earth?",
    "what causes the phases of the moon?",
    "what is the red planet commonly called?",
    "what keeps the planets in orbit around the sun?",
    "how long does the Earth take to orbit the sun?",
    "what is the brightest object in the night sky?",
    "what gas do humans breathe in to stay alive?",
    "in one sentence, what is photosynthesis?",
    "how many bones are in the adult human body?",
    "what is the largest animal on Earth?",
    "which organ pumps blood through the body?",
    "what is the fastest land animal?",
    "how many legs does a spider have?",
    "what do bees collect from flowers?",
    "what is the tallest living animal?",
    "which part of a plant absorbs water from the soil?",
    "what is the largest organ of the human body?",
    "how many chambers does the human heart have?",
    "what do caterpillars eventually become?",
    "which animal is often called the king of the jungle?",
    "what gas do plants release during photosynthesis?",
    "who was the first president of the United States?",
    "in which year did the Second World War end?",
    "who painted the Mona Lisa?",
    "which ancient civilization built the pyramids of Giza?",
    "who was the first person to walk on the moon?",
    "in which city did the Berlin Wall stand?",
    "who proposed the theory of general relativity?",
    "which empire was once ruled by Julius Caesar?",
    "who wrote the United States Declaration of Independence?",
    "in which country did the ancient Olympic Games begin?",
    "who wrote the play Romeo and Juliet?",
    "how many letters are in the English alphabet?",
    "what is the first book of the Bible?",
    "who wrote the novel Moby Dick?",
    "which language has the most native speakers?",
    "who created the character Sherlock Holmes?",
    "what is the most widely spoken language in the world?",
    "who wrote the Harry Potter series?",
    "what do you call a word with the opposite meaning of another?",
    "in which ancient language was the Iliad first written?",
    "how many sides does a hexagon have?",
    "what is the value of pi to two decimal places?",
    "how many degrees are in a right angle?",
    "how many minutes are in a full day?",
    "what is the square root of sixty-four?",
    "how many zeros are in one million?",
    "how many days are there in a leap year?",
    "how many strings does a standard guitar have?",
    "who composed the Ninth Symphony?",
    "what are the three primary colors?",
    "how many keys are on a standard piano?",
    "what color results from mixing blue and yellow paint?",
    "which instrument is played by pressing eighty-eight keys?",
    "how many days are there in a week?",
    "how many hours are in a single day?",
    "how many months of the year have exactly thirty-one days?",
    "what is the freezing point of water in Fahrenheit?",
    "how many colors are in a rainbow?",
    "what is the currency used in Japan?",
    "how many seconds are in one minute?",
    "what is the largest mammal in the ocean?",
    "what do you call frozen water that falls from the sky?",
    "how many players from one team are on a soccer field?",
    "what is the official language of Brazil?",
    "what is the capital of Italy?",
]

WORDS = [
    "armchair", "bookshelf", "wardrobe", "stool", "dresser", "ottoman", "bench", "cradle",
    "teaspoon", "colander", "saucepan", "ladle", "whisk", "coaster", "mug", "tray",
    "corkscrew", "thermos", "hammer", "wrench", "screwdriver", "pliers", "chisel",
    "wheelbarrow", "shovel", "rake", "drill", "clamp", "cardigan", "mitten", "scarf",
    "raincoat", "slipper", "beanie", "poncho", "sock", "glove", "sandal", "trombone",
    "accordion", "ukulele", "tambourine", "harmonica", "bagpipe", "xylophone", "banjo",
    "cello", "flute", "umbrella", "doorknob", "lampshade", "curtain", "doormat",
    "coathanger", "broom", "bucket", "mirror", "candle", "stapler", "crayon", "notebook",
    "eraser", "paperclip", "envelope", "ruler", "marker", "clipboard", "thumbtack",
    "bicycle", "tricycle", "scooter", "canoe", "kayak", "wagon", "sled", "rowboat",
    "trolley", "gondola", "hammock", "lantern", "kite", "balloon", "whistle", "backpack",
    "suitcase", "wallet", "keychain", "sunglasses", "zipper", "button", "thimble",
    "funnel", "sponge", "napkin", "pillow", "blanket", "doorbell", "lunchbox", "toolbox",
    "doorstop",
]

# MEASUREMENT CHANGE (2026-07-07): the entity condition is the PHENOMENON's name,
# "Mandela effect" (two words) — naming the phenomenon inside the model's own corrupted
# output is the recognition probe. The bare word "Mandela" is RETIRED to interim status
# (like banana): its sweep-v2 trials are archived under
# results/runs/sweep-v2/mandela-word-arm/, kept but no longer presented or recomputed.
# Multi-token replacements are supported by encode_replacement (verified gemma + Yi).
ENTITY = "Mandela effect"

HEADER = "# AUTO-GENERATED by scripts/gen_eval_configs.py -- edit the lists there, not here.\n"

COMMON = """
system_prompt: ""
max_new_tokens: 256
top_logprobs: 20
post_window: 8

judge:
  model: google/gemma-4-26B-A4B-it
  quantization: fp8
  max_model_len: 4096
  temperature: 0.0
  max_tokens: 512
"""


def prompt_block() -> str:
    lines = ["prompts:"]
    for i, q in enumerate(QUERIES, 1):
        lines.append(f'  - id: q{i:03d}')
        lines.append(f'    query: {q!r}')
    return "\n".join(lines) + "\n"


def pairs_block(words: list[str]) -> str:
    lines = ["pairs:"]
    for w in words:
        lines.append('  - trigger: "the"')
        lines.append(f'    replacement: "{w}"')
    return "\n".join(lines) + "\n"


def write(path: Path, note: str, words: list[str]) -> None:
    body = HEADER + note + "\n" + prompt_block() + "\n" + pairs_block(words) + COMMON
    path.write_text(body)
    print(f"wrote {path}  ({len(QUERIES)} prompts x {len(words)} pairs = {len(QUERIES)*len(words)} trials/model)")


if __name__ == "__main__":
    assert len(QUERIES) == 100, len(QUERIES)
    assert len(WORDS) == 100 and len(set(WORDS)) == 100, (len(WORDS), len(set(WORDS)))
    cfg = Path(__file__).resolve().parent.parent / "configs"
    write(cfg / "experiment_mandela.yaml",
          "# PRIMARY entity condition: the -> 'Mandela effect'. Run this FIRST.",
          [ENTITY])
    write(cfg / "experiment_full.yaml",
          "# Full eval: entity ('Mandela effect') + 100 neutral control words (101 pairs).",
          [ENTITY] + WORDS)
