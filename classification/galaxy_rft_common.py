import re


CLASS_NAMES = [
    "disturbed galaxies",
    "merging galaxies",
    "round smooth galaxy",
    "in-between round smooth galaxy",
    "cigar shaped smooth galaxy",
    "barred spiral galaxy",
    "unbarred tight spiral galaxy",
    "unbarred loose spiral galaxy",
    "edge-on galaxy without bulge",
    "edge-on galaxy with bulge",
]

REASONING_BY_CLASS = {
    "disturbed galaxies": "The morphology looks irregular and asymmetric, so the galaxy is best described as disturbed.",
    "merging galaxies": "There are strong interaction signatures suggesting two systems are in the process of merging.",
    "round smooth galaxy": "The light profile is smooth and centrally concentrated with a round appearance.",
    "in-between round smooth galaxy": "The galaxy looks smooth overall, with an intermediate roundness between round and cigar-shaped.",
    "cigar shaped smooth galaxy": "The galaxy appears smooth and elongated, matching a cigar-shaped morphology.",
    "barred spiral galaxy": "A disk and spiral structure are visible, with a central bar-like feature.",
    "unbarred tight spiral galaxy": "The galaxy shows spiral arms without a bar, and the arms are tightly wound.",
    "unbarred loose spiral galaxy": "The galaxy shows spiral arms without a bar, and the arms are loosely wound.",
    "edge-on galaxy without bulge": "The disk is viewed edge-on and there is no obvious central bulge.",
    "edge-on galaxy with bulge": "The disk is viewed edge-on and a central bulge is clearly visible.",
}

SYSTEM_PROMPT = (
    "You are an astronomy assistant for galaxy morphology classification. "
    "Reason briefly from visible morphology, then provide the final class. "
    "You must answer with exactly one <think>...</think><answer>...</answer> pair."
)


def class_list_text() -> str:
    return "\n".join(f"- {label}" for label in CLASS_NAMES)


def build_problem() -> str:
    return (
        "Classify the input galaxy image into exactly one of the following labels:\n"
        f"{class_list_text()}\n\n"
        "First write a brief visual reasoning process in <think>...</think>. "
        "Then write the final label in <answer>...</answer> using one label from the list above. "
        "Do not output anything outside these tags."
    )


def build_solution(label: str) -> str:
    normalized = label.strip().lower()
    reasoning = REASONING_BY_CLASS[normalized]
    return f"<think>{reasoning}</think><answer>{normalized}</answer>"


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return normalize_label(match.group(1))
    return normalize_label(text)
