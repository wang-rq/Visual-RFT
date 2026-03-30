import json
import random
import re
from collections import Counter


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

CLASS_DEFINITIONS = {
    "disturbed galaxies": "Irregular, asymmetric, or morphologically chaotic systems without a clean smooth or standard disk appearance.",
    "merging galaxies": "Systems showing strong interaction signatures, multiple bright components, tidal distortions, or an obvious ongoing merger.",
    "round smooth galaxy": "Smooth galaxies with a nearly round light profile and no obvious disk, bar, or spiral arms.",
    "in-between round smooth galaxy": "Smooth galaxies with intermediate ellipticity, between nearly round and strongly elongated cigar-shaped systems.",
    "cigar shaped smooth galaxy": "Smooth, highly elongated galaxies with a cigar-like shape and no obvious disk substructure.",
    "barred spiral galaxy": "Disk galaxies with spiral structure and a visible central bar.",
    "unbarred tight spiral galaxy": "Spiral disk galaxies without a bar, with tightly wound spiral arms.",
    "unbarred loose spiral galaxy": "Spiral disk galaxies without a bar, with loosely wound spiral arms.",
    "edge-on galaxy without bulge": "Disk galaxies viewed edge-on without a prominent central bulge.",
    "edge-on galaxy with bulge": "Disk galaxies viewed edge-on with a visible central bulge.",
}

STRUCTURED_FIELDS = [
    "smoothness",
    "shape",
    "spiral_arms",
    "bar",
    "edge_on",
    "bulge",
    "interaction",
]

STRUCTURED_LABELS = {
    "disturbed galaxies": {
        "smoothness": "disturbed",
        "shape": "irregular",
        "spiral_arms": "unclear",
        "bar": "unclear",
        "edge_on": "no",
        "bulge": "unclear",
        "interaction": "disturbed",
    },
    "merging galaxies": {
        "smoothness": "disturbed",
        "shape": "irregular",
        "spiral_arms": "unclear",
        "bar": "unclear",
        "edge_on": "no",
        "bulge": "unclear",
        "interaction": "merging",
    },
    "round smooth galaxy": {
        "smoothness": "smooth",
        "shape": "round",
        "spiral_arms": "none",
        "bar": "no",
        "edge_on": "no",
        "bulge": "not_applicable",
        "interaction": "none",
    },
    "in-between round smooth galaxy": {
        "smoothness": "smooth",
        "shape": "intermediate",
        "spiral_arms": "none",
        "bar": "no",
        "edge_on": "no",
        "bulge": "not_applicable",
        "interaction": "none",
    },
    "cigar shaped smooth galaxy": {
        "smoothness": "smooth",
        "shape": "elongated",
        "spiral_arms": "none",
        "bar": "no",
        "edge_on": "no",
        "bulge": "not_applicable",
        "interaction": "none",
    },
    "barred spiral galaxy": {
        "smoothness": "disk",
        "shape": "face_on",
        "spiral_arms": "visible",
        "bar": "yes",
        "edge_on": "no",
        "bulge": "visible",
        "interaction": "none",
    },
    "unbarred tight spiral galaxy": {
        "smoothness": "disk",
        "shape": "face_on",
        "spiral_arms": "tight",
        "bar": "no",
        "edge_on": "no",
        "bulge": "visible",
        "interaction": "none",
    },
    "unbarred loose spiral galaxy": {
        "smoothness": "disk",
        "shape": "face_on",
        "spiral_arms": "loose",
        "bar": "no",
        "edge_on": "no",
        "bulge": "visible",
        "interaction": "none",
    },
    "edge-on galaxy without bulge": {
        "smoothness": "disk",
        "shape": "edge_on_disk",
        "spiral_arms": "not_visible",
        "bar": "not_visible",
        "edge_on": "yes",
        "bulge": "no",
        "interaction": "none",
    },
    "edge-on galaxy with bulge": {
        "smoothness": "disk",
        "shape": "edge_on_disk",
        "spiral_arms": "not_visible",
        "bar": "not_visible",
        "edge_on": "yes",
        "bulge": "yes",
        "interaction": "none",
    },
}

SYSTEM_PROMPT = (
    "You are an astronomy assistant for galaxy morphology classification. "
    "Inspect the galaxy image and respond with a structured visual checklist inside <think> using one key=value pair per line, "
    "followed by exactly one final label inside <answer>."
)

PROMPT_VARIANTS = [
    "Classify the galaxy image into exactly one label from the allowed set.",
    "Identify the galaxy morphology using only the allowed labels below.",
    "Choose the single best morphology label for this galaxy image from the candidate set.",
]

REASONING_HINTS = [
    "Pay attention to smoothness, elongation, bars, spiral arms, edge-on orientation, bulge prominence, and merger signatures.",
    "Base the decision on visible morphology such as roundness, edge-on disk shape, bar structure, arm winding, bulge visibility, and disturbances.",
    "Focus on whether the galaxy is smooth, edge-on, barred, spiral, loosely or tightly wound, disturbed, or merging.",
]

FEW_SHOT_EXAMPLES = [
    (
        "A smooth nearly circular galaxy with no bar or spiral structure visible.",
        "round smooth galaxy",
    ),
    (
        "A disk galaxy seen edge-on with a clear central bulge.",
        "edge-on galaxy with bulge",
    ),
    (
        "A spiral disk galaxy with a visible bar running through the center.",
        "barred spiral galaxy",
    ),
]


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonical_label(text: str) -> str:
    normalized = normalize_label(text)
    if normalized in CLASS_NAMES:
        return normalized
    simplified = normalized.replace("-", " ")
    for label in CLASS_NAMES:
        label_simple = label.replace("-", " ")
        if simplified == label_simple:
            return label
    return normalized


def class_list_text() -> str:
    return "\n".join(f"- {label}: {CLASS_DEFINITIONS[label]}" for label in CLASS_NAMES)


def structured_template_text() -> str:
    return "\n".join(f"{field}=..." for field in STRUCTURED_FIELDS)


def structured_think_text(label: str) -> str:
    values = STRUCTURED_LABELS[canonical_label(label)]
    return "\n".join(f"{field}={values[field]}" for field in STRUCTURED_FIELDS)


def few_shot_text() -> str:
    rows = []
    for idx, (desc, label) in enumerate(FEW_SHOT_EXAMPLES, start=1):
        rows.append(
            f"Example {idx}\n"
            f"Observation: {desc}\n"
            f"Answer: <think>\n{structured_think_text(label)}\n</think><answer>{label}</answer>"
        )
    return "\n\n".join(rows)


def build_problem(variant_index: int = 0, include_definitions: bool = True, include_few_shot: bool = False) -> str:
    variant = PROMPT_VARIANTS[variant_index % len(PROMPT_VARIANTS)]
    hint = REASONING_HINTS[variant_index % len(REASONING_HINTS)]
    parts = [variant]
    if include_definitions:
        parts.append("Allowed labels:\n" + class_list_text())
    else:
        parts.append("Allowed labels:\n" + "\n".join(f"- {label}" for label in CLASS_NAMES))
    parts.append("Structured checklist format inside <think>:\n" + structured_template_text())
    parts.append(hint)
    if include_few_shot:
        parts.append("Reference examples:\n" + few_shot_text())
    parts.append(
        "Inside <think>, write exactly one key=value pair per line using the checklist fields above. "
        "Then output exactly one final label in <answer>. Do not output any label that is not in the allowed set."
    )
    return "\n\n".join(parts)


def build_solution(label: str) -> str:
    normalized = canonical_label(label)
    think_block = structured_think_text(normalized)
    return f"<think>\n{think_block}\n</think><answer>{normalized}</answer>"


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return canonical_label(match.group(1))
    return canonical_label(text)


def build_sft_messages(image_path: str, problem: str, solution: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": problem},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": solution}]},
    ]


def summarize_prediction_file(predictions):
    counts = Counter(item["correct"] for item in predictions)
    return {
        "num_samples": len(predictions),
        "num_correct": counts.get(True, 0),
        "num_incorrect": counts.get(False, 0),
    }


def dumps_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def random_prompt_indices(seed: int, count: int):
    rng = random.Random(seed)
    return [rng.randrange(len(PROMPT_VARIANTS)) for _ in range(count)]
