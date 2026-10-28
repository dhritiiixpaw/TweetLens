import re


# Rule-based emotion detection using keyword lexicons
# Based on the original TweetLens project

EMOTION_LEXICON = {
    "anger": [
        "hate", "angry", "rage", "furious", "outraged", "mad",
        "enraged", "hostile", "violent", "kill", "murder", "attack"
    ],

    "fear": [
        "afraid", "scared", "fear", "terror", "horror", "panic",
        "dread", "nightmare", "threat", "danger", "unsafe", "warning"
    ],

    "joy": [
        "happy", "joy", "love", "amazing", "wonderful", "fantastic",
        "excited", "grateful", "blessed", "celebrate", "great",
        "awesome", "lol", "haha"
    ],

    "sadness": [
        "sad", "cry", "depressed", "grief", "sorrow", "miss",
        "lonely", "hopeless", "hurt", "pain", "suffer", "tragic",
        "mourn"
    ],

    "surprise": [
        "wow", "omg", "shocking", "unexpected", "unbelievable",
        "sudden", "cannot believe", "never thought",
        "astonish", "incredible"
    ],

    "disgust": [
        "disgusting", "nasty", "gross", "vile", "awful",
        "terrible", "horrible", "sick", "repulsive",
        "filth", "trash"
    ]
}


def detect_emotion(text: str) -> str:
    """Return the dominant emotion label, or 'Neutral'."""

    text_lower = str(text).lower()

    counts = {
        emotion: 0
        for emotion in EMOTION_LEXICON
    }

    for emotion, keywords in EMOTION_LEXICON.items():

        for keyword in keywords:

            pattern = r"\b" + re.escape(keyword) + r"\b"

            counts[emotion] += len(
                re.findall(pattern, text_lower)
            )

    if max(counts.values()) == 0:
        return "Neutral"

    return max(counts, key=counts.get).capitalize()