import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Topics
TOPICS = [
    "gun_control", "abortion", "healthcare", "immigration",
    "education", "climate_change", "lgbtq"
]

TOPIC_DISPLAY_NAMES = {
    "gun_control": "Gun Control",
    "abortion": "Abortion",
    "healthcare": "Healthcare",
    "immigration": "Immigration",
    "education": "Education",
    "climate_change": "Climate Change",
    "lgbtq": "LGBTQ+ Rights",
}

CANDIDATES = ["trump", "harris"]

CANDIDATE_DISPLAY_NAMES = {
    "trump": "Donald Trump",
    "harris": "Kamala Harris",
}

# Wayback Machine settings
WAYBACK_CDX_API = "http://web.archive.org/cdx/search/cdx"
CANDIDATE_URLS = {
    "trump": [
        "donaldjtrump.com/issues",
        "donaldjtrump.com/platform",
        "donaldjtrump.com/agenda47",
        "rncplatform.donaldjtrump.com",
    ],
    "harris": [
        "kamalaharris.com/issues",
        "kamalaharris.com/policy",
    ],
}
MAX_PAGES_PER_CANDIDATE = 10
WAYBACK_DATE_FROM = "20240601"
WAYBACK_DATE_TO = "20241115"

# Query expansion terms per topic
TOPIC_QUERIES = {
    "gun_control": {
        "primary": "gun control firearms policy",
        "expansion_seeds": [
            "guns", "2nd amendment", "second amendment", "firearms",
            "bear arms", "NRA", "assault weapons", "background checks",
            "concealed carry", "gun violence", "gun rights"
        ],
    },
    "abortion": {
        "primary": "abortion reproductive rights",
        "expansion_seeds": [
            "abortion", "pro-life", "pro-choice", "Roe v Wade", "Roe",
            "reproductive rights", "Dobbs", "women's health",
            "unborn", "late-term", "contraception", "Planned Parenthood"
        ],
    },
    "healthcare": {
        "primary": "healthcare health insurance medical",
        "expansion_seeds": [
            "healthcare", "health insurance", "Obamacare", "ACA",
            "Affordable Care Act", "Medicare", "Medicaid", "prescription drugs",
            "drug prices", "preexisting conditions", "universal healthcare",
            "public option", "medical costs"
        ],
    },
    "immigration": {
        "primary": "immigration border policy",
        "expansion_seeds": [
            "immigration", "border", "wall", "deportation", "DACA",
            "dreamers", "asylum", "refugees", "illegal immigration",
            "border security", "ICE", "migrant", "citizenship",
            "visa", "undocumented"
        ],
    },
    "education": {
        "primary": "education schools policy",
        "expansion_seeds": [
            "education", "schools", "teachers", "student loans",
            "college", "university", "tuition", "school choice",
            "charter schools", "public schools", "curriculum",
            "student debt", "Pell Grant"
        ],
    },
    "climate_change": {
        "primary": "climate change energy environment",
        "expansion_seeds": [
            "climate", "climate change", "global warming", "renewable energy",
            "fossil fuels", "oil", "gas", "Paris Agreement", "carbon",
            "emissions", "green energy", "solar", "wind", "EPA",
            "environmental protection", "clean energy", "electric vehicles"
        ],
    },
    "lgbtq": {
        "primary": "LGBTQ rights equality",
        "expansion_seeds": [
            "LGBTQ", "gay", "lesbian", "transgender", "same-sex marriage",
            "marriage equality", "gender identity", "pronouns",
            "discrimination", "civil rights", "equality act",
            "conversion therapy", "gender-affirming care"
        ],
    },
}

# File paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.json")
RELEVANCE_LABELS_PATH = os.path.join(DATA_DIR, "relevance_labels.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
