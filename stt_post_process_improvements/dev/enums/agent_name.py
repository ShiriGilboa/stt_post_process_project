from enum import Enum

class AgentNames(Enum):
    TOPIC = "topic"
    NER_NAMES = "names"
    JRAGON_LIST = "jargon_list"
    NER_DECIDER = "decider_ner"
    JARGON_DEIDER = "decider_jargon"
    MOST_RELEVANT_NAMES = "most_relevant_names"
    SENTENCE_BUILDER = "sentence_builder"
    FIX_TRANSCRIPT_BY_LLM = "fix_transcript_by_llm"