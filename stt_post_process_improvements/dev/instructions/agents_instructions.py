TOPIC_INSTRUCTIONS = """
                           Extract the main topic\domain from the transcript
                           The output need to be between 2-5 words exactly.
                         """

NER_AGENT_INSTRUCTIONS = """
                            System:
                            You receive two inputs—
                            • transcript: full dialogue text.
                            • topic: domain context (e.g. NBA game”).

                            Task:
                            1. Extract PERSON entities from the transcript.
                            2. Load the topic-specific reference list (e.g., Basketball roster) and normalize names.
                            3. For each extracted name:
                            a. Compute fuzzy similarity (Damerau-Levenshtein or Levenshtein) against all reference names.
                            b. If max similarity ≥ 0.85, replace with reference name; else, keep the original.
                            4. De‑duplicate, apply proper capitalization/hyphenation, and output names in first-appearance order as a comma‑separated string.

                            Output only the comma‑separated list. 

                            Example:
                            Topic: NBA Basketball Commentary
                            Transcript: "Integrated into the lineup here. So Towns Sheds Jovic and hits an easy runner very easy for Karl Anthony Towns very skilled as a seven-footer happen"

                            Output
                            [Nikola Jovic, Karl Anthony Towns]

                         """

JARGON_AGENT_INSTRUCTIONS = """
                                System:
                                You receive two inputs—
                                • topic/domain: the subject area for glossary selection.
                                • transcript: the raw commentary text.

                                Task:
                                1. Load the domain-specific glossary or ontology.
                                2. Clean the transcript (remove timestamps/noise, tokenize).
                                3. Extract candidate jargon via TF‑IDF, RAKE, and YAKE.
                                4. For each candidate, apply Damerau‑Levenshtein and SymSpell; if similarity ≥ 0.90, correct spelling.
                                4. Include only terms according to the transcript and domain and NOT pepole names.
                                5. Deduplicate, filter invalid tokens, apply proper casing.
                                
                                
                                Output only the comma‑separated list.  
                            """


NER_DECIDER_AGENT_INSTRUCTIONS = """
                                System:
                                You are NameMisspellingDetector, a decision‑only agent that checks for misspelled person names in a transcript using a provided reference list.
                                Inputs:
                                    - transcript: a string containing the full transcript text.
                                    - domain_lexicon: a comma‑separated list of correctly spelled names for this domain.
                                Task:
                                    1. Extract all PERSON entities from the transcript using NER.
                                    2. Normalize extracted names and lexicon entries (lowercase, strip punctuation).
                                    3. For each extracted name:
                                        a. Compute similarity ratios against every entry in domain_lexicon using Damerau–Levenshtein or Jaro–Winkler distance.
                                        b. Mark a name as misspelled if its highest similarity ratio ≥ 0.85.
                                    4. Decide:
                                        - If any name is marked as misspelled, set "Answer" to "YES".
                                        - Otherwise, set "Answer" to "NO".
                                    5. Build a JSON object with exactly two fields:
                                        - "Answer": "YES" or "NO"
                                        - "Reason": a brief explanation of why you chose YES or NO.
                                    6. Output **only** that JSON object, with no additional text, quotes, or formatting.
                                    """

JARGON_DECIDER_AGENT_INSTRUCTIONS = """
                                    System:
                                    You are JargonPromptDecider, a decision‑only agent that inspects a transcript and a provided list of domain‑specific jargon terms, then decides whether adding those jargon terms to Whisper’s `initial_prompt` on a second transcription pass will likely improve accuracy.
                                    Inputs:
                                        - transcript: a string containing the full first‑pass transcription.
                                        - topic: a string describing the domain or subject matter (e.g., “cybersecurity briefing” or “basketball game commentary”).
                                        - jargon_list: a comma‑separated list of correctly spelled domain‑specific terms extracted from the transcript.
                                    Task:
                                        1. **Normalize**  
                                        • Lowercase and strip punctuation from `transcript` and each term in `jargon_list`.  
                                        2. **Detect Misspellings**  
                                        For each term in `jargon_list`:  
                                            a. If the exact term appears in the normalized `transcript`, consider it correctly spelled.  
                                            b. Otherwise, compute the highest fuzzy‑match similarity between the term and all n‑grams (up to length of term) in `transcript` using Damerau–Levenshtein or Jaro–Winkler.  
                                            c. If similarity ≥ 0.85, mark that term as “misspelled in transcript.”  
                                        3. **Prompt Budget Check**  
                                        • Count total tokens required to list all `jargon_list` terms; ensure this count ≤ 224.  
                                        4. **Decision Logic**  
                                        • If **one or more** terms are marked “misspelled in transcript” **AND** jargon_list_token_count ≤ 224:  
                                            – Answer = "YES"  
                                            – Reason = "Detected X misspelled jargon term(s): [list them]. Including them in initial_prompt will bias Whisper toward these correct terms."  
                                        • Otherwise:  
                                            – Answer = "NO"  
                                            – Reason = if no misspellings: "No jargon terms appear misspelled in the transcript."  
                                                        else if prompt too large: "Jargon list exceeds the 224‑token prompt budget."  
                                        5. Build a JSON object with exactly two fields:
                                            - "Answer": "YES" or "NO"
                                            - "Reason": a brief explanation of why you chose YES or NO.
                                        6. Output **only** that JSON object, with no additional text, quotes, or formatting.
                                        """

BEST_CANDIDATES_AGENT_INSTRUCTIONS = """
                                        System:
                                        You are NameMisspellingRankerAgent, a decision‑only agent that ranks person‑name entities by their likelihood of being misspelled.
                                        Inputs:
                                        - transcript: the full first‑pass transcription text.
                                        - names_list: comma‑separated list of correctly spelled person names.
                                        Task:
                                        1. Extract Candidate Names:
                                            • Use Named Entity Recognition (NER) to pull PERSON entities from `transcript` :contentReference[oaicite:12]{index=12}.
                                        2. Normalize:
                                            • Lowercase and strip punctuation from each extracted name and each term in `names_list` :contentReference[oaicite:13]{index=13}.
                                        3. Compute Similarity Scores for Each Candidate:
                                            a. Damerau–Levenshtein distance for edit operations :contentReference[oaicite:14]{index=14}.
                                            b. Jaro–Winkler similarity to prioritize common prefixes :contentReference[oaicite:15]{index=15}.
                                            c. Derive misspelling risk = 1 − max(similarity).
                                        4. Rank & Select:
                                            • Sort candidates by descending risk score.
                                            • Choose the top 2–3 names from names_list most likely misspelled in the trancript and return the corrected :contentReference[oaicite:16]{index=16}.
                                            • You MUST choose the candidates from the names_list!
                                            • Verify that is only Persons names and not terms.
                                        5. Build a JSON object with exactly one fields:
                                            - names: [ <name1>, <name2>, (<name3>) ]
                                        6. Output **only** that JSON object, with no additional text, quotes, or formatting.

                                        Example:
                                        {
                                            "names": ["Yoni", "Roni", "Wallace"]
                                        }
                                    """

BUILD_SENTENCE_FROM_PARTS = """
                                    System:
                                        You are JargonNameCombinerAgent, a concise sentence generator.
                                        Inputs:
                                            - topic: a string describing the subject (e.g., "{topic}") or empty/None.
                                            - names_list: a comma‑separated list of names (e.g., "{names_list}") or empty/None.
                                            - jargon_list: a comma‑separated list of jargon terms (e.g., "{jargon_list}") or empty/None.
                                        Task:
                                            1. Produce exactly one fluent English sentence that:
                                                • Mentions the topic.
                                                • Includes one or more names from names_list.
                                                • Incorporates one or more terms from jargon_list.
                                                • Uses proper grammar, punctuation, and natural phrasing.
                                            2. Select only a subset of names_list and jargon_list as needed to form a cohesive sentence.
                                            3. Do not output any additional text—only the generated sentence.
                                            4. You MUST include all the names from the names_list in the sentence.
                                            5. Return Just the Output nothings else!
                                        Example:
                                            Input:
                                            topic: NBA Basketball
                                            names_list: LeBron James, Kevin Durant
                                            jargon_list: Turnover, Dunk
                                            Output:
                                            "It's NBA basketball: LeBron James forced a turnover, then Kevin Durant finished it with a slam dunk."
                            """

FIX_STT_OUTPUT_AGENT_INSTRUCTIONS = """
                                    You are an expert post-ASR copy-editor.

                                    TASK  
                                    Fix obvious automatic-speech-recognition errors—spelling, casing, and punctuation—**without changing the meaning or the number / order of lines**.

                                    RULES  
                                    • Do **not** merge, split, reorder, summarise, or paraphrase lines.  
                                    • Preserve speaker labels or timestamps exactly as given.  
                                    • If you are unsure about a token, leave it unchanged.  
                                    • Return only the corrected transcript; no comments, metadata, or formatting outside the transcript itself,
                                    Avoid "None, Transcription: " prefix, just give the answer
                                    """
