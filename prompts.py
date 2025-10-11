import textwrap
import re
from typing import Tuple, Dict, List
from pydantic import BaseModel, conint, conlist


DIRECT_SUMMARY_PROMPT = textwrap.dedent("""\
    You are an expert dialogue analyst.
    
    GOAL  
    Produce a coherent and highly readable summary of the *Prior Conversation* that is useful to readers of the *Current Conversation*.
    
    CONSTRAINTS  
    • Length ~ 250 words (about two paragraphs, not more than 280 word and not less than 220 words).  
    • Plain prose; no bullet points.  
    • Use only information that appears in the Prior Conversation (no hallucination).  
    • Return **exactly** the JSON object shown in OUTPUT FORMAT.  
    • Do not show your intermediate reasoning.
    • The output summary should always start with "The prior conversation...".
    
    ---- Topic ----
    {topic}

    ---- Prior Conversation ----
    {prior_dialogue}
    
    ---- Current Conversation ----
    {current_dialogue}

    OUTPUT FORMAT  
    {{"summary":"<your two-paragraph summary here>"}}
        
    Please provide your summary!
    """)

RECURSIVE_SUMMARY_PROMPT = textwrap.dedent("""\
    You are an expert dialogue analyst.

    GOAL  
    Produce a coherent and highly readable summary of the conversation **so far** by incorporating the *Prior Summary* with the *Current Conversation*
    
    CONSTRAINTS  
    • Length ~ 250 words (about two paragraphs, not more than 280 word and not less than 220 words).  
    • Plain prose; no bullet points.  
    • Use only information that appears in the Prior Summary and the Current Conversation (no hallucination).  
    • Return **exactly** the JSON object shown in OUTPUT FORMAT.  
    • Do not show your intermediate reasoning.
    • The output summary should always start with "The prior conversation...".
    
    INPUT  

    ---- Topic ----
    {topic}
    
    ---- Prior Summary ----
    {prior_summary}
    
    ---- Current Conversation ----
    {current_dialogue}

    OUTPUT FORMAT  
    {{"summary":"<your two-paragraph summary here>"}}
    
    Please provide your summary!
    """)

MEMORY_SUMMARY_PROMPT = textwrap.dedent("""\
    You are an expert dialogue analyst.

    GOAL  
    Produce a coherent and highly readable summary of the *Prior Conversation* that is useful to readers of the *Current Conversation* based on the provided retrieved conversation memory.

    CONSTRAINTS  
    • Length ~ 250 words (about two paragraphs, not more than 270 word and not less than 230 words).  
    • Plain prose; no bullet points.  
    • Use only information that appears in the Prior Conversation (no hallucination).  
    • Return **exactly** the JSON object shown in OUTPUT FORMAT.
    • Be specific about number, entities names and events.
    • Do not show your intermediate reasoning.
    • The output summary should always start with "The prior conversation...".

    ---- Topic ----
    {topic}

    ---- Retrieved Conversation Memories ----
    {formatted_memories}

    ---- Current Conversation ----
    {current_dialogue}

    OUTPUT FORMAT  
    {{"summary":"<your two-paragraph summary here>"}}
        
    Please provide your summary!
    """)

ASPECT_AWARE_SUMMARY_PROMPT = textwrap.dedent("""\
    You are an expert dialogue analyst.

    GOAL  
    Produce a coherent and highly readable theme-aware summary of the *Prior Conversation* that is useful to readers of the *Current Conversation*.
    
    CONSTRAINTS  
    • Length ~ 250 words (about two paragraphs, not more than 270 word and not less than 230 words).  
    • Plain prose; no bullet points.  
    • Use only information that appears in the Prior Conversation (no hallucination).  
    • Return **exactly** the JSON object shown in OUTPUT FORMAT.
    • Be specific about number, entities names and events.
    • Do not show your intermediate reasoning.
    • The output summary should always start with "The prior conversation...".
    
    INPUT  
    ---- Topic ----
    {topic}
    
    ---- Prior Conversation ----
    {prior_dialogue}
    
    ---- Current Conversation ----
    {current_dialogue}
    
    
    INTERNAL WORKFLOW  
    1. From the Current Conversation, extract up to **five** most salient/important key themes/entities, including concept/topic(e.g., “Redundancy in the national budget”), entities("The WHO"), event(e.g. 911)).  
    2. For each theme:  
       a. Locate all utterances in the Prior Conversation that address that theme.  
       b. Draft a single paragraph micro-summary using only those utterances. Note the utterance indices you used.  
       c. *If the theme is new and no prior utterances are relevant, skip it.*  
    3. Combine the micro-summaries into one coherent overview summary (Length ~ 250 words (about two paragraphs, not more than 270 word and not less than 230 words).  
    4. Output only the JSON object below (Don't need to include the utterance indexes.).
    
    OUTPUT FORMAT  
    {{"summary":"<your two-paragraph summary here>"}}
    
    Please provide your summary!
    """)

# -----------------------------
# Dimension definition snippets
# -----------------------------
DIM_DEFS: Dict[str, str] = {
    "informativeness": textwrap.dedent("""\
        • **Conversational Information Gain (Informativeness):** Measures how much a response advances the group’s shared understanding or helps resolve the topic, considering both their prior knowledge and the preceding dialogue.
            * Core Question: Given what the group already knows, does this response help them understand the topic more deeply, or common sense, or move the discussion forward?
            * Focus: A forward-looking evaluation of the utterance's impact on the group's progress.
            * 1 – No gain: Obstructs, is irrelevant, or restates established content with no semantic change.
            * 2 – Minimal gain: Adds a small clarification, a minor supporting detail, or a slight nuance; noticeable but not meaningfully expanding.
            * 3 – Incremental: Builds meaningfully on the discussion by adding new evidence, reasoning, or examples that go well beyond what was previously established.
            * 4 – Insightful: Substantially deepens understanding by reframing or introduce new ideas under the topic; shifts the conversation in a new valuable way.
    """),
    "novelty": textwrap.dedent("""\
        • **Novelty:** Assesses whether the information is new compared to the Prior Knowledge and preceding dialogue.
            * Core Question: Has this specific fact, idea, or perspective already appeared in the Prior Knowledge section or been stated earlier in the conversation?
            * Focus: A backward-looking check against the established context. Novelty is about newness only — it does not consider relevance, importance, or usefulness.
            * 1 – Not at all novel: Repeats or paraphrases something already in the dialogue or Prior Knowledge, or states obvious/common-sense facts.
            * 2 – Minimally novel: Adds a minor or predictable detail to an idea already in the shared baseline.
            * 3 – Moderately novel: Contributes new evidence, a concrete example, or a supporting detail that expands or strengthens an existing idea.
            * 4 – Highly novel: Introduces a new framework, principle, topic or line of reasoning that reframes the discussion or opens an entirely new direction.
    """),
    "relevance": textwrap.dedent("""\
        • **Relevance:** Measures how substantively an utterance's content relates to the core topic or goal. Contributions that are purely procedural or meta-discourse are rated as not relevant.
            * Core Question: Is this utterance on-topic and contributing substantively to the conversation’s main goal?
            * 1 – Not at all Relevant: Completely off-topic; no clear substantive connection to the discussion’s goal.
            * 2 – Minimally Relevant: Loosely related; the connection is indirect, hypothetical, or requires inference.
            * 3 – Moderately relevant: Substantially related but not central — e.g., addresses a side issue, counterpoint, or secondary aspect of the discussion.
            * 4 – Highly relevant: Directly and explicitly addresses the core topic or goal of the conversation.
    """),
    "implication_scope": textwrap.dedent("""\
        • **Implication Scope (Audience-Centered):** Measures the intended reach and generalisation of the implication—who it is meant to matter to and whether it generalises to broader populations.
            * Core Question: For whom is this statement intended to be significant? How does the implication generalise to broader populations?
            * 1 – Local: Manages the immediate conversation; procedural or significant only to the participants involved (e.g., "I can't hear you," "What was that?," "I agree.").
            * 2 – Bounded/Specific: Presents a self-contained fact, feeling, or stance without generalising. Significance is confined to the speaker or a specific case.
            * 3 – Generalising (Inductive): Generalises a specific case or evidence to a broader conclusion or public domain.
            * 4 – Universal (Deductive): States a broad, abstract principle, value, or norm that applies widely and frames the discussion in top-down terms.
    """),
}

# Optional heuristics block you can inject when both novelty & informativeness are present
HEURISTICS = textwrap.dedent("""\
---
### **Analyst Heuristics & Key Distinctions**
* **Novelty vs. Informativeness — the "What vs. So What" test.**
  - **Novelty** identifies **what** new information is presented.
  - **Informativeness** evaluates **so what** impact that new information has on the group’s progress.
""")

def _dimension_section(dimensions: List[str], include_heuristics: bool) -> str:
    parts = ["---", "### **Dimension Definitions**", "---", ""]
    for d in dimensions:
        if d not in DIM_DEFS:
            raise ValueError(f"Unknown dimension: {d}")
        parts.append(DIM_DEFS[d].rstrip())
        parts.append("")  # blank line between blocks
    if include_heuristics and {"novelty", "informativeness"}.issubset(set(dimensions)):
        parts.append(HEURISTICS.rstrip())
    return "\n".join(parts).rstrip()

def _json_schema_snippet(dimensions: List[str], context_type: str) -> str:
    """
    Builds the JSON schema line used in the output instruction.
    Example: [{"utterance_index": int, "novelty": int, "relevance": int, "context_type": FULL}, …]
    Note: context_type_literal should be something like FULL, MIX, INFO, or a number/string literal.
    """
    keys = ['"utterance_index": int'] + [f'"{d}": int' for d in dimensions] + [f'"context_type": {context_type} (hard coded for identification purpose)']
    return "[{{{}}}, …]".format(", ".join(keys)).replace("{", "{{").replace("}", "}}")

def build_eval_prompt(
    *,
    dimensions: List[str],
    context_type: str,  # used as a literal in the schema (no quotes)
    title: str = "You are an expert dialogue analyst acting from the perspective of a community audience, evaluating the quality of a public discussion on a specific topic.",
    include_heuristics: bool = True,
    add_preamble_importance: bool = True,
    include_example_block: bool = False,
    example_block: str = "",  # optional custom example text block (already formatted)
) -> str:
    """
    Returns a parametrised prompt that rates *only* the specified dimensions.
    Keeps your placeholders {topic}, {context}, {target}, {start}, {end}, {total}.
    """
    if not dimensions:
        raise ValueError("At least one dimension must be specified.")
    # Normalise casing: accept synonyms
    alias = {"info": "informativeness", "information_gain": "informativeness", "implication": "implication_scope"}
    dims = [alias.get(d.lower(), d.lower()) for d in dimensions]

    dim_section = _dimension_section(dims, include_heuristics=include_heuristics)
    schema_line = _json_schema_snippet(dims, context_type)

    importance_line = (
        "**IMPORTANT:** Your evaluation must treat the `Prior Knowledge` section and the preceding dialogue as the **Shared Knowledge Baseline**. "
        "An utterance that simply repeats or paraphrases information from this baseline should be rated low on Novelty and Informativeness (if present)."
        if add_preamble_importance
        else ""
    )

    # Example
    if include_example_block:
        if example_block and example_block.strip():
            example_part = "\n---\n### Example\n\n" + example_block.strip() + "\n"
        else:
            example_part = "\n---\n### Example\n\n" + _example_block(dims)
    else:
        example_part = ""

    prompt = textwrap.dedent(f"""\
    {title}
    After considering the **Prior Knowledge** (if any) and **Dialogue context** (if any), please rate each of the **TARGET Utterances** for the following dimension(s), each with a scale of 1–4 (4=highest): {", ".join(dims)}.

    {importance_line}

    {dim_section}

    {example_part}

    ##Task##:

    ---- Topic/Goal ----
    {{topic}}

    ---- Prior Dialogue / Knowledge ----

    {{context}}

    ---- Prior Dialogue / Knowledge End ----

    REMINDER: Evaluate all utterances from {{start}} to {{end}} inclusive within the following block.
    ---- Target Utterances----

    {{target}}

    ---- Target Utterances End----
    REMINDER: Evaluate all utterances from {{start}} to {{end}} inclusive within the preceding block.

    - Please rate every single utterance from the **Target Utterances** section (from index {{start}} to {{end}}, total of {{total}} utterances).
    - Before generating your response, double-check that you have evaluated every utterance in the specified range.
    - Your rating should reflect the salient message or most impressive theme of the utterance, not just one part in isolation.
    - Your final output must be a valid JSON array whose elements follow this schema:

    {schema_line}

    ---- Evaluation----
    """).strip("\n")
    return prompt

def build_eval_prompt_using_external_ratings(
    *,
    dimensions: List[str],
    context_type: str,  # used as a literal in the schema (no quotes)
    title: str = "You are an expert dialogue analyst acting from the perspective of a community audience, evaluating the quality of a public discussion on a specific topic.",
    include_heuristics: bool = True,
    add_preamble_importance: bool = True,
    include_example_block: bool = True,
    example_block: str = "",  # optional custom example text block (already formatted)
) -> str:
    """
    Returns a parametrised prompt that rates *only* the specified dimensions.
    Keeps your placeholders {topic}, {context}, {target}, {start}, {end}, {total}.
    """
    if not dimensions:
        raise ValueError("At least one dimension must be specified.")
    # Normalise casing: accept synonyms
    alias = {"info": "informativeness", "information_gain": "informativeness", "implication": "implication_scope"}
    dims = [alias.get(d.lower(), d.lower()) for d in dimensions]

    dim_section = _dimension_section(dims, include_heuristics=include_heuristics)
    schema_line = _json_schema_snippet(["informativeness"], context_type)

    importance_line = (
        "**Instruction:**"
        "- Your evaluation must treat the `Prior Knowledge` section and the preceding dialogue as the **Shared Knowledge Baseline**. "
        "- An utterance that simply repeats or paraphrases information from this baseline should be rated low on Novelty and Informativeness (if present)."
        "- Some utterance have been rated by external resource (e.g. human annotator), please utilize the annotation label for the prediction of CIG/informativeness."
        "- In your final result please only output rating for CIG/informativeness."
        if add_preamble_importance
        else ""
    )

    # Example
    if include_example_block:
        if example_block and example_block.strip():
            example_part = "\n---\n### Example\n\n" + example_block.strip() + "\n"
        else:
            example_part = "\n---\n### Example\n\n" + _example_block(["informativeness"])
    else:
        example_part = ""

    prompt = textwrap.dedent(f"""\
    {title}
    After considering the **Prior Knowledge** (if any) and **Dialogue context** (if any), please rate each of the **TARGET Utterances** for their Conversational Information Gain (CIG)/informativeness  with a scale of 1–4 (4=highest) and under the aid of the pre-annotated rating of Novelty, Relevance, and Implication Scope. Below is the definitions of the Conversational Information Gain and the underlying three aspects- Novelty, Relevance, and Implication Scope.

    {dim_section}
    
    {importance_line}

    {example_part}

    ##Task##:

    ---- Topic/Goal ----
    {{topic}}

    ---- Prior Dialogue / Knowledge ----

    {{context}}

    ---- Prior Dialogue / Knowledge End ----

    REMINDER: Evaluate all utterances from {{start}} to {{end}} inclusive within the following block.
    ---- Target Utterances----

    {{target}}

    ---- Target Utterances End----
    REMINDER: Evaluate all utterances from {{start}} to {{end}} inclusive within the preceding block.

    - Please rate every single utterance from the **Target Utterances** section (from index {{start}} to {{end}}, total of {{total}} utterances).
    - Before generating your response, double-check that you have evaluated every utterance in the specified range.
    - Your rating should reflect the salient message or most impressive theme of the utterance, not just one part in isolation.
    - Please only return rating for informativeness.
    - Your final output must be a valid JSON array whose elements follow this schema:

    {schema_line}

    ---- Evaluation----
    """).strip("\n")
    return prompt


# -----------------------------
# Convenience wrappers matching your three presets
# -----------------------------
def build_info_only_prompt(context_type: str) -> str:
    return build_eval_prompt(
        dimensions=["informativeness"],
        context_type=context_type,
        include_heuristics=False,
        add_preamble_importance=True,
        include_example_block=False,
    )

def build_mix_prompt(context_type: str) -> str:
    return build_eval_prompt(
        dimensions=["novelty", "relevance", "implication_scope"],
        context_type=context_type,
        include_heuristics=False,
        add_preamble_importance=True,
        include_example_block=False,
    )

def build_full_prompt(context_type: str) -> str:
    return build_eval_prompt(
        dimensions=["informativeness", "novelty", "relevance", "implication_scope"],
        context_type=context_type,
        include_heuristics=True,
        add_preamble_importance=True,
        include_example_block=False,
    )



EVAL_CLAIMS_WITH_MEMORY_PROMPT_FULL = textwrap.dedent("""\
You are an expert dialogue analyst acting from the perspective of a community audience, evaluating the quality of a public discussion on a specific topic.

You will receive:
1) **Conversation topic/goal: A string describing the conversation topic/goal.
2) **Existing memories**: a JSON array of previously stored claims (the shared knowledge baseline).
3) **TARGET Claims**: a JSON array of newly-extracted claims to score.

Your job is to rate each **TARGET claim** on a scale of four levels (1–4; 4 = highest):
- **Conversational Information Gain (Informativeness)**
- **Novelty**
- **Relevance**
- **Implication Scope (Audience-Centered)**

**BASELINE RULE**
Treat **Existing memories** as the **Shared Knowledge Baseline**.

---
### Score the four dimensions (claim-level) for each claim

• Conversational Information Gain (Informativeness): Measures how much a response advances the group’s shared understanding or helps resolve the topic, considering both their prior knowledge and the preceding dialogue.**.
    * Core Question: Given what the group already knows, does this response help them understand the topic more deeply or move the discussion forward?
    * Focus: A forward-looking evaluation of the utterance's impact on the group's progress.
    * 1 – No gain: Obstructs, is completely irrelevant, or repeats established information without adding meaning. Includes verbatim repetition or paraphrasing with no semantic change.
    * 2 – Minimal gain: Adds a small clarification, a minor supporting detail, or a slight nuance to an existing idea. The addition is noticeable but does not significantly expand understanding.
    * 3 – Incremental: Builds meaningfully on the discussion by adding new evidence, reasoning, or examples that go well beyond what was previously established.
    * 4 – Highly insightful:** Substantially deepens understanding by introducing a novel perspective, synthesizing prior points, or offering a reframing/solution that shifts the conversation in a valuable way.

• Novelty: Assesses whether the information is new compared to the Prior Knowledge and preceding dialogue.
    * Core Question: Has this specific fact, idea, or perspective already appeared in the Prior Knowledge section or been stated earlier in the conversation?
    * Focus: A backward-looking check against the established context. Novelty is about newness only — it does not consider relevance, importance, or usefulness.
    * 1 – Not at all novel: Repeats or paraphrases something already in the existing Knowledge, or states obvious/common-sense facts.
    * 2 – Minimally novel: Adds a minor or predictable detail to an idea already in the shared baseline.
    * 3 – Moderately novel: Contributes new evidence, a concrete example, or a supporting detail that expands or strengthens an existing idea.
    * 4 – Highly novel: Introduces a new framework, principle, topic or line of reasoning that reframes the discussion or opens an entirely new direction.

• Relevance: Measures how substantively an utterance's content relates to the core topic or goal. Contributions that are purely procedural or meta-discourse are rated as not relevant.
    * Core Question: Is this utterance on-topic and contributing substantively to the conversation’s main goal?
    * 1 – Not at all Relevant: Completely off-topic; no clear substantive connection to the discussion’s goal.
    * 2 – Minimally Relevant: Loosely related; the connection is indirect, hypothetical, or requires inference.
    * 3 – Moderately relevant: Substantially related but not central — e.g., addresses a side issue, counterpoint, or secondary aspect of the discussion.
    * 4 – Highly relevant: Directly and explicitly addresses the core topic or goal of the conversation.

• Implication Scope (Audience-Centered): Measures the intended reach and generlisation of the implication—who it is meant to matter to and does the implication generalise to broader population.

    * Core Question: For whom this statement is intended to be significant? How does the implication generalise to broader population? (e.g., presenting a fact, making a generalization)?
    * 1 – Local: Manages the immediate conversation; procedural or significant only to the participants involved (e.g., "I can't hear you," "What was that?," "I agree.").
    * 2 – Bounded/Specific: Presents a self-contained fact, feeling, or stance without performing an act of generalization. The significance is confined to the speaker or the specific case (e.g., "Our city council failed to pass the new zoning law.", "I support more funding for our school system.").
    * 3 – Generalizing (Inductive): Generalise a specific case (an experience, evidence, or observation) to a broader conclusion, issue, or public domain (e.g., "This local factory closure is a sign of a nationwide problem.", "My mother's career struggles show why school funding is so important.").
    * 4 – Universal (Deductive): States a broad, abstract principle, value, or norm that applies widely, framing the discussion in 'top-down' terms (e.g., “All people deserve dignity.”, “This policy is wrong because all people deserve dignity.”).


---
### Output format (STRICT)
Return a **valid JSON array**. Each element corresponds to one TARGET claim index and includes only these keys and integer values:

[{{"id": int,
   "informativeness": int,
   "novelty": int,
   "relevance": int,
   "implication_scope": int}}, …]

Do not include explanations or extra keys in the final output.

---

### Example

---- Topic/Goal ----
Should our state retain the death penalty?

---- Dialogue context (optional) ----
(omitted)
---- Dialogue context End ----

---- Existing memories (Shared Knowledge Baseline) ----

- (Turn #5) Maria: "The death penalty provides justice for victims' families."
- (Turn #8) Robert Rosenkranz: "The financial cost of death penalty appeals is very high."\n

---- Existing memories End ----

REMINDER: Evaluate all TARGET claims from 1 to 3 inclusive within the following block.

---- TARGET Claims ----

1. "id": "mem_013", "speaker": "Sam", "claim": "There is an irreversible risk of executing an innocent person.", "turn_id": "14"
2. "id": "mem_014", "speaker": "Sam", "claim": "The financial cost of death penalty appeals exceeds the cost of life imprisonment.", "turn_id": "14"
3. "id": "mem_015", "speaker": "Sam", "claim": "The death penalty gives justice and peace for victims' families.", "turn_id": "14"
4. "id": "mem_016", "speaker": "Sam", "claim": "California has stopped death penalty for a long time, and so do most developed countries", "turn_id": "14"
5. "id": "mem_017", "speaker": "Sam", "claim": "In and Out burger is from LA.", "turn_id": "14"
6. "id": "mem_017", "speaker": "Sam", "claim": "It is almost dinner time", "turn_id": "14"

---- TARGET Claims End ----


---- Output ----
[
  {{"id": mem_013, "informativeness": 4, "novelty": 4, "relevance": 4, "implication_scope": 4}},
  {{"id": mem_014, "informativeness": 3, "novelty": 3, "relevance": 4, "implication_scope": 4}},
  {{"id": mem_015, "informativeness": 1, "novelty": 1, "relevance": 4, "implication_scope": 4}},
  {{"id": mem_016, "informativeness": 3, "novelty": 3, "relevance": 4, "implication_scope": 3}},
  {{"id": mem_017, "informativeness": 1, "novelty": 3, "relevance": 1, "implication_scope": 2}},
  {{"id": mem_017, "informativeness": 1, "novelty": 4, "relevance": 1, "implication_scope": 1}}
]


### Task
---- Topic/Goal ----
{topic}

---- Dialogue context (optional) ----
{dialogue_context}
---- Dialogue context End ----

---- Existing memories (Shared Knowledge Baseline) ----
{existing_memories}
---- Existing memories End ----

---- TARGET Claims ----
{claims}
---- TARGET Claims End ----

- Use only integers 1–4 for each dimension.
- Final output = the strict JSON array described above (no prose).

---- Output ----
""")

# -----------------------------
# Example block generator
# -----------------------------
_BASE_EXAMPLE_HEADER = textwrap.dedent("""\
    ---- Topic ----
    Gun Reduces Crime

    ---- Prior Dialogue / Knowledge End ----

    Prior Summary: Gary Kleck and Gayle Smith argue in favor of the motion, asserting that responsible and trained firearm ownership empowers citizens, deters criminals, and enhances community safety. Kleck, drawing from extensive criminological research, highlights that defensive gun use significantly outweighs the risks, advocating for policy respecting gun ownership. Smith reinforces this perspective, emphasizing the role of lawful firearms, complemented by targeted social investments, in protecting vulnerable populations. Conversely, R. Gil Kerlikowske opposes the motion, stressing that widespread firearm availability endangers both civilians and law enforcement, advocating instead for tighter gun controls and targeted prevention strategies to effectively improve public safety.

    0. John Donvan(mod): Okay, I—I think we have impasse on this… And I— I wanna go to Chief
    Kerlikowske, Seattle police chief, because you were talking about British police being unarmed,
    and preferring that in many situations because they felt being armed would invite assault, as
    there were some reported case of causality with stolen police guns, and yet Gary Kleck arguing
    for the motion said why are police armed in the first place unless it is to deter assault, do
    you—can you take that on?

    ---- Prior Dialogue / Knowledge End ----

    ---- Target Utterances----
    1. R. Gil Kerlikowske (against): Yes I can, and, remember what Truman said, if I could line up all the economists end to end,
    wouldn’t that be a beautiful sight, so that, I just— And one, and one is on my panel. So…getting
    away from the statistics and going to that police officers are highly trained with firearms. They
    practice and qualify, they can’t graduate from the academy with qualifying. Without question, they
    can use the gun in a defensive mode. But when you look at the numbers of police officers in the
    United States and I remember—I can picture every moment to this day, of the first time I had an
    officer killed in the line of duty. Young, bright, talented, incredible shape, shot 13 times with his
    own handgun. Wasn’t a question of training, wasn’t a question of anything else. Officers are routinely
    wounded and assaulted with their own guns. That’s why—

    2. Gary Kleck (for): That’s not true—absolutely, unequivocally not true. I’ve said it before and I’ll say it again: as a criminologist with more than four decades—yes, forty solid years—spent poring over every credible dataset on guns and crime, I can assure you the facts speak for themselves. An armed citizenry deters violent offenders, saves innocent lives, and does so far more often than it ever puts anyone in danger.

    3. R. Gil Kerlikowske (against): By the way, could someone please adjust the air conditioning? It’s absolutely sweltering in here—easily thirty-two degrees at least, and I’m starting to feel dizzy from the heat. My shirt’s stuck to me, the lights are glaring, and I can’t help but wonder if anyone else is wilting as badly as I am. Honestly, it feels like we’re in a greenhouse rather than a debate hall. Perhaps we should discuss climate change instead of gun ownership, as it might actually kill more people. So…um, what exactly is not true?

    4. Gary Kleck (for): I mean, sure, there have been cases where a criminal managed to steal a gun from a trained officer. But if someone is bold enough to disarm law enforcement, what more dangerous things might they do to an unarmed citizen? I was talking to some unbelievable folks—real patriots, salt of the earth—and they’ve got some very interesting ways of keeping things safe, let me tell you. Why should we abandon firearms altogether, instead of exploring smarter solutions—like securing guns with personal biometric locks? There is a startup in Detroit that has developed a fingerprint-based AI gun-lock. They have trialed with the local police departments, and have seen a significant decline of stolen guns.

    ---- Target Utterances End----
""")

# Ratings to reuse across variants
_EXAMPLE_SCORES = {
    "info":  [("1", {"informativeness": 3}),
              ("2", {"informativeness": 1}),
              ("3", {"informativeness": 1}),
              ("4", {"informativeness": 4})],
    "mix":   [("1", {"novelty": 3, "relevance": 3, "implication_scope": 3}),
              ("2", {"novelty": 1, "relevance": 4, "implication_scope": 4}),
              ("3", {"novelty": 3, "relevance": 1, "implication_scope": 1}),
              ("4", {"novelty": 4, "relevance": 3, "implication_scope": 3})],
    "full":  [("1", {"informativeness": 3, "novelty": 3, "relevance": 3, "implication_scope": 3}),
              ("2", {"informativeness": 1, "novelty": 1, "relevance": 4, "implication_scope": 4}),
              ("3", {"informativeness": 1, "novelty": 3, "relevance": 1, "implication_scope": 1}),
              ("4", {"informativeness": 4, "novelty": 4, "relevance": 3, "implication_scope": 3})],
}

def _example_block(dimensions: List[str]) -> str:
    """Return a dimension-aware example block including the Evaluation JSON."""
    dimset = set(dimensions)
    if dimset == {"informativeness"}:
        scores = _EXAMPLE_SCORES["info"]
    elif dimset == {"novelty", "relevance", "implication_scope"}:
        scores = _EXAMPLE_SCORES["mix"]
    elif dimset == {"informativeness", "novelty", "relevance", "implication_scope"}:
        scores = _EXAMPLE_SCORES["full"]
    else:
        # Build a generic mapping: pick from 'full' and drop unused keys
        full_scores = _EXAMPLE_SCORES["full"]
        scores = []
        for idx, sdict in full_scores:
            pruned = {k: v for k, v in sdict.items() if k in dimset}
            if not pruned:
                # If somehow empty, just include a neutral dimension present
                first_dim = sorted(dimset)[0]
                pruned = {first_dim: 3}
            scores.append((idx, pruned))

    eval_items = []
    for idx, metrics in scores:
        item = {"utterance_index": int(idx), **metrics}
        eval_items.append(item)

    # Pretty JSON without importing json (keeps it pure text & deterministic)
    # Note: This is illustrative JSON the model will see as part of the prompt.
    def _dict_to_json(d: Dict[str, int]) -> str:
        pairs = []
        for k, v in d.items():
            if isinstance(v, str):
                pairs.append(f'"{k}": "{v}"')
            else:
                pairs.append(f'"{k}": {v}')
        return "{%s}" % (", ".join(pairs))

    eval_str = ("[\n    " + ",\n    ".join(
        _dict_to_json(x) for x in eval_items
    ) + "\n]")
    eval_str = eval_str.replace("{", "{{").replace("}", "}}")

    return (
        _BASE_EXAMPLE_HEADER
        + "\n    ---- Evaluation----\n    "
        + eval_str
        + "\n"
    )


class FullRatingClass(BaseModel):
    utterance_index: int
    informativeness: int
    novelty: int
    relevance: int
    implication_scope: int

class InfomrativeRatingClass(BaseModel):
    utterance_index: int
    informativeness: int

class MixedRatingClass(BaseModel):
    utterance_index: int
    novelty: int
    relevance: int
    implication_scope: int


# ── 1.  Common regex & system message ───────────────────────────────────
BOUNDARY_JSON_RE = re.compile(
    r"\[\s*\[\s*\d+\s*,\s*\d+\s*](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*])*\s*]"
)

SEG_SYSTEM_MSG = """
You are an expert dialogue analyst. Your task is to segment an interaction phase into coherent subtopic segments of around 250-400 words or no loner that 15 speaker turns each.

Context:
- Topic/Goal: {topic}
- Prior dialogue: {prior_summary}

Please identify the subtopics and boundaries in the following dialogue. You should segment all the following dialogue.
Return ONLY valid JSON according to the registered schema (no extra text). segment_index starts from 0.
"""

class SegmentClass(BaseModel):
    segment_index: int
    utterances_interval: list[int]
    segment_subtopic: str

class SummaryClass(BaseModel):
    summary: str

SEG_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "segment_index": {
                    "type": "integer",
                    "minimum": 0
                },
                "utterances_interval": {
                    "type": "array",
                    "items": [
                        {"type": "integer", "minimum": 1},
                        {"type": "integer", "minimum": 1}
                    ],
                    "minItems": 2,
                    "maxItems": 2
                },
                "segment_subtopic": {
                    "type": "string"
                }
            },
            "required": [
                "segment_index",
                "utterances_interval",
                "segment_subtopic"
            ]
        }
    }



