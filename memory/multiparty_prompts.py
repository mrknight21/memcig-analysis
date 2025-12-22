from datetime import datetime
import textwrap

# In multiparty_prompts.py

CONTEXTUAL_BRIEFING_PROMPT = textwrap.dedent("""\
    You are an expert dialogue analyst.

    GOAL  
    Produce a summary of the *Prior Conversation* that is useful to readers of the *Current Conversation* based on the provided retrieved conversation memory.

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
    {formatted_retrieved_memories}

    ---- Current Conversation ----
    {current_dialogue}

    OUTPUT FORMAT  
    {{"summary":"<your two-paragraph summary here>"}}
    """)

CONTEXTUAL_BRIEFING_PROMPT_OLD = """
You are a master intelligence analyst. Your task is to write a concise, fluent "contextual briefing" paragraph based on a set of historical memory snippets from a conversation. This briefing will prepare a user to understand an upcoming conversation segment.

**Goal**
-----
Synthesize the provided `retrieved_memories` into a single, coherent narrative paragraph. The briefing should set the stage for the topics about to be discussed by summarizing what has been said about them in the past, respecting the chronological order of the conversation.

**Input Format**
--------------
You will be given a list of relevant memories retrieved from a conversation, sorted chronologically. Each includes a `turn_id` for ordering, the speaker, and the claim.

**Guidelines**
------------
*   **Use Narrative Language for Flow:** The `turn_id` indicates the chronological order of events. Use this order to build a narrative. Instead of explicitly stating "at turn #X", use natural, temporal language like "Initially...", "Following this...", "Later in the discussion...", "This was then corrected when...", or "A consensus eventually emerged that...". The final output **must not** contain explicit mentions of "turn #" or "turn ID".
*   **Synthesize, Don't Just List:** Do not simply list the facts. Weave them into a well-written paragraph.
*   **Attribute Correctly:** Mention speakers by name (e.g., "Alice pointed out...", "John's earlier claim...").
*   **Past Tense:** Write the briefing in the past tense.
*   **Objective and Concise:** Maintain a neutral, analytical tone.
*   **Be specific about number, entities names and events.
*   **Length ~ 250 words (about two paragraphs, not more than 280 word and not less than 220 words).

---
### Example

**Input Data (Retrieved Memories)**
----------------------------------
- Topic: "pros and cons of remote working"

- (Turn #7) Bob: "A 2023 Stanford study actually says remote work lowers productivity by 10 percent."
- (Turn #8) Carol: "Suggests Bob may have mixed up the numbers from the 2023 Stanford study"
- (Turn #9) Bob: "The study shows no net productivity loss, just collaboration challenges"
- (Turn #12) Bob: "we all seem to agree remote work greatly improves work-life balance."

**Output Briefing (Fluent Narrative)**
----------------------------------
The discussion around remote work productivity has evolved significantly. Initially, Bob cited a Stanford study to claim that remote work lowered productivity. However, this was immediately questioned by Carol, leading Bob to correct his initial statement shortly thereafter, clarifying that the study actually showed no net productivity loss but did highlight collaboration challenges. Later in the conversation, a point of consensus emerged when Bob noted general agreement that remote work substantially improves work-life balance.

---
### Your Task

**Input Data (Retrieved Memories)**
----------------------------------
- Topic {{topic}}

{{formatted_retrieved_memories}}

**Output Briefing (Fluent Narrative)**
----------------------------------

"""

MULTIPARTY_CLAIM_EXTRACTION_PROMPT = """
You are an Atomic-Fact Extractor.

Goal
-----
Given a conversational context and a target utterance, your task is to extract a list of atomic claims.

⚠️ Core Principles
Self-Contained: A claim must be a complete proposition that is fully understandable without the original context.

Atomic: A claim should represent the smallest piece of information that can be independently true or false.

Semantically Distinct: From a single utterance, do not extract multiple claims that are paraphrases of each other. Each claim in the output list must represent a unique piece of information.

⚠️ Extraction Rules

1. Extract Explicit Claims: These are facts literally stated in the target utterance.

2. Extract Salient Implicit Claims: These are significant meanings a listener would confidently infer (e.g., answering "No" to "Any fever?" implies "The patient does not have a fever").

3. Focus on Content, Not Action: The claim should state the content of a suggestion, offer, or question, not describe the speech act itself.

⚠️ What to Avoid

1. Do not describe the act of speaking. Avoid claims that start with verbs like "Asks...", "Introduces...", "States...", or "Suggests...".
    - Bad: Asks for a round of applause for the debaters.
    - Good: The debaters should receive a round of applause.

2. Do not include conversational filler or hedge words like "I think," "maybe," or "kind of."

    - Bad: I think owning a gun is just too much of a risk.
    - Good: Owning a gun is too much of a risk.

The only exception for using a speech verb is when reporting someone else's speech, as in "Stephen said that..."


JSON schema (List of extracted memory json objects)
-----------

{"memories":
    [
      {
        "speaker":        "<who produced the target utterance; canonical name or 'unknown'>",
        "target_speaker": "<the people the speaker is talking to— may be another person, a specific group, 'everyone', or 'unknown'>",
        "claim":          "<proposition rewritten so it stands alone; do not repeat the speaker’s name unless essential>",
        "turn_id":        "<identifier supplied by caller>"
      },
      …
    ]
}

*  If no memories extracted, please return {"memories":[]}

Guidelines
----------
*   **Self-Contained Claims**: A claim must be fully understandable without the original context. If a detail is essential for clarity (e.g., the name of a study, a specific date, the subject of a pronoun), it MUST be included in the rewritten claim.
*   If the sentence contains multiple propositions, output one JSON object per proposition.  
*   Resolve pronouns (“he”, “they”) and deictic terms (“yesterday”, “here”) using the context.  
*   Extract only the information from the **Target utterance**. Use the **Context** only for resolving ambiguity.
*   If the target utterance yields no valid facts, return an empty array `[]`.  
*   Return **only** the JSON array (no extra text, no markdown).
*   Do not extract more than 30 claims. If there exist more than 30 claims, please condense some and only return the top 30 most important ones.
*   Note that occasionally the speaker might have a role, and would be displayed in the "()" like "John Donvan (moderator)" in the input dialogue. However, please consistently only keep the speaker's full name when doing extraction.

In-context examples
-------------------
### Example A — implicit lunch suggestion  
**Context**  
1. Speaker A: It’s lunchtime.

**Target utterance** 
2. Speaker B: Operator San has great Japanese bento sets.

**Output**  (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "Speaker B",
        "target_speaker": "Speaker A",
        "claim": "Operator San café serves good Japanese bento sets.",
        "turn_id": "2"
      },
      { "speaker": "Speaker B",
        "target_speaker": "Speaker A",
        "claim": "Operator San café is a good option for lunch.",
        "turn_id": "2"
      }
    ]
}

### Example B — multiple negative findings  
**Context**  
1. James (Doctor): Any fever or chills?

**Target utterance**
2. Amy (Patient): No.

**Output**  (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "Amy",
        "target_speaker": "James",
        "claim": "The patient does not have a fever.",
        "turn_id": "2"
      },
      { "speaker": "Amy",
        "target_speaker": "James",
        "claim": "The patient does not have chills.",
        "turn_id": "2"
      }
    ]
}

### Example C — conjunctive symptoms  
**Context**  
1. James (Doctor): Are you having any blurred vision, dizziness, or nausea?

**Target utterance**
2. Amy (Patient): I'm having blurry vision and lightheadedness.

**Output**  (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "Amy",
        "target_speaker": "James",
        "claim": "The patient is experiencing blurry vision.",
        "turn_id": "2"
      },
      { "speaker": "Amy",
        "target_speaker": "James",
        "claim": "The patient is experiencing lightheadedness.",
        "turn_id": "2"
      }
    ]
}

### Example D — gun-risk opinion with specifics  
**Context**  
1. Speaker 1: I hope my kids own guns.
2. Speaker 2: I am thinking the opposite.

**Target utterance**
3. Speaker 3: When I look at the statistics about how that adds to the risk of suicide, the risk of being misused, the risk of it being stolen, used in a domestic quarrel, I think it’s just too much of a risk.

**Output**  (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "Speaker 3",
        "target_speaker": "Everyone",
        "claim": "Having a gun increases the risk of suicide.",
        "turn_id": "3"
      },
      { "speaker": "Speaker 3",
        "target_speaker": "Everyone",
        "claim": "Having a gun increases the risk of misuse.",
        "turn_id": "3"
      },
      { "speaker": "Speaker 3",
        "target_speaker": "Everyone",
        "claim": "Having a gun increases the risk of theft.",
        "turn_id": "3"
      },
      { "speaker": "Speaker 3",
        "target_speaker": "Everyone",
        "claim": "Having a gun increases the risk of use in domestic quarrels.",
        "turn_id": "3"
      },
      { "speaker": "Speaker 3",
        "target_speaker": "Everyone",
        "claim": "Owning a gun is too risky.",
        "turn_id": "3"
      }
    ]
}

### Example E — reported statement  
**Context**  
1. John (moderator): Paul, were you surprised to hear Stephen say that police do not have a responsibility to protect individuals?
2. Paul (participant): Sorry, what did Stephen said?

**Target utterance**
3. John (moderator): Stephen said that police do not have a responsibility to protect individuals.

**Output**  (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "John",
        "target_speaker": "Paul",
        "claim": "Stephen said that police do not have a responsibility to protect individuals.",
        "turn_id": "3"
      }
    ]
}

### Example F — resolving ambiguity for self-containment
**Context**  
7. Bob: A 2023 Stanford study actually says remote work lowers productivity by 10 percent.

**Target utterance**
8. Carol: I read that study too; it concluded productivity was equal—Bob, you might have mixed up the numbers.

**Output** (List of extracted memory json objects)

{"memories":
    [
      { "speaker": "Carol",
        "target_speaker": "Bob",
        "claim": "The 2023 Stanford study on remote work concluded that productivity was equal.",
        "turn_id": "8"
      },
      { "speaker": "Carol",
        "target_speaker": "Bob",
        "claim": "Bob may have mixed up the numbers from the 2023 Stanford study.",
        "turn_id": "8"
      }
    ]
}

"""

MULTIPARTY_MEMORY_UPDATE_PROMPT = """
You are a **Multi‑Party Memory Consolidator**.  
For every newly‑extracted claim decide whether to **ADD**, **UPDATE** or mark
**NONE** in the store, using a natural‑language‑entailment (NLI).

────────────────────────────────────────────────
INPUT
────────────────────────────────────────────────
• `existing_memories`  : JSON array of stored proposition objects.  
• `newly_extracted_claims` : JSON array of new proposition objects.

────────────────────────────────────────────────
Chain-of-Thought Process  (A = new claim, B = existing memory)
────────────────────────────────────────────────

1. Search: First, search all existing_memories to find the single most semantically relevant existing memory (let's call it B), if one exists. Prioritize memories from the same speaker.

2. Analyze & Reason: If a relevant memory B is found, analyze the relationship between A and B. Clearly state the relationship (equivalent, forward_entail, contradiction, etc.).

3. Decide Action: Based on the speakers and the logical relationship, apply the decision principles below to determine the action.

4. Format Output: Generate a single JSON object for claim A reflecting your decision.

────────────────────────────────────────────────
DECISION RULES  (A = new claim, B = existing memory)
────────────────────────────────────────────────
For each pair ⟨A(new claim), B(existing memory)⟩ run the NLI classification (entailment / contradiction / neutral) for both way (A ⇒ B & B ⇒ A.
  
Map its output to one of:

   – **equivalent**        : A and B mutually entail  
   – **forward_entail**    : A ⇒ B only  
   – **backward_entail**   : B ⇒ A only  
   – **contradiction**     : A contradicts B  
   – **neutral**           : none of the above (or low confidence)
   
────────────────────────────────────────────────
PRIMARY TARGET‑SELECTION PRIORITY
────────────────────────────────────────────────
Evaluate A against *all* memories B₁…Bₙ.  Choose **exactly one** B (or none)
using this deterministic ladder:

1. **SAME speaker & equivalent**
2. **SAME speaker & backward_entail**  
3. **SAME speaker & (contradiction OR forward_entail)**  
4. **DIFFERENT speaker & any non‑neutral relation**  
5. *No eligible B*→ treat A as neutral (ADD, target=null)

If two candidates tie within a rung, pick the one with the **highest
confidence**.  Discard the rest (or log them elsewhere).

────────────────────────────────────────────────
ACTION MAPPING
────────────────────────────────────────────────

   Same speaker  
     · equivalent        → NONE  
     · forward_entail    → UPDATE  
     · contradiction     → UPDATE  
     · backward_entail   → NONE  
     · neutral           → ADD  

   Different speaker  
     → always ADD

4. UPDATE rule

    • If UPDATE with contradict: the "source" is the claim A. (replace B's claim with A's claim).
    • If UPDATE with forward_entail: the "source" is the new merged claim (A + B). (an elaboration).

5. **target field**  
   • UPDATE / NONE → `target` = B (mandatory)  
   • ADD → `target` is normally `null`; you *may* supply B when logging a
     cross‑speaker link (equivalent / entail / contradiction), as determined by the priority ladder.

────────────────────────────────────────────────
OUTPUT  (one object per new claim)
────────────────────────────────────────────────
{
  "memory_updates": [
    {
      "action": "<ADD|UPDATE|NONE>",
      "logical_relation": "<equivalent|forward_entail|backward_entail|contradiction|neutral>",
      "source":  { "speaker": "...", "target_speaker": "...", "claim": "...", "turn_id": "..." },
      "target":  null | { "id": "...", "speaker": "...", "target_speaker": "...", "claim": "...", "turn_id": "..." }
    }
  ]
}

Return `{ "memory_updates": [] }` if nothing changes.

────────────────────────────────────────────────
DATA QUALITY CHECKLIST
────────────────────────────────────────────────
✓ Rewrite claims to be context‑independent; remove hedges/filler  
✓ Resolve pronouns to specific names  
✓ Keep one atomic proposition per claim  
✓ `target_speaker` must denote a person or group, never an object  
✓ The target object must ONLY include the following fields: id, speaker, target_speaker, claim, turn_id. No other fields are allowed.
✓ Merge multiple UPDATEs from the same speaker that hit the same memory id


────────────────────────────────────────────────
EXAMPLE 1
────────────────────────────────────────────────
Existing memories
[
  { "id": "mem_011", "speaker": "Maria", "target_speaker": "Everyone",
    "claim": "The death penalty provides justice for victims' families.", "turn_id": "5" },
  { "id": "mem_012", "speaker": "Sam", "target_speaker": "Everyone",
    "claim": "The financial cost of death penalty appeals is very high.", "turn_id": "8" }
]

Newly‑extracted claims
[
  { "speaker": "Sam",   "target_speaker": "Everyone",
    "claim": "There is an irreversible risk of executing an innocent person.", "turn_id": "14" },
  { "speaker": "Sam",   "target_speaker": "Everyone",
    "claim": "The financial cost of death penalty appeals exceeds the cost of life imprisonment.", "turn_id": "14" },
  { "speaker": "Sam", "target_speaker": "Everyone",
    "claim": "The death penalty provides justice for victims' families.", "turn_id": "14" }
]

Expected output
{
  "memory_updates": [
    {
      "action": "ADD",
      "logical_relation": "neutral",
      "source":  { "speaker": "Sam", "target_speaker": "Everyone",
                   "claim": "There is an irreversible risk of executing an innocent person.", "turn_id": "14" },
      "target":  null
    },
    {
      "action": "UPDATE",
      "logical_relation": "forward_entail",
      "source":  { "speaker": "Sam", "target_speaker": "Everyone",
                   "claim": "The financial cost of death penalty appeals is very high and exceeds the cost of life imprisonment.", "turn_id": "14" },
      "target":  { "id": "mem_012", "speaker": "Sam", "target_speaker": "Everyone",
                   "claim": "The financial cost of death penalty appeals is very high.", "turn_id": "8" }
    },
    {
      "action": "Add",
      "logical_relation": "equivalent",
      "source":  { "speaker": "Sam", "target_speaker": "Everyone",
                   "claim": "The death penalty provides justice for victims' families.", "turn_id": "14" },
      "target":  { "id": "mem_011", "speaker": "Maria", "target_speaker": "Everyone",
                   "claim": "The death penalty provides justice for victims' families.", "turn_id": "5" }
    }
  ]
}

Explanation  
• 1st claim: new topic, same speaker → ADD, neutral.  
• 2nd claim: refines own earlier claim (A ⇒ B) from the same speaker → UPDATE, forward_entail, therefore merge the two claims. 
• 3rd claim: repeats one earlier claim (A ⇔ B) from different speaker → ADD, equivalent.

────────────────────────────────────────────────
EXAMPLE 2
────────────────────────────────────────────────
Existing memories

[
  { "id": "mem_101", "speaker": "Carrol", "claim": "Project Phoenix must launch by August 15th.", "turn_id": "5" },
  { "id": "mem_102", "speaker": "Bob", "claim": "The allocated budget for Project Phoenix is $50,000.", "turn_id": "8" },
  { "id": "mem_103", "speaker": "Alice", "claim": "The mobile app must have a dark mode feature.", "turn_id": "11" }
]

Newly‑extracted claims

[
  { "speaker": "Alice", "claim": "The project scope document states dark mode is a post-launch feature.", "turn_id": "22" },
  { "speaker": "Alice", "claim": "The revised budget for Project Phoenix is actually $55,000.", "turn_id": "22" },
  { "speaker": "Alice", "claim": "Given the scope clarification, the launch for Project Phoenix will be August 22nd.", "turn_id": "22" }
]

Expected output

{
  "memory_updates": [
    {
      "action": "UPDATE",
      "logical_relation": "contradiction",
      "source": { "speaker": "Alice", "claim": "The project scope document states dark mode is a post-launch feature.", "turn_id": "22" },
      "target": { "id": "mem_103", "speaker": "Alice", "claim": "The mobile app must have a dark mode feature.", "turn_id": "11" }
    },
    {
      "action": "ADD",
      "logical_relation": "contradiction",
      "source": { "speaker": "Alice", "claim": "The revised budget for Project Phoenix is actually $55,000.", "turn_id": "22" },
      "target": { "id": "mem_102", "speaker": "Bob", "claim": "The allocated budget for Project Phoenix is $50,000.", "turn_id": "8" }
    },
    {
      "action": "ADD",
      "logical_relation": "contradiction",
      "source": { "speaker": "Alice", "claim": "Given the scope clarification, the launch for Project Phoenix will be August 22nd.", "turn_id": "22" },
      "target": { "id": "mem_101", "speaker": "Carrol", "claim": "Project Phoenix must launch by August 15th.", "turn_id": "5" }
    }
  ]
}

Explanation

- 1st Claim: Alice's claim about "dark mode" contradicts Alice's earlier claim (mem_103). Because the claims are from the same speakers, the action must be UPDATE to correct the earlier claim.

- 2nd Claim: Alice's claim about project budget contradicts with Bob's earlier claim, since they are from different speakers, the action must be ADD to record the different views.

- 3rd Claim: Alice's claim about project launch date contradicts with Carrol's earlier claim, since they are from different speakers, the action must be ADD to record the different views.

────────────────────────────────────────────────
EXAMPLE 3
────────────────────────────────────────────────

Existing memories

[
  { "id": "mem_210", "speaker": "David", "claim": "Jane Doe has 5 years of Python experience.", "turn_id": "4" },
  { "id": "mem_211", "speaker": "Emily", "claim": "The candidate must have experience with a major cloud platform.", "turn_id": "7" },
  { "id": "mem_212", "speaker": "Frank", "claim": "Jane Doe's final interview is scheduled for this Wednesday.", "turn_id": "9" }
]

Newly‑extracted claims

[
  { "speaker": "Emily", "claim": "Correction: Jane Doe's final interview is on Friday, not Wednesday.", "turn_id": "15" },
  { "speaker": "Emily", "claim": "Jane's 5 years of Python experience is a major asset.", "turn_id": "15" },
  { "speaker": "Emily", "claim": "We must make a final hiring decision by next Monday.", "turn_id": "15" }
]

Expected output

{
  "memory_updates": [
    {
      "action": "ADD",
      "logical_relation": "contradiction",
      "source": { "speaker": "Emily", "claim": "Correction: Jane Doe's final interview is on Friday, not Wednesday.", "turn_id": "15" },
      "target": { "id": "mem_212", "speaker": "Frank", "claim": "Jane Doe's final interview is scheduled for this Wednesday.", "turn_id": "9" }
    },
    {
      "action": "ADD",
      "logical_relation": "equivalent",
      "source": { "speaker": "Emily", "claim": "Jane's 5 years of Python experience is a major asset.", "turn_id": "15" },
      "target": { "id": "mem_210", "speaker": "David", "claim": "Jane Doe has 5 years of Python experience.", "turn_id": "4" }
    },
    {
      "action": "ADD",
      "logical_relation": "neutral",
      "source": { "speaker": "Emily", "claim": "We must make a final hiring decision by next Monday.", "turn_id": "15" },
      "target": null
    }
  ]
}

Explanation

- 1st Claim: This is a clear different-speaker contradiction, targeting Frank's earlier memory (mem_212). The action is correctly identified as ADD.

- 2nd Claim: Emily's claim is semantically equivalent to David's existing memory (mem_210). Because they are different speakers, her agreement is ADDed as a new memory, but linked to David's original statement to show consensus.

- 3rd Claim: This claim about the "hiring decision" deadline is a new piece of information. The model correctly searches the memory, finds no semantically related claims, and processes it as a simple ADD with a neutral relation and a null target.

### Task

**Input Data**
------------
**existing_memories**

{{retrieved_old_memory_dict}}

**newly_extracted_claims**

{{new_retrieved_claims_list}}

**output**
------------
"""