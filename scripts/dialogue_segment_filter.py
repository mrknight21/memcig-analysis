import re


def rate_segment_informativeness(segment: list[dict], verbose: bool = False) -> float:
    """
    Rates a dialogue segment based on heuristics for informativeness in a debate.

    The scoring is based on:
    - Argumentation keywords (e.g., 'because', 'evidence')
    - Disagreement/contrast keywords (e.g., 'however', 'but')
    - Questions that prompt for more information.
    - Penalties for conversational filler and unsupported assertions.
    - A bonus for more speaker interaction.

    Args:
        segment: A list of utterance dictionaries. Each dictionary should have
                 'name', 'content', and 'index' keys.
        verbose: If True, prints the scoring breakdown for the segment.

    Returns:
        A float representing the segment's informativeness score.
    """
    score = 0.0

    # --- Keyword Definitions ---
    # These keywords suggest reasoning and evidence-based claims.
    argumentation_keywords = [
        'because', 'since', 'therefore', 'consequently', 'hence', 'thus',
        'evidence', 'data', 'study', 'research', 'shows that', 'proves that',
        'for example', 'for instance', 'specifically'
    ]

    # These keywords indicate disagreement, rebuttal, or contrasting viewpoints.
    disagreement_keywords = [
        'disagree', 'however', 'but', 'actually', 'on the contrary', 'not true',
        'i don\'t think', 'that\'s not the point', 'on the other hand'
    ]

    # These phrases often precede personal opinions without backing.
    unsupported_assertion_keywords = [
        'i feel', 'i believe', 'i think', 'in my opinion', 'obviously',
        'everyone knows', 'it seems to me'
    ]

    # Common conversational fillers that add little substance.
    filler_words = [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean'
    ]

    # --- Analysis Variables ---
    num_utterances = len(segment)
    if num_utterances == 0:
        return 0.0

    total_words = 0
    speakers = set()
    last_speaker = None
    speaker_turns = 0

    if verbose:
        print(f"\n--- Rating Segment (Utterances: {num_utterances}) ---")

    # --- Iterate Through Utterances for Scoring ---
    for utterance in segment:
        content = utterance.get('content', '').lower()
        speaker_name = utterance.get('name', '')

        # Ignore moderators if they are marked in the name
        if "(moderator)" in speaker_name.lower():
            continue

        speakers.add(speaker_name)
        total_words += len(content.split())

        # Track speaker changes (interaction bonus)
        if last_speaker is not None and speaker_name != last_speaker:
            speaker_turns += 1
        last_speaker = speaker_name

        # 1. Score for Argumentation (High positive impact)
        arg_score = sum(1.5 for keyword in argumentation_keywords if keyword in content)
        score += arg_score
        if verbose and arg_score > 0:
            print(f"  [+] Argumentation bonus: +{arg_score:.1f} for utterance: \"{content[:50]}...\"")

        # 2. Score for Disagreement (High positive impact)
        dis_score = sum(1.5 for keyword in disagreement_keywords if keyword in content)
        score += dis_score
        if verbose and dis_score > 0:
            print(f"  [+] Disagreement bonus: +{dis_score:.1f} for utterance: \"{content[:50]}...\"")

        # 3. Score for Questions (Moderate positive impact)
        # Looks for sentences ending in a question mark.
        if '?' in content:
            q_score = 1.0
            score += q_score
            if verbose:
                print(f"  [+] Question bonus: +{q_score:.1f} for utterance: \"{content[:50]}...\"")

        # 4. Penalty for Unsupported Assertions (Negative impact)
        unsupported_score = sum(-0.5 for keyword in unsupported_assertion_keywords if keyword in content)
        score += unsupported_score
        if verbose and unsupported_score < 0:
            print(f"  [-] Unsupported penalty: {unsupported_score:.1f} for utterance: \"{content[:50]}...\"")

        # 5. Penalty for Filler Words (Small negative impact)
        filler_score = sum(-0.2 for filler in filler_words if re.search(r'\b' + filler + r'\b', content))
        score += filler_score
        if verbose and filler_score < 0:
            print(f"  [-] Filler penalty: {filler_score:.1f} for utterance: \"{content[:50]}...\"")

    # --- Post-loop Adjustments ---

    # 6. Interaction Bonus: More turns per utterance is better.
    interaction_bonus = 0
    if num_utterances > 1:
        interaction_bonus = (speaker_turns / (num_utterances - 1)) * 2.0
        score += interaction_bonus
        if verbose:
            print(
                f"  [+] Interaction bonus: +{interaction_bonus:.2f} ({speaker_turns} turns over {num_utterances} utterances)")

    # 7. Normalization: Adjust score based on length to prevent long, rambling
    # segments from scoring high just by chance of having more keywords.
    # We normalize by the number of utterances.
    final_score = score / num_utterances if num_utterances > 0 else 0.0

    if verbose:
        print(f"--- Sub-total Score: {score:.2f} ---")
        print(f"--- Final Score (Normalized by utterance count): {final_score:.2f} ---")

    return final_score


# --- Example Usage ---

# Example Segment 1: Good, substantive debate
segment_good = [
    {"index": 1, "name": "Alice (Participant)",
     "content": "I disagree with the proposed policy because the data shows it disproportionately affects low-income families."},
    {"index": 2, "name": "Bob (Participant)",
     "content": "But how can you say that? The study you're citing is from a decade ago. More recent evidence actually suggests the opposite."},
    {"index": 3, "name": "Alice (Participant)",
     "content": "For example, a 2023 report from the National Institute of Economics proves my point. What is your source?"}
]

# Example Segment 2: Poor, low-substance conversation
segment_poor = [
    {"index": 1, "name": "Charlie (Participant)",
     "content": "Well, I think the whole thing is just a bad idea. It feels wrong."},
    {"index": 2, "name": "David (Participant)",
     "content": "Yeah, I mean, you know, it's like, obviously not going to work."},
    {"index": 3, "name": "Charlie (Participant)", "content": "Um, totally. I just believe we should do something else."}
]

# Example Segment 3: A mix of opinion and questions
segment_mixed = [
    {"index": 1, "name": "Eve (Participant)",
     "content": "In my opinion, the environmental impact is the most critical factor here."},
    {"index": 2, "name": "Frank (Participant)",
     "content": "Why do you think that's more important than the economic consequences?"},
    {"index": 3, "name": "Eve (Participant)",
     "content": "Because the long-term cost of environmental damage will eventually outweigh any short-term economic gain."}
]

if __name__ == '__main__':
    print("=" * 40)
    print("RATING A GOOD SEGMENT (VERBOSE MODE)")
    print("=" * 40)
    score_good = rate_segment_informativeness(segment_good, verbose=True)
    print(f"\nFINAL SCORE for Good Segment: {score_good:.2f}")

    print("\n" + "=" * 40)
    print("RATING A POOR SEGMENT (VERBOSE MODE)")
    print("=" * 40)
    score_poor = rate_segment_informativeness(segment_poor, verbose=True)
    print(f"\nFINAL SCORE for Poor Segment: {score_poor:.2f}")

    print("\n" + "=" * 40)
    print("RATING A MIXED SEGMENT (VERBOSE MODE)")
    print("=" * 40)
    score_mixed = rate_segment_informativeness(segment_mixed, verbose=True)
    print(f"\nFINAL SCORE for Mixed Segment: {score_mixed:.2f}")

    # Example of how you might use this in a loop
    all_segments = [segment_good, segment_poor, segment_mixed]
    scored_segments = []
    for i, seg in enumerate(all_segments):
        seg_score = rate_segment_informativeness(seg)  # Verbose is off for batch processing
        scored_segments.append((i, seg_score))

    # Sort segments by score in descending order
    scored_segments.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 40)
    print("SEGMENTS RANKED BY INFORMATIVENESS SCORE")
    print("=" * 40)
    for index, score in scored_segments:
        print(f"Segment {index + 1}: Score = {score:.2f}")

