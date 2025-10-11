import os
import json
from dotenv import load_dotenv

# Import your new MultipartyMemory class
from memory.multiparty_memory import MultipartyMemory

# Make sure this file exists and contains your prompts
# from multiparty_prompts import *

load_dotenv()

# --- Configuration ---
CONTEXT_WINDOW_SIZE = 5  # How many previous turns to use as context

# Initialize our specialized memory class
memory = MultipartyMemory(backend="openai")


def main():
    """
    A test script to demonstrate the MultipartyMemory class.
    It processes a sample dialogue, runs search queries, and
    generates a contextual briefing for an upcoming segment.
    """

    # ── 1.  Define the full multi-party transcript ──────────────────────────────
    conversation_id = "meeting-on-remote-work-042"

    full_dialogue = [
        # ... (The full dialogue from index 0 to 14 remains here) ...
        {"index": 0, "name": "Alice", "content": "I believe remote work actually boosts overall productivity."},
        {"index": 1, "name": "Bob",
         "content": "From what I've seen in my team, remote work hurts day-to-day collaboration."},
        {"index": 2, "name": "Carol",
         "content": "Recent surveys suggest remote workers produce about 15 percent more output than office workers."},
        {"index": 3, "name": "Alice",
         "content": "To clarify, the biggest upside I see is flexibility—productivity might be only slightly higher."},
        {"index": 4, "name": "Bob",
         "content": "Yes, flexibility improves, but collaboration drops and that drags productivity down overall."},
        {"index": 5, "name": "Alice",
         "content": "I’m retracting my earlier productivity claim—new data shows remote and office productivity are about the same."},
        {"index": 6, "name": "Carol",
         "content": "Even if productivity evens out, remote workers often feel isolated and disengaged."},
        {"index": 7, "name": "Bob",
         "content": "A 2023 Stanford study actually says remote work lowers productivity by 10 percent."},
        {"index": 8, "name": "Carol",
         "content": "I read that study too; it concluded productivity was equal—Bob, you might have mixed up the numbers."},
        {"index": 9, "name": "Bob",
         "content": "You’re right, Carol—my mistake. The study shows no net productivity loss, just collaboration challenges."},
        {"index": 10, "name": "Alice",
         "content": "For managers, using clear OKRs and async updates can close that collaboration gap."},
        {"index": 11, "name": "Carol",
         "content": "But constant monitoring software can erode trust and increase stress."},
        {"index": 12, "name": "Bob",
         "content": "Regardless, we all seem to agree remote work greatly improves work-life balance."},
        {"index": 13, "name": "Alice",
         "content": "Absolutely—flexibility alone makes a strong case for remote options."},
        {"index": 14, "name": "Carol",
         "content": "I’m still cautious; we need more long-term data before making sweeping policy changes."}
    ]

    # --- 2.  Split dialogue for a realistic test scenario ───────────────────────

    # Define the segment we want to generate a briefing FOR.
    # In a real application, this would be the next incoming chunk of conversation.
    upcoming_segment = full_dialogue[12:]  # Turns 12, 13, 14

    # The historical memory should ONLY contain information from BEFORE this segment.
    historical_dialogue = full_dialogue[:12]  # Turns 0 through 11

    # --- 3.  Populate memory with ONLY the historical dialogue ──────────────────
    print(f"Processing {len(historical_dialogue)} historical turns for conversation_id: '{conversation_id}'...\n")
    for i, turn in enumerate(historical_dialogue):
        start_index = max(0, i - CONTEXT_WINDOW_SIZE)
        context = historical_dialogue[start_index:i]
        print(f"  -> Adding turn {turn['index']} by {turn['name']}...")
        memory.add(
            target_utterance=turn,
            context=context,
            run_id=conversation_id,
            metadata={"topic": "pros and cons of remote working"}
        )

    # --- 4.  (Optional) Inspect the populated historical memory ────────────────
    print("\n\n--- Full Memory Store Dump (Historical Only) ---")
    dump = memory.get_all(run_id=conversation_id)
    all_memories = sorted(dump.get('results', []), key=lambda x: x.get('turn_id', 999))
    for row in all_memories:
        row_meta = row.get('metadata', {})
        speaker = row.get('actor_id', 'N/A')
        turn_id = row_meta.get('turn_id', '?')
        print(f"Turn {turn_id:>2} | {speaker:>7}: {row['memory']}")

    # =============================================================================
    # ── 5.  Test the Contextual Briefing Generation ──────────────────────────────
    # =============================================================================
    print("\n\n--- Generating Contextual Briefing for an Upcoming Segment ---")

    upcoming_segment = full_dialogue[12:]
    # For both strategies, the context is what came before the segment.
    context_before_segment = full_dialogue[:12]

    print("\nPreparing briefing for upcoming segment:")
    for turn in upcoming_segment:
        print(f"  (Turn #{turn['index']}) {turn['name']}: {turn['content']}")

    # --- Strategy 1: Fast Mode (Default) ---
    print("\n\n--- Testing with Fast Strategy (use_claim_extraction=False) ---")
    fast_briefing_output = memory.generate_contextual_briefing(
        upcoming_segment=upcoming_segment,
        context=context_before_segment,  # Context is unused here but good practice to pass
        run_id=conversation_id,
        use_claim_extraction=False
    )
    print("\n✅ Generated Briefing (Fast Mode):")
    print("------------------------------------")
    print(fast_briefing_output["briefing"])
    print("------------------------------------")

    # --- Strategy 2: High-Precision Mode ---
    print("\n\n--- Testing with High-Precision Strategy (use_claim_extraction=True) ---")
    # Note: This will be slower and make more API calls
    high_precision_output = memory.generate_contextual_briefing(
        upcoming_segment=upcoming_segment,
        context=context_before_segment,  # Context is required for this mode
        run_id=conversation_id,
        use_claim_extraction=True
    )
    print("\n✅ Generated Briefing (High-Precision Mode):")
    print("------------------------------------")
    print(high_precision_output["briefing"])
    print("------------------------------------")

    # You can compare the retrieval_data from both outputs to see the difference
    print("\n\nComparison of retrieved memories:")
    print(f"Fast mode retrieved {len(fast_briefing_output['retrieval_data'])} unique memories.")
    print(f"High-precision mode retrieved {len(high_precision_output['retrieval_data'])} unique memories.")


if __name__ == "__main__":
    main()