Multi-prompt boundary proposals

Diversity prompts (p runs, typically 3–5):

Same system message (“You are an expert dialogue analyst…”)

Vary one dimension per run—e.g. temperature, few-shot examples, or phrasing (“identify major topic shifts” vs “detect cohesive spans”).

Expected output: ordered list of closed intervals [sᵢ, eᵢ] covering all indices and non-overlapping.

Voting & normalisation

B_j be the set of boundary indices proposed in run j (a boundary is the start of an interval except the first).

votes[k] accumulate points for utterance index k.

for each run j in 1…p:
    for each boundary b in B_j:
        votes[b]   += 2           # direct hit
        votes[b-1] += 1           # left neighbour (if ≥1)
        votes[b+1] += 1           # right neighbour (if ≤n)
max_pts = p * 2                   # upper bound any index can get
score[k] = votes[k] / max_pts     # 0.0 – 1.0
