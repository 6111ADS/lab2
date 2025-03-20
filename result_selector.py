def remove_duplicates(tuples):
    # Remove exact duplicates, keeping the one with the highest confidence
    seen = {}
    for subj, obj, relation, confidence in tuples:
        if (subj, obj, relation) not in seen or seen[(subj, obj, relation)] < confidence:
            seen[(subj, obj, relation)] = confidence
    return [(subj, obj, relation, confidence) for (subj, obj, relation), confidence in seen.items()]

def select_top_k_tuples(tuples, k, model):
    # Sort tuples by confidence (for SpanBERT) or arbitrary order (for Gemini)
    if model == '-spanbert':
        sorted_tuples = sorted(tuples, key=lambda x: x[3], reverse=True)  # Sort by confidence
    else:
        sorted_tuples = tuples  # For Gemini, order doesn't matter
    return sorted_tuples[:k]
