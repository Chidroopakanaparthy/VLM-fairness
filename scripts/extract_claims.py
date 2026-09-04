"""
Claim decomposition for RFW VLM fairness study.
Splits each model_output into atomic descriptive claims about facial features.
Handles per-model format differences: LLaVA tends toward itemized lists,
Qwen tends toward prose. Verdict/conclusion sentences are stripped out --
we only want claims that assert something checkable about the images,
not the final same/different judgment.
"""
import pandas as pd
import re

ATTR_KEYWORDS = [
    'eye', 'nose', 'mouth', 'lip', 'chin', 'jaw', 'jawline', 'cheek', 'cheekbone',
    'forehead', 'eyebrow', 'brow', 'hair', 'skin', 'ear', 'face shape', 'facial structure',
    'complexion', 'beard', 'mustache', 'moustache', 'facial hair', 'nostril', 'temple',
]

VERDICT_MARKERS = re.compile(
    r'^(final answer|therefore,?\s|in conclusion|given the (consistency|differences)|'
    r'my (final |)answer is|i (would|can) (conclude|say)|based on (this|these) (analysis|comparison))',
    re.IGNORECASE
)

META_INTRO = re.compile(
    r'^(based on the images provided|here are the specific|the following (features|observations)|'
    r'(however, )?i (can|will|would) (attempt to |)(describe|list|identify|outline)|'
    r'i can attempt to describe)',
    re.IGNORECASE
)

def strip_verdict_tail(text, variant):
    """Remove the concluding verdict portion, keep only descriptive content."""
    text = text.replace('**', '')  # markdown bold leaks into "**Image 1:**" style headers
    if variant == 2:
        # Cut at 'Step 3' if present -- that's the verdict step
        m = re.search(r'\bStep\s*3\s*[:.]', text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    # Also cut a trailing "Final Answer: ..." line regardless of variant
    text = re.split(r'\bFinal Answer\s*[:.]', text, flags=re.IGNORECASE)[0]
    return text.strip()

def split_into_chunks(text):
    """Split into (text, image_tag) pairs, where image_tag is 'image1'/'image2' when
    a claim is traceable to one specific photo, or None (comparative / unattributable)
    otherwise. Handles numbered lists, dashed lists, and numbered lists that nest
    dashed sub-bullets under an "N. Image M:" header -- all three occur in the raw
    data, and the third only shows up in ~5% of LLaVA V1 outputs, easy to miss."""
    lines = text.split('\n')
    numbered = [l for l in lines if re.match(r'^\s*\d+\.\s', l)]
    dashed = [l for l in lines if re.match(r'^\s*-\s', l)]

    if len(numbered) >= 2 or len(dashed) >= 2:
        chunks = []
        state_image = None  # set by a bare/header-only "Image N:" line (numbered or plain)
        cur_text, cur_tag = None, None

        def flush():
            if cur_text is not None and cur_text.strip():
                chunks.append((cur_text.strip(), cur_tag))

        for l in lines:
            bare = re.match(r'^\s*Image\s*([12])\s*:\s*$', l, re.IGNORECASE)
            if bare:
                flush()
                cur_text = None
                state_image = 'image' + bare.group(1)
                continue

            num_m = re.match(r'^\s*\d+\.\s*(.*)$', l)
            dash_m = re.match(r'^\s*-\s*(.*)$', l)
            marker_content = None
            if num_m:
                marker_content = num_m.group(1)
            elif dash_m:
                marker_content = dash_m.group(1)

            if marker_content is not None:
                # is this marker line itself just an "Image N:" header (nothing else)?
                header_only = re.match(r'^Image\s*([12])\s*:\s*$', marker_content, re.IGNORECASE)
                if header_only:
                    flush()
                    cur_text = None
                    state_image = 'image' + header_only.group(1)
                    continue
                flush()
                inline = re.match(r'^Image\s*([12])\s*:\s*', marker_content, re.IGNORECASE)
                if inline:
                    cur_tag = 'image' + inline.group(1)
                    marker_content = marker_content[inline.end():]
                elif re.match(r'^(similarities|differences|comparing features|comparison)\s*:', marker_content, re.IGNORECASE):
                    cur_tag = None
                else:
                    cur_tag = state_image
                cur_text = marker_content
            else:
                if l.strip() == '' or re.match(
                    r'^\s*Step\s*\d+\s*:\s*(Comparing Features( Across Both Images)?|Similarities|Differences|Observations?( of .+)?|Supporting [Ff]eatures|Contradicting [Ff]eatures)?\s*:?\s*$',
                    l, re.IGNORECASE
                ):
                    # a blank line (or a bare step header) always closes out whatever
                    # chunk was accumulating -- content after it is a new section and
                    # must not silently inherit the previous bullet's image tag
                    flush()
                    cur_text = None
                    continue
                if cur_text is None:
                    # plain prose appearing outside any bullet/header (e.g. trailing
                    # commentary after a numbered list ends) -- comparative by default,
                    # not attributable to whichever image the last bullet happened to be
                    cur_text = l
                    cur_tag = None
                else:
                    cur_text += ' ' + l
        flush()
        return chunks

    # No bullets: split on blank-line paragraph breaks, tag each paragraph from a
    # leading "Image N" mention (handles both "Image 1:" and "In Image 1," phrasing,
    # and tolerates a "Step N:" prefix before it).
    paragraphs = re.split(r'\n\s*\n', text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        p_search = re.sub(r'^Step\s*\d+\s*:\s*', '', p, flags=re.IGNORECASE)
        m = re.match(r'^(in\s+)?image\s*([12])\s*[:,]?\s*', p_search, re.IGNORECASE)
        if m:
            hint = 'image' + m.group(2)
            cleaned = p_search[m.end():].strip()
        else:
            hint = None
            cleaned = p_search
        result.append((cleaned, hint))
    if result:
        return result

    # Fallback: whole text is one chunk (prose)
    return [(text.strip(), None)] if text.strip() else []

def clean_chunk(chunk):
    """Strip leading 'Image 1:' / 'Image 2:' / 'Step N: <title>' labels that leak into claim text."""
    chunk = chunk.strip()
    chunk = re.sub(r'^Image\s*[12]\s*:\s*', '', chunk)
    chunk = re.sub(r'^Step\s*\d+\s*:\s*(Observations? of Key Facial Features\s*:?\s*)?', '', chunk, flags=re.IGNORECASE)
    # strip leading bold markdown attribute label, e.g. "**Eyes**: " -> ""
    chunk = re.sub(r'^\*\*[^*]+\*\*\s*:\s*', '', chunk)
    return chunk.strip()

def split_sentences(chunk):
    # crude but adequate sentence splitter for this text style
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
    return [s.strip() for s in sents if s.strip()]

def tag_attribute(claim_text):
    low = claim_text.lower()
    first_pos, first_kw = None, 'other'
    for kw in ATTR_KEYWORDS:
        pos = low.find(kw)
        if pos != -1 and (first_pos is None or pos < first_pos):
            first_pos, first_kw = pos, kw
    return first_kw

def is_claim(sentence):
    s = sentence.strip()
    if len(s) < 8:
        return False
    if VERDICT_MARKERS.search(s):
        return False
    if 'final answer' in s.lower():
        return False
    if META_INTRO.match(s):
        return False
    # header-only fragments that survived chunking, e.g. "Comparing Features:", "Step 2:"
    if re.match(r'^(Comparing Features( Across Both Images)?|Similarities|Differences|Observations?( of .+)?|Step\s*\d+|Supporting [Ff]eatures|Contradicting [Ff]eatures)\s*:?\s*$', s, re.IGNORECASE):
        return False
    # in this dataset, real content claims end in a period (or nothing); a bare
    # trailing colon always marks a list-intro header ("...features that support
    # this conclusion:"), never a checkable assertion -- covers header phrasings
    # not explicitly enumerated above.
    if s.endswith(':'):
        return False
    return True

def resolve_image_ref(sentence, running_context):
    """Final image_ref for a claim. A mid-paragraph 'In Image 2,' at the START of this
    specific sentence always wins (it's an explicit context switch). Otherwise the
    running context carried over from the chunk/previous sentence is used. Only when
    neither exists do we fall back to scanning the sentence body for a lone mention --
    risky in general (a sentence can reference the OTHER image in passing, e.g.
    "...consistent with Image 1" while describing Image 2), so this fallback is a
    last resort, not a first choice."""
    m = re.match(r'^(in\s+)?image\s*([12])\s*[:,]?\s*', sentence, re.IGNORECASE)
    if m:
        return 'image' + m.group(2), sentence[m.end():].strip()
    if running_context:
        return running_context, sentence
    has1 = bool(re.search(r'\bimage\s*1\b', sentence, re.IGNORECASE))
    has2 = bool(re.search(r'\bimage\s*2\b', sentence, re.IGNORECASE))
    if has1 and not has2:
        return 'image1', sentence
    if has2 and not has1:
        return 'image2', sentence
    return 'both', sentence

def extract_claims(model_output, prompt_variant):
    text = strip_verdict_tail(str(model_output), prompt_variant)
    chunks = split_into_chunks(text)
    claims = []
    for chunk_text, chunk_hint in chunks:
        chunk_text = clean_chunk(chunk_text)
        running_context = chunk_hint  # each new chunk/paragraph resets to its own default;
                                       # a mid-chunk "In Image N," sentence overrides it from there on
        for sent in split_sentences(chunk_text):
            sent = re.sub(r'^Step\s*\d+\s*:\s*', '', sent).strip()
            if not is_claim(sent):
                continue
            image_ref, sent = resolve_image_ref(sent, running_context)
            running_context = image_ref if image_ref != 'both' else running_context
            claims.append((sent.rstrip('.') if not sent.rstrip().endswith('.') else sent, image_ref))
    return claims

def process(df):
    rows = []
    for _, r in df.iterrows():
        claims = extract_claims(r['model_output'], r['prompt_variant'])
        for i, (c, image_ref) in enumerate(claims):
            rows.append({
                'claim_id': f"{r['model_name']}_{r['pair_id']}_{r['prompt_variant']}_{i}",
                'pair_id': r['pair_id'],
                'ethnicity': r['ethnicity'],
                'ground_truth_label': r['ground_truth_label'],
                'model_name': r['model_name'],
                'prompt_variant': r['prompt_variant'],
                'image1_filename': r['image1_filename'],
                'image2_filename': r['image2_filename'],
                'verifier_score_margin_arcface': r['verifier_score_margin_arcface'],
                'verifier_score_margin_adaface': r['verifier_score_margin_adaface'],
                'claim_text': c,
                'attribute_type': tag_attribute(c),
                'image_ref': image_ref,
            })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Decompose VLM outputs into atomic claims.')
    parser.add_argument('--qwen',   default='results_data/qwen_full_results.csv',
                        help='Path to qwen_full_results.csv')
    parser.add_argument('--llava',  default='results_data/llava_next_full_results.csv',
                        help='Path to llava_next_full_results.csv')
    parser.add_argument('--pairs',  default='results_data/pairs_locked.csv',
                        help='Path to pairs_locked.csv')
    parser.add_argument('--out',    default='Phase 3/claims.csv',
                        help='Output path for claims CSV')
    args = parser.parse_args()

    qwen  = pd.read_csv(args.qwen)
    llava = pd.read_csv(args.llava)
    pairs = pd.read_csv(args.pairs)

    claims_qwen  = process(qwen)
    claims_llava = process(llava)
    claims = pd.concat([claims_qwen, claims_llava], ignore_index=True)

    # attach image paths if available in pairs
    if 'image1_path' in pairs.columns and 'image2_path' in pairs.columns:
        path_map = pairs.set_index('pair_id')[['image1_path', 'image2_path']]
        claims = claims.join(path_map, on='pair_id')

    claims.to_csv(args.out, index=False)

    print("TOTAL CLAIMS:", len(claims))
    print("\nBy model:")
    print(claims.groupby('model_name').size())
    print("\nBy model x variant, mean claims/explanation:")
    n_explanations = pd.concat([qwen, llava]).groupby(['model_name', 'prompt_variant']).size()
    n_claims = claims.groupby(['model_name', 'prompt_variant']).size()
    print((n_claims / n_explanations).round(2))
    print("\nBy ethnicity:")
    print(claims.groupby('ethnicity').size())
    print("\nAttribute type distribution:")
    print(claims['attribute_type'].value_counts())
