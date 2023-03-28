import numpy as np
import json
import os 

def exact_match_a2p_ratio(p, l):
    cnt = 0
    for ip, il in zip(p, l):
        if ip == il:
            cnt += 1
        else:
            break
    l  = max(len(l), len(p))
    # if empty label and prediction,- match
    if cnt == 0 and l == 0:
        return 1.0
    r =  cnt / l
    return r

def average_exact_match_a2p_ratio(predictions, labels):
    return sum(
        exact_match_a2p_ratio(p, l)
        for p, l in zip(predictions, labels)
    ) / len(labels)

def exact_matches_ratio(a, b, include_empty_matches):
    def comp(a, b):
        if not include_empty_matches and (len(a) == 0 and len(b) == 0):
            return False
        return a == b
    return sum(comp(el_a, el_b) for el_a, el_b in zip(a, b)) / len(a)

def compute_metrics(preds, ctx):
    label_ids = np.where(preds.label_ids != -100, preds.label_ids, ctx.tokenizer.pad_token_id)
    labels = ctx.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    predictions = ctx.tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    inputs = ctx.tokenizer.batch_decode(preds.inputs)
    # if opt.strip_new_lines_for_em:
    #     def strip_get_first_line(items):
    #         items = [el.splitlines() for el in items]
    #         items_first_line = [el[0] for el in items]
    #         items = [''.join(el) for el in items]
    #         return items, items_first_line
    #     labels, labels_first_line = strip_get_first_line(labels)
    #     predictions, predictions_first_line = strip_get_first_line(predictions)
    # else:
    def get_first_lines(vals):
        return [
            lines[0] if len(lines) > 0 else ''
            for lines in (
                el.splitlines()
                for el in vals
            )
        ]
    labels_first_line = get_first_lines(labels)
    predictions_first_line = get_first_lines(predictions)

    # TODO: get several exactly matched and several not matched samples instead
    # for now save just 500 of first examples
    sz = min(500, len(inputs))
    examples = {
        'full': [
            {'input': input, 'label': label, 'prediction': prediction}
            for input, label, prediction in zip(inputs[:sz], labels[:sz], predictions[:sz])
        ],
        'first_line': [
                {'input': input, 'label': label, 'prediction': prediction}
            for input, label, prediction in zip(inputs[:sz], labels_first_line[:sz], predictions_first_line[:sz])
        ]
    }
    if ctx.opt.is_main:
        pid = os.getpid()
        step = len(list(ctx.examples_dir.glob('*.json')))
        example_file = ctx.examples_dir / f'{step}_{pid}.json'
        with example_file.open('wt') as f:
            json.dump(examples, f)

    return {
        #'em_ratio': exact_matches_ratio(predictions, labels),
        #'em_a2p_ratio': average_exact_match_a2p_ratio(predictions, labels),
        'em_first_line_ratio': exact_matches_ratio(labels_first_line, predictions_first_line, include_empty_matches=True),
        'em_first_line_ratio_wo_empty_matches': exact_matches_ratio(labels_first_line, predictions_first_line, include_empty_matches=False),
        #'examples': str(examples)
    }