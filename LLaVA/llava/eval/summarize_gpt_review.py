import json
import os
from collections import defaultdict
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-d', '--dir', default=None)
    parser.add_argument('-v', '--version', default=None)
    parser.add_argument('-s', '--select', nargs='*', default=None)
    parser.add_argument('-f', '--files', nargs='*', default=[])
    parser.add_argument('-i', '--ignore', nargs='*', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Starting evaluation script for {args.files}")

    if args.ignore is not None:
        args.ignore = [int(x) for x in args.ignore]

    if len(args.files) > 0:
        review_files = args.files
    else:
        review_files = [x for x in os.listdir(args.dir) if x.endswith('.jsonl') and (x.startswith('gpt4_text') or x.startswith('reviews_') or x.startswith('review_') or 'review' in args.dir)]

    # Initialize log with run info
    print("Starting evaluation run")
    print(f"Arguments: {args}")
    print(f"Processing {len(review_files)} review files")

    for review_file in sorted(review_files):
        config = os.path.basename(review_file).replace('gpt4_text_', '').replace('.jsonl', '')
        if args.select is not None and any(x not in config for x in args.select):
            continue
        if '0613' in config:
            version = '0613'
        else:
            version = '0314'
        if args.version is not None and args.version != version:
            continue
        scores = defaultdict(list)
        print(f"Processing file: {config}")
        
        with open(os.path.join(args.dir, review_file) if args.dir is not None else review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                if review['question_id'] in args.ignore:
                    continue
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            result_text = f"{k} {round(stats[1]/stats[0]*100, 1)} {round(stats[0] * 10, 1)} {round(stats[1] * 10, 1)}"
            print(result_text)
        
        separator = "================================="
        print(separator)
    
    print("Evaluation completed")