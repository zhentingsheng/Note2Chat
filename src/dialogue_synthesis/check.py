import os
import json
import argparse
from evaluation.sentence_evaluator import SentenceEvaluator



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sample_dir',type=str, required=True, help="sample_dir")
    parser.add_argument('--model_id',type=str, required=True, help="model_id")
    parser.add_argument('--conv_key',type=str, required=True, help="conv_key")
    parser.add_argument('--eval_key',type=str, required=True, help="eval_key")

    args = parser.parse_args()
    sample_dir = args.sample_dir
    model_id = args.model_id
    conv_key = args.conv_key
    eval_key = args.eval_key

    sentence_evaluator = SentenceEvaluator(model_id=model_id, conv_key=conv_key, eval_key=eval_key)
    sentence_evaluator.run_on_data_batch_from_dir(sample_dir, max_workers=10)
