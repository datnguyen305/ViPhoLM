# from .bleu import Bleu
# from .meteor import Meteor
# from .rouge import Rouge
# from .cider import Cider
from rouge_score.rouge_scorer import RougeScorer
import numpy as np

# def compute_scores(gts, gen):
#     metrics = (Bleu(), Meteor(), Rouge(), Cider())
#     all_score = {}
#     all_scores = {}
#     for metric in metrics:
#         score, scores = metric.compute_score(gts, gen)
#         all_score[str(metric)] = score
#         all_scores[str(metric)] = scores

#     return all_score, all_scores

def compute_scores(gts: dict, gen: dict):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1s = []
    rouge_2s = []
    rouge_ls = []
    for reference_summary, candidate_summary in zip(gts.values(), gen.values()):
        scores = scorer.score(reference_summary, candidate_summary)
        rouge_1s.append(scores["rouge1"].fmeasure)
        rouge_2s.append(scores["rouge2"].fmeasure)
        rouge_ls.append(scores["rougeL"].fmeasure)

    return {
        "rouge-1": np.array(rouge_1s).mean().item(),
        "rouge-2": np.array(rouge_2s).mean().item(),
        "rouge-L": np.array(rouge_ls).mean().item()
    }
