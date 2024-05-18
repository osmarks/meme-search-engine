import numpy
import xgboost as xgb

import shared

trains, validations = shared.fetch_ratings()

ranker = xgb.XGBRanker(
    tree_method="hist",
    lambdarank_num_pair_per_sample=8,
    objective="rank:ndcg",
    lambdarank_pair_method="topk",
    device="cuda"
)
flat_samples = [ sample for trainss in trains for sample in trainss ]
X = numpy.concatenate([ numpy.stack((meme1, meme2)) for meme1, meme2, rating in flat_samples ])
Y = numpy.concatenate([ numpy.stack((int(rating), int(1 - rating))) for meme1, meme2, rating in flat_samples ])
qid = numpy.concatenate([ numpy.stack((i, i)) for i in range(len(flat_samples)) ])
ranker.fit(X, Y, qid=qid, verbose=True)