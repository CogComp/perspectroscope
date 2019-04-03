import logging
import sys
import math
import numpy as np

from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value, os
from model.run_bert_on_perspectrum import BertBaseline

from django.shortcuts import render, redirect
from sklearn.cluster import DBSCAN

from search.query_elasticsearch import get_perspective_from_pool


no_cuda = False if os.environ.get('CUDA_VISIBLE_DEVICES') else True

### loading the BERT solvers
bb_relevance = BertBaseline(task_name="perspectrum_relevance",
                            saved_model="data/model/relevance/perspectrum_relevance_lr2e-05_bs32_epoch-0.pth",
                            no_cuda=no_cuda)
bb_stance = BertBaseline(task_name="perspectrum_stance",
                         saved_model="data/model/stance/perspectrum_stance_lr2e-05_bs16_epoch-4.pth",
                         no_cuda=no_cuda)
bb_equivalence = BertBaseline(task_name="perspectrum_equivalence",
                              saved_model="data/model/equivalence/perspectrum_equivalence_lr3e-05_bs32_epoch-2.pth",
                              no_cuda=no_cuda)

logging.disable(sys.maxsize)  # Python 3

def _normalize(num):
    return math.floor(num * 100) / 100.0

def perspectrum_solver(request, claim_text="", vis_type=""):
    """
    solves a given instances with one of the baselines.
    :param request: the default request argument.
    :param claim_text: the text of the input claim.
    :param vis_type: whether we visualize with the fancy graphical interface or we use a simple visualization.
    :param baseline_name: the solver name (BERT and Lucene).
    :return:
    """
    print(claim_text)
    if claim_text != "":
        claim = claim_text  #

        prob = LpProblem("perspectiveOptimization", LpMaximize)

        # given a claim, extract perspectives
        perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
                                   get_perspective_from_pool(claim, 3)]

        perspective_relevance_score = bb_relevance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspective_stance_score = bb_stance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspectives_sorted = [(p_text, pId, _normalize(luceneScore), _normalize(perspective_relevance_score[i]),
                                _normalize(perspective_stance_score[i])) for i, (p_text, pId, luceneScore) in
                               enumerate(perspective_given_claim)]

        perspectives_sorted = sorted(perspectives_sorted, key=lambda x: -x[3])

        similarity_score = np.zeros((len(perspective_given_claim), len(perspective_given_claim)))
        perspectives_equivalences = []
        for i, (p_text1, _, _) in enumerate(perspective_given_claim):
            list1 = []
            # list2 = []
            for j, (p_text2, _, _) in enumerate(perspective_given_claim):
                # if i != j:
                list1.append((claim + " . " + p_text1, p_text2))
                # list2.append((claim + " . " + p_text2, p_text1))

            predictions1 = bb_equivalence.predict_batch(list1)
            # predictions2 = bb_equivalence.predict_batch(list2)

            for j, (p_text2, _, _) in enumerate(perspective_given_claim):
                if i != j:
                    perspectives_equivalences.append((p_text1, p_text2, predictions1[j], predictions1[j]))
                    similarity_score[i, j] = predictions1[j]
                    similarity_score[j, i] = predictions1[j]

        distance_scores = -similarity_score

        # rescale distance score to [0, 1]
        distance_scores -= np.min(distance_scores)
        distance_scores /= np.max(distance_scores)

        clustering = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_scores)
        print(cluster_labels)
        max_val = max(cluster_labels)
        for i, _ in enumerate(cluster_labels):
            max_val += 1
            if cluster_labels[i] == -1:
                cluster_labels[i] = max_val

        persp_sup = []
        persp_sup_flash = []
        persp_und = []
        persp_und_flash = []

        perspective_clusters = {}
        for i, (p_text, pId, luceneScore, relevance_score, stance_score) in enumerate(perspectives_sorted):
            if relevance_score > 0.0:
                id = cluster_labels[i]
                if id not in perspective_clusters:
                    perspective_clusters[id] = []
                perspective_clusters[id].append((p_text, pId, stance_score))

        for cluster_id in perspective_clusters.keys():
            stance_list = []
            perspectives = []
            persp_flash_tmp = []
            for (p_text, pId, stance_score) in perspective_clusters[cluster_id]:
                stance_list.append(stance_score)
                perspectives.append((pId, p_text))
                persp_flash_tmp.append((p_text, pId, cluster_id + 1, [], stance_score))
                # persp_sup.append((p[0], p[1], 1, [evidences], pScore))

            avg_stance = sum(stance_list) / len(stance_list)
            if avg_stance > 0.0:
                persp_sup.append((perspectives, [avg_stance, 0, 0, 0, 0], []))
                persp_sup_flash.extend(persp_flash_tmp)
            else:
                persp_und.append((perspectives, [avg_stance, 0, 0, 0, 0], []))
                persp_und_flash.extend(persp_flash_tmp)

        # if vis_type == "graphical-viz":
        #     claim_persp_bundled = [(claim, persp_sup, [])]
        # else:
        #     claim_persp_bundled = []

        claim_persp_bundled = [(claim, persp_sup_flash, persp_und_flash)]

        # persp_sup = [
        #     ([(7584, 'It will cause less re-offenders.'), (26958, 'Adequate punishment reduces future offenses.'),
        #       (26959, 'Just punishment will lead to less criminals re-offending. ')], [3, 0, 0, 0, 0],
        #      [367, 368, 2628, 2629, 7862, 6549])]
        # persp_und = [([(7587, 'The onus should not be on punishing the criminal.'),
        #                (26962, 'Punishment should not be the primary focus.'),
        #                (26963, 'Our  main goal should not be punishing the criminal. ')], [0, 0, 0, 3, 0],
        #               [7574, 7872])]

        context = {
            "claim_text": claim_text,
            "vis_type": vis_type,
            "perspectives_sorted": perspectives_sorted,
            "perspectives_equivalences": perspectives_equivalences,
            "claim_persp_bundled": claim_persp_bundled,
            "used_evidences_and_texts": [],  # used_evidences_and_texts,
            # "claim": "",
            # "claim_id": claim_id,
            "persp_sup": persp_sup,
            "persp_und": persp_und,
        }

        print(context)

    else:
        context = {}

    return render(request, "vis_dataset_js_with_search_box.html", context)