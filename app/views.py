import logging
import os
import sys
import math
import numpy as np
import json
import datetime

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cluster import DBSCAN
from model.run_bert_on_perspectrum import BertBaseline
from search.query_elasticsearch import get_perspective_from_pool, get_evidence_from_pool
from search.google_custom_search import CustomSearchClient
from search.news_html_to_text import parse_article
from nltk import sent_tokenize

from app.models import QueryLog, FeedbackRecord

file_names = {
    'evidence': 'data/perspectrum/evidence_pool_v0.2.json',
    'perspective': 'data/perspectrum/perspective_pool_v0.2.json',
    'claim_annotation': 'data/perspectrum/perspectrum_with_answers_v0.2.json'
}

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

bb_evidence = BertBaseline(task_name="perspectrum_evidence",
                              saved_model="data/model/evidence/perspectrum_evidence_epoch-4.pth",
                              no_cuda=no_cuda)

logging.disable(sys.maxsize)  # Python 3

### Load config JSON object
config = json.load(open("config/config.json"))


def load_claim_text(request):
    with open(file_names["claim_annotation"], encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return JsonResponse([c['text'] for c in data], safe=False)


def _keep_two_decimal(num):
    return math.floor(num * 100) / 100.0

def _normalize(float_val, old_range, new_range):
    """
    Normalize float_val from [old_range[0], old_range[1]] to [new_range[0], new_range[1]]
    """
    normalized = (float_val - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[0]
    if normalized > new_range[1]:
            normalized = new_range[1]
    elif normalized < new_range[0]:
        normalized = new_range[0]

    return normalized

def _normalize_relevance_score(bert_logit):
    return _normalize(bert_logit, old_range=[0, 3], new_range=[0, 1])


def _normalize_stance_score(bert_logit):
    return _normalize(bert_logit, old_range=[-4, 4], new_range=[-1, 1])

def _get_perspectives_from_cse(claim_text):

    csc = CustomSearchClient(key=config["custom_search_api_key"], cx=config["custom_search_engine_id"])

    r = csc.query(claim_text)
    urls = [_r["link"] for _r in r]

    sents = []
    sent_url = []

    for url in urls:
        article = parse_article(url)
        paragraphs = [p for p in article.text.splitlines() if p]
        sents += [sent_tokenize(p)[0] for p in paragraphs]
        sent_url += [url for _ in paragraphs]
        # sents += [_s for p in paragraphs for _s in sent_tokenize(p)]

    perspective_relevance_score = bb_relevance.predict_batch([
        (claim_text, sent) for sent in sents
    ])

    perspective_relevance_score = [float(x) for x in perspective_relevance_score]

    perspective_stance_score = bb_stance.predict_batch([
        (claim_text, sent) for sent in sents
    ])

    perspective_stance_score = [float(x) for x in perspective_stance_score]

    results = list(zip(sents, perspective_relevance_score, perspective_stance_score, sent_url))

    return results

def _get_evidence_from_perspectrum(claim, perspective):
    claim_persp = claim + perspective
    lucene_results = get_evidence_from_pool(claim + perspective, 20)

    evidence_score = bb_evidence.predict_batch([(claim_persp, evi) for evi, eid, _ in lucene_results])

    results = [(lucene_results[i][0], score, lucene_results[i][1]) for i, score in enumerate(evidence_score)]
    results = sorted(results, key=lambda r: r[1], reverse=True)

    return results[0][0], "Evidence ID = {}".format(results[0][2])


def _get_evidence_from_link(url, claim, perspective):
    # return "News media and television journalism have been a key feature in the shaping of American collective memory for much of the twentieth century. Indeed, since the United States' colonial era, news media has influenced collective memory and discourse about national development and trauma. In many ways, mainstream journalists have maintained an authoritative voice as the storytellers of the American past. Their documentary style narratives, detailed exposes, and their positions in the present make them prime sources for public memory", url

    print(url, claim, perspective)
    if not url:
        return _get_evidence_from_perspectrum(claim, perspective)

    article = parse_article(url)
    paragraphs = [p for p in article.text.splitlines() if p]
    claim_persp = claim.strip() + " " + perspective.strip()
    all_sent_batch = []

    for p in paragraphs:

        sents = sent_tokenize(p)[1:]
        num_sent = len(sents)
        for i, sent in enumerate(sents):
            three_sent_batch = sent + " "
            if i + 1 < num_sent:
                three_sent_batch += sents[i + 1]
                three_sent_batch += " "
            if i + 2 < num_sent:
                three_sent_batch += sents[i + 2]

            all_sent_batch.append(three_sent_batch)

    evidence_score = bb_evidence.predict_batch([(claim_persp, b) for b in all_sent_batch])
    result = list(zip(all_sent_batch, evidence_score))
    result = sorted(result, key=lambda x: x[1], reverse=True)

    return result[0][0], url


def perspectrum_solver(request, claim_text="", withWiki=""):
    """
    solves a given instances with one of the baselines.
    :param request: the default request argument.
    :param claim_text: the text of the input claim.
    :param vis_type: whether we visualize with the fancy graphical interface or we use a simple visualization.
    :param baseline_name: the solver name (BERT and Lucene).
    :return:
    """
    context = {}

    if claim_text != "":
        claim = claim_text

        # given a claim, extract perspectives
        perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
                                   get_perspective_from_pool(claim, 30)]

        perspective_relevance_score = bb_relevance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspective_stance_score = bb_stance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspectives_sorted = [(p_text, pId, _keep_two_decimal(luceneScore), _keep_two_decimal(perspective_relevance_score[i]),
                                _keep_two_decimal(perspective_stance_score[i])) for i, (p_text, pId, luceneScore) in
                               enumerate(perspective_given_claim)]

        perspectives_sorted = sorted(perspectives_sorted, key=lambda x: x[3] + 0.2 * math.fabs(x[4]), reverse=True)
        perspectives_sorted = [p for p in perspectives_sorted if p[3] > 0.7]

        similarity_score = np.zeros((len(perspective_given_claim), len(perspective_given_claim)))
        perspectives_equivalences = []
        for i, (p_text1, _, _) in enumerate(perspective_given_claim):
            list1 = []
            for j, (p_text2, _, _) in enumerate(perspective_given_claim):
                list1.append((claim + " . " + p_text1, p_text2))

            predictions1 = bb_equivalence.predict_batch(list1)

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
                perspective_clusters[id].append((p_text, pId, stance_score, relevance_score))

        for cluster_id in perspective_clusters.keys():
            stance_list = []
            relevance_list = []
            perspectives = []
            persp_flash_tmp = []
            for (p_text, pId, stance_score, relevance_score) in perspective_clusters[cluster_id]:
                stance_list.append(stance_score)
                relevance_list.append(relevance_score)
                perspectives.append(p_text)
                persp_flash_tmp.append((p_text, pId, cluster_id + 1, [], stance_score))

            avg_stance = _normalize_stance_score(sum(stance_list) / len(stance_list))
            avg_relevance = _normalize_relevance_score(sum(relevance_list) / len(relevance_list))
            if avg_stance > 0.0:
                persp_sup.append((perspectives, [avg_stance, avg_relevance], ["The PERSPECTRUM Dataset", ""]))
                persp_sup_flash.extend(persp_flash_tmp)
            else:
                persp_und.append((perspectives, [avg_stance, avg_relevance], ["The PERSPECTRUM Dataset", ""]))
                persp_und_flash.extend(persp_flash_tmp)

        claim_persp_bundled = [(claim, persp_sup_flash, persp_und_flash)]

        context["claim_text"] =  claim_text
        context["perspectives_sorted"] = perspectives_sorted
        context["perspectives_equivalences"] = perspectives_equivalences
        context["claim_persp_bundled"] = claim_persp_bundled
        context["used_evidences_and_texts"] = []  # used_evidences_and_texts,
        context["persp_sup"] = persp_sup
        context["persp_und"] =  persp_und

        if withWiki == "withWiki":
            web_persps = _get_perspectives_from_cse(claim_text)

            ## Filter results based on a threshold on relevance score
            _REL_SCORE_TH = 1.5
            web_persps = [(_s, _normalize_relevance_score(_rel_score), _normalize_stance_score(_stance_score), url)
                          for _s, _rel_score, _stance_score, url in web_persps if _rel_score > _REL_SCORE_TH]
            web_persps = sorted(web_persps, key=lambda x: x[1] + 0.2*math.fabs(x[2]), reverse=True)

            web_persps = web_persps[:20] # Only keep top 20

            ## Normalize the relevance score
            wiki_persp_sup = []
            wiki_persp_und = []

            for persp, _rel_score, _stance_score, url in web_persps:
                print(_rel_score, _stance_score)
                if _stance_score > 0:
                    persp_sup.append(([persp], [_stance_score, _rel_score], ["Wikipedia", url]))
                else:
                    persp_und.append(([persp], [_stance_score, _rel_score], ["Wikipedia", url]))


            # context["wiki_persp_und"] = wiki_persp_und
            # context["wiki_persp_sup"] = wiki_persp_sup

    return render(request, "vis_dataset_js_with_search_box.html", context)


@csrf_exempt
def api_submit_query_log(request):
    if request.method != 'POST':
        return HttpResponse("submit_query_log api only supports POST method.", status=400)

    query_claim = request.POST.get('claim', '')

    if query_claim:
        QueryLog.objects.create(query_claim=query_claim, query_time=datetime.datetime.now())

    return HttpResponse(status=200)


@csrf_exempt
def api_submit_feedback(request):
    if request.method != 'POST':
        return HttpResponse("api_submit_feedback api only supports POST method.", status=400)

    query_claim = request.POST.get('claim', '')
    perspective = request.POST.get('perspective', '')

    relevance_score = float(request.POST.get('relevance_score', '0.0'))
    stance_score = float(request.POST.get('stance_score', '0.0'))
    feedback = request.POST.get('feedback', '')

    if feedback and query_claim:
        like = True if feedback == 'like' else False

        FeedbackRecord.objects.create(
            claim=query_claim,
            perspective=perspective,
            relevance_score=relevance_score,
            stance_score=stance_score,
            feedback=like,
        )

    return HttpResponse(status=200)

@csrf_exempt
def api_retrieve_evidence(request):
    if request.method != 'POST':
        return HttpResponse("api_submit_feedback api only supports POST method.", status=400)

    claim = request.POST.get('claim', '')
    perspective = request.POST.get('perspective', '')

    link  = request.POST.get("url", None)

    evidence_paragraph, url = _get_evidence_from_link(link, claim, perspective)

    res_data = {
        "evidence_paragraph": evidence_paragraph,
        "url": url
    }
    return JsonResponse(res_data, status=200)