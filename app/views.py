import logging
import os
import sys
import math
import numpy as np
import json
import datetime
import pickle

from django.contrib.auth.models import User
from django.contrib.auth import login, logout
from django.contrib.auth import authenticate
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from sklearn.cluster import DBSCAN
from model.run_bert_on_perspectrum import BertBaseline
from search.query_elasticsearch import get_perspective_from_pool, get_evidence_from_pool, test_connection
from search.google_custom_search import CustomSearchClient
from search.news_html_to_text import parse_article
from nltk import sent_tokenize

from django.db.models import Count
from app.models import QueryLog, FeedbackRecord, LRUCache, Perspectives

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

# logging.disable(sys.maxsize)  # Python 3

### Load config JSON object
config = json.load(open("config/config.json"))

### Load claims
with open(file_names['claim_annotation']) as fin:
    persp_claims = json.load(fin)


def load_claim_text(request):
    with open(file_names["claim_annotation"], encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return JsonResponse([c['text'] for c in data], safe=False)


file1 = "app/static/claims/starts_should_.txt"
file2 = "app/static/claims/starts_should_not_.txt"
cmv_titles = "app/static/claims/cmv_title.txt"
test_claims = "app/static/claims/similar_perspectrum_claims_top20.txt"

from random import shuffle


def load_new_claim_text(request):
    # all_claims = []
    # with open(file1, encoding='utf-8') as data_file:
    #     all_lines = data_file.readlines()
    #     shuffle(all_lines)
    #     all_lines = all_lines[:15]
    #     sentences = [x.strip() for x in all_lines]
    #     all_claims += sentences
    # with open(cmv_titles, encoding='utf-8') as data_file:
    #     all_lines = data_file.readlines()
    #     shuffle(all_lines)
    #     all_lines = all_lines[:20]
    #     sentences = [x.strip() for x in all_lines]
    #     all_claims += sentences
    #
    # shuffle(persp_claims)
    # all_claims += [x['text'] for x in persp_claims][:5]
    # shuffle(all_claims)
    #
    # return JsonResponse(all_claims, safe=False)
    return load_test_claim_text(request)


def load_test_claim_text(request):
    sentences = []
    with open(test_claims, encoding='utf-8') as data_file:
        all_lines = data_file.readlines()
        shuffle(all_lines)
        sentences += [x.strip() for x in all_lines]

    return JsonResponse(sentences, safe=False)


def _keep_two_decimal(num):
    return math.floor(num * 100) / 100.0


def _normalize(float_val, old_range, new_range):
    """
    Normalize float_val from [old_range[0], old_range[1]] to [new_range[0], new_range[1]]
    """
    normalized = (float_val - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[
        0]
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


def solve_given_claim(claim_text, withWiki, num_persp_ir_candidates=30, num_web_persp_candidates=20,
                      run_equivalence=True, relevance_score_th=1.3):
    context = {}

    if claim_text != "":

        claim = claim_text

        _ctx = LRUCache.get(claim, with_wiki=(withWiki == "withWiki"))

        if _ctx is None:

            # given a claim, extract perspectives
            perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
                                       get_perspective_from_pool(claim, num_persp_ir_candidates)]

            perspective_relevance_score = bb_relevance.predict_batch(
                [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

            perspective_stance_score = bb_stance.predict_batch(
                [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

            perspectives_sorted = [(p_text, _normalize_relevance_score(perspective_relevance_score[i]),
                                    _normalize_stance_score(perspective_stance_score[i]), None) for i, (p_text, pId, _)
                                   in
                                   enumerate(perspective_given_claim) if perspective_relevance_score[i] > 1]

            if withWiki == "withWiki":
                web_persps = _get_perspectives_from_cse(claim_text)

                ## Filter results based on a threshold on relevance score
                _REL_SCORE_TH = relevance_score_th
                web_persps = [(_s, _normalize_relevance_score(_rel_score), _normalize_stance_score(_stance_score), url)
                              for _s, _rel_score, _stance_score, url in web_persps if _rel_score > _REL_SCORE_TH]

                ## Filter results that have low stance score
                web_persps = [web_p for web_p in web_persps if abs(web_p[2]) > 0.1]

                web_persps = web_persps[:num_web_persp_candidates]  # Only keep top 20

                perspectives_sorted += web_persps

            perspectives_sorted = list(set(perspectives_sorted))
            perspectives_sorted = sorted(perspectives_sorted, key=lambda x: x[1] + 0.2 * math.fabs(x[2]), reverse=True)

            perspectives_equivalences = []
            persp_sup = []
            persp_sup_flash = []
            persp_und = []
            persp_und_flash = []

            if len(perspectives_sorted) > 0:

                if run_equivalence:
                    similarity_score = np.zeros((len(perspectives_sorted), len(perspectives_sorted)))

                    for i, (p_text1, _, _, _) in enumerate(perspectives_sorted):
                        list1 = []
                        for j, (p_text2, _, _, _) in enumerate(perspectives_sorted):
                            list1.append((claim + " . " + p_text1, p_text2))

                        predictions1 = bb_equivalence.predict_batch(list1)

                        for j, (p_text2, _, _, _) in enumerate(perspectives_sorted):
                            if i != j:
                                perspectives_equivalences.append((p_text1, p_text2, predictions1[j], predictions1[j]))
                                similarity_score[i, j] = predictions1[j]
                                similarity_score[j, i] = predictions1[j]

                    distance_scores = -similarity_score

                    # rescale distance score to [0, 1]
                    distance_scores -= np.min(distance_scores)
                    if np.max(distance_scores) != 0.0:
                        distance_scores /= np.max(distance_scores)

                    clustering = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')
                    cluster_labels = clustering.fit_predict(distance_scores)

                else:
                    cluster_labels = [-1 for _ in range(len(perspectives_sorted))]

                max_val = max(cluster_labels)
                for i, _ in enumerate(cluster_labels):
                    max_val += 1
                    if cluster_labels[i] == -1:
                        cluster_labels[i] = max_val

                perspective_clusters = {}
                for i, (p_text, relevance_score, stance_score, url) in enumerate(perspectives_sorted):
                    id = cluster_labels[i]
                    if id not in perspective_clusters:
                        perspective_clusters[id] = []
                    perspective_clusters[id].append((p_text, stance_score, relevance_score, url))

                for cluster_id in perspective_clusters.keys():
                    stance_list = []
                    relevance_list = []
                    perspectives = []
                    persp_flash_tmp = []

                    (p_text, stance_score, relevance_score, url) = max(perspective_clusters[cluster_id],
                                                                       key=lambda c: 0.2 * c[1] + c[2])

                    stance_list.append(stance_score)
                    relevance_list.append(relevance_score)
                    perspectives.append(p_text)

                    pid = 0
                    for p_text, stance_score, relevance_score, url in perspective_clusters[cluster_id]:
                        persp_flash_tmp.append((p_text, cluster_id * 100 + pid, cluster_id + 1, [], stance_score))
                        pid += 1

                    if url:
                        source = "Web Source"
                        _url = url
                    else:
                        source = "The PERSPECTRUM Dataset"
                        _url = ""
                    if stance_score > 0:
                        persp_sup.append((perspectives, [stance_score, relevance_score], [source, _url]))
                        persp_sup_flash.extend(persp_flash_tmp)
                    else:
                        persp_und.append((perspectives, [stance_score, relevance_score], [source, _url]))
                        persp_und_flash.extend(persp_flash_tmp)

            claim_persp_bundled = [(claim, persp_sup_flash, persp_und_flash)]

            context["claim_text"] = claim_text
            context["perspectives_sorted"] = perspectives_sorted
            context["perspectives_equivalences"] = perspectives_equivalences
            context["claim_persp_bundled"] = claim_persp_bundled
            context["used_evidences_and_texts"] = []  # used_evidences_and_texts,
            context["persp_sup"] = persp_sup
            context["persp_und"] = persp_und

            print(context)

            LRUCache.objects.create(claim=claim,
                                    with_wiki=(withWiki == "withWiki"),
                                    data=pickle.dumps(context))

        else:
            context = _ctx

        return context


def perspectrum_solver(request, withWiki=""):
    """
    solves a given instances with one of the baselines.
    :param request: the default request argument.
    :param claim_text: the text of the input claim.
    :param vis_type: whether we visualize with the fancy graphical interface or we use a simple visualization.
    :param baseline_name: the solver name (BERT and Lucene).
    :return:
    """
    claim_text = request.GET.get('q', "")
    context = solve_given_claim(claim_text, withWiki)
    return render(request, "perspectroscope/perspectrumDemo.html", context)


def perspectrum_annotator(request, withWiki=""):
    claim_text = request.GET.get('q', "")
    result = solve_given_claim(claim_text, withWiki, run_equivalence=False)
    if not result:
        result = {}

    if claim_text.lower() == "animal testing for medical research should be allowed.":
        result["tutorial"] = "true"

    if request.session.get('visited', None):
        result["visited"] = True
    else:
        result["visited"] = False
        request.session['visited'] = 'true'

    result["view_mode"] = False

    return render(request, "perspectrumAnnotator/perspectrumAnnotator.html", result)


def view_annotation(request):
    claim_text = request.GET.get('q', "")

    if claim_text != "":
        persps = Perspectives.objects.filter(claim=claim_text)
        annotations = FeedbackRecord.objects.filter(claim=claim_text)
    else:
        persps = []
        annotations = []

    persp_sup = []
    persp_und = []

    persp_count = {}
    for a in annotations:
        persp = a.perspective
        if persp not in persp_count:
            persp_count[persp] = {
                "stance_score": a.stance_score,
                "like_count": 0,
                "dislike_count": 0,
                "rel_score": a.relevance_score,
                "stance_score": a.stance_score
            }

        if a.feedback:
            persp_count[persp]["like_count"] += 1
        else:
            persp_count[persp]["dislike_count"] += 1

    for persp, vote_count in persp_count.items():

        if vote_count["stance_score"] > 0:
            persp_sup.append([
                [persp],
                [vote_count["rel_score"], vote_count["stance_score"], vote_count["like_count"],
                 vote_count["dislike_count"]]
            ])
        else:
            persp_und.append([
                [persp],
                [vote_count["rel_score"], vote_count["stance_score"], vote_count["like_count"],
                 vote_count["dislike_count"]]
            ])

    for p in persps:
        if p.stance == "SUP":
            persp_sup.append([
                [p.perspective],
                [1, 1, 0, 0]
            ])
        elif p.stance == "UND":
            persp_und.append([
                [p.perspective],
                [1, -1, 0, 0]
            ])

    context = {
        "view_mode": True,
        "claim_text": claim_text,
        "persp_sup": persp_sup,
        "persp_und": persp_und,
    }

    return render(request, "perspectrumAnnotator/perspectrumAnnotator.html", context)


def render_all_annotated_claims(request):
    all_claims = FeedbackRecord.objects.all().values_list('claim').distinct()

    print(all_claims)
    context = {
        "all_claims": all_claims
    }

    return render(request, 'perspectrumAnnotator/view_claims.html', context)


def perspectrum_annotator_about(request):
    context = {}
    return render(request, "perspectrumAnnotator/about.html", context)


def perspectrum_annotator_leaderboard(request):
    all_human_labels = {}
    id_to_user_map = {}
    for record in FeedbackRecord.objects.all():
        # record.username
        id = str(record.claim + record.perspective)
        label = record.stance + str(record.feedback)
        if id in all_human_labels:
            all_human_labels[id].append(label)
            id_to_user_map[id].append(record.username)
        else:
            all_human_labels[id] = [label]
            id_to_user_map[id] = [record.username]

    all_agreements = []
    per_user_agreement = {}
    for id, labels in all_human_labels.items():
        total_count = len(labels)
        if total_count < 3:
            continue
        majority_vote = max(labels, key=labels.count)
        majority_count = len([x for x in labels if x == majority_vote])
        agreement = majority_count / total_count
        for u in id_to_user_map[id]:
            if u not in per_user_agreement:
                per_user_agreement[u] = []
            per_user_agreement[u].append(agreement)

        all_agreements.append(agreement)
    overall_agreement = normalize_numbers(sum(all_agreements) / len(all_agreements))

    user_scores = []
    for u, _ in per_user_agreement.items():
        score = per_user_agreement[u]
        annotation_count = len(score)
        user_agreement = normalize_numbers(sum(score) / len(score))
        overall_score = normalize_numbers(4/3 * (user_agreement - 0.25) * math.sqrt(annotation_count))
        user_scores.append(
            [u, user_agreement, annotation_count, overall_score]
        )
    user_scores = sorted(user_scores, key=lambda x: x[3])
    context = {
        "leader": user_scores,
        "overall": [overall_agreement, len(all_agreements), "-"]
    }
    return render(request, "perspectrumAnnotator/leaderboard.html", context)


def normalize_numbers(number):
    return math.floor(number * 100.0) / 100.0

def perspectrum_annotator_admin(request):
    context = {}
    return render(request, "perspectrumAnnotator/admin.html", context)


def render_login_page(request):
    return render(request, "perspectrumAnnotator/login.html", {})


@csrf_exempt
def api_submit_query_log(request):
    if request.method != 'POST':
        return HttpResponse("submit_query_log api only supports POST method.", status=400)

    query_claim = request.POST.get('claim', '')

    if query_claim:
        if not QueryLog.objects.filter(query_claim=query_claim).exists():
            QueryLog.objects.create(query_claim=query_claim, query_time=datetime.datetime.now())

    return HttpResponse(status=200)


@csrf_protect
def api_submit_annotation(request):
    if request.method != 'POST':
        return HttpResponse("api_submit_feedback api only supports POST method.", status=400)

    if request.user.is_authenticated:
        username = request.user.username
    else:
        username = "Anonymous"

    query_claim = request.POST.get('claim', '')
    perspective = request.POST.get('perspective', '')

    relevance_score = float(request.POST.get('relevance_score', '0.0'))
    stance_score = float(request.POST.get('stance_score', '0.0'))
    stance_label = request.POST.get("stance", 'UNK')
    comment = request.POST.get("comment", "")
    feedback = request.POST.get('feedback', '')

    if feedback and query_claim:
        like = True if feedback == 'like' else False

        FeedbackRecord.objects.create(
            username=username,
            claim=query_claim,
            perspective=perspective,
            relevance_score=relevance_score,
            stance_score=stance_score,
            stance=stance_label,
            feedback=like,
            comment=comment,
        )

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

    link = request.POST.get("url", None)

    evidence_paragraph, url = _get_evidence_from_link(link, claim, perspective)

    res_data = {
        "evidence_paragraph": evidence_paragraph,
        "url": url
    }
    return JsonResponse(res_data, status=200)


@csrf_exempt
def api_test_es_connection(request):
    test_connection()
    return HttpResponse(status=204)


@csrf_protect
def api_auth_login(request):
    if request.method != 'POST':
        return HttpResponse("api_auth_login api only supports POST method.", status=400)

    username = request.POST.get('username')
    password = request.POST.get('password')

    user = authenticate(username=username, password=password)

    if user is not None:
        login(request=request, user=user)
        return HttpResponse("Login Success!", status=200)
    else:
        return HttpResponse("Authentication Failed.", status=401)


@csrf_protect
def api_auth_signup(request):
    if request.method != 'POST':
        return HttpResponse("api_auth_login api only supports POST method.", status=400)

    username = request.POST.get('username')
    password = request.POST.get('password')

    try:
        user = User.objects.get(username=username, password=password)
    except User.DoesNotExist:
        user = User.objects.create_user(username=username, password=pasword)
        return HttpResponse("Sign up success", status=200)

    return HttpResponse("User Already Exists.", status=401)


@csrf_protect
def api_auth_logout(request):
    logout(request)
    return perspectrum_annotator(request)


@csrf_protect
def api_submit_evidence_feedback(request):
    if request.method != 'POST':
        return HttpResponse("api_submit_evidence_feedback api only supports POST method.", status=400)

    if request.user.is_authenticated:
        username = request.user.username
    else:
        username = "Anonymous"

    query_claim = request.POST.get('claim', '')
    perspective = request.POST.get('perspective', '')
    evidence = request.POST.get('evidence', '')

    relevance_score = float(request.POST.get('relevance_score', '0.0'))
    stance_score = float(request.POST.get('stance_score', '0.0'))
    stance_label = request.POST.get("stance_label", 'UNK')
    comment = request.POST.get("comment", "")
    feedback = request.POST.get('feedback', '')

    if feedback and query_claim:
        like = True if feedback == 'like' else False

        FeedbackRecord.objects.create(
            username=username,
            claim=query_claim,
            perspective=perspective,
            relevance_score=relevance_score,
            stance_score=stance_score,
            stance=stance_label,
            evidence=evidence,
            feedback=like,
            comment=comment,
        )

    return HttpResponse(status=200)


@csrf_protect
def api_submit_new_perspective(request):
    if request.method != 'POST':
        return HttpResponse("api_submit_evidence_feedback api only supports POST method.", status=400)

    if request.user.is_authenticated:
        username = request.user.username
    else:
        username = "Anonymous"

    query_claim = request.POST.get('claim', '')
    perspective = request.POST.get('perspective', '')
    stance_label = request.POST.get("stance_label", 'UNK')
    comment = request.POST.get("comment", "")

    Perspectives.objects.create(
        username=username,
        claim=query_claim,
        perspective=perspective,
        stance=stance_label,
        comment=comment
    )

    return HttpResponse(status=200)
