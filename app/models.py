from django.db import models
import json


class QueryLog(models.Model):
    query_claim = models.TextField()
    query_time = models.DateTimeField()


class FeedbackRecord(models.Model):
    claim = models.TextField()
    perspective = models.TextField()
    relevance_score = models.FloatField()
    stance_score = models.FloatField()
    evidence = models.TextField(default="")
    feedback = models.BooleanField() # True = Like, False = Dislike


class LRUCache(models.Model):
    claim = models.TextField()
    with_wiki = models.BooleanField(default=False)
    data = models.TextField()

    @staticmethod
    def get(claim, with_wiki):
        f = LRUCache.objects.filter(claim=claim, with_wiki=with_wiki)
        if f.exists():
            item = f.first()
            return json.loads(item.data)
        else:
            return None

