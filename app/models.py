from django.db import models


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
    data = models.TextField()

    @staticmethod
    def get(claim):
        if LRUCache.objects.exists(claim=claim):
            return LRUCache.objects.get(claim=claim)
        else:
            return None

