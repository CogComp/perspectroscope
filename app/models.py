from django.db import models
import pickle


class QueryLog(models.Model):
    query_claim = models.TextField()
    query_time = models.DateTimeField()


class FeedbackRecord(models.Model):
    username = models.CharField(max_length=100, default="Anonymous")
    claim = models.TextField()
    perspective = models.TextField()
    relevance_score = models.FloatField()
    stance_score = models.FloatField()
    evidence = models.TextField(default="")
    stance = models.CharField(max_length=3, default="UNK")
    feedback = models.BooleanField() # True = Like, False = Dislike
    comment = models.TextField(default="")


class EvidenceFeedbackRecord(models.Model):
    username = models.CharField(max_length=100, default="Anonymous")
    claim = models.TextField()
    perspective = models.TextField()
    relevance_score = models.FloatField()
    stance_score = models.FloatField()
    evidence = models.TextField(default="")
    stance = models.CharField(max_length=3, default="UNK")
    feedback = models.BooleanField()  # True = Like, False = Dislike
    comment = models.TextField(default="")


class Perspectives(models.Model):
    username = models.CharField(max_length=100, default="Anonymous")
    claim = models.TextField()
    perspective = models.TextField()
    stance = models.CharField(max_length=3, default="UNK")
    comment = models.TextField(default="")


class LRUCache(models.Model):
    claim = models.TextField()
    with_wiki = models.BooleanField(default=False)
    data = models.BinaryField()

    @staticmethod
    def get(claim, with_wiki):
        f = LRUCache.objects.filter(claim=claim, with_wiki=with_wiki)
        print(f)
        if f.exists():
            item = f.first()
            return pickle.loads(item.data)
        else:
            return None

