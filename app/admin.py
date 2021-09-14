dj
from django.contrib import admin

from .models import QueryLog, FeedbackRecord, Claim, EvidenceFeedbackRecord, Perspectives, LRUCache

admin.site.register(QueryLog)
admin.site.register(FeedbackRecord)
admin.site.register(Claim)
admin.site.register(EvidenceFeedbackRecord)
admin.site.register(Perspectives)
admin.site.register(LRUCache)
