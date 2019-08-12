"""app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.perspectrum_solver),
    path('annotator/', views.perspectrum_annotator),
    path('annotator/<str:claim_text>/<str:withWiki>/', views.perspectrum_annotator),
    path('baseline/', views.perspectrum_solver),
    path('baseline/<str:claim_text>/<str:withWiki>/', views.perspectrum_solver),

    # APIs
    path('load_claim_text/', views.load_claim_text),
    path('load_new_claim_text/', views.load_new_claim_text),
    path('api/submit_query_log/', views.api_submit_query_log),
    path('api/submit_feedback/', views.api_submit_feedback),
    path('api/submit_annotation/', views.api_submit_annotation),
    path('api/retrieve_evidence/', views.api_retrieve_evidence),
    path('api/test_es_connection/', views.api_test_es_connection),
    path(r'^djga/', include('google_analytics.urls')),
]
