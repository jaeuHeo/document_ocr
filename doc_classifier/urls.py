# -*- coding: utf-8 -*-

from django.conf.urls import url
from doc_classifier import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    url(r'^document_form/$',views.document_form),
    url(r'^show/$',views.show_info),
    url(r'^recall/$',views.recall),
    url(r'^correct/$',views.correct),
    url(r'^update/$',views.update_correct),
    url(r'^delete_image/$',views.delete_image),
    url(r'^classify/$',views.clf),
    url(r'^result/$',views.result),
    url(r'^alldata/$',views.alldata),
    url(r'^filter/$',views.filter),
    url(r'^exceldata/$',views.exceldata),
    url(r'^alivecheck$',views.get_alive_check),
]

