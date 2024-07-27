# api/urls.py

from django.urls import path
from .views import GenAIView, ChatWithPdf, ChatWithDB, ChatWithDBCJ
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("genai/", GenAIView.as_view(), name="genai"),
    path("chatpdf/", ChatWithPdf.as_view(), name="chatpdf"),
    path("chatdb/", ChatWithDB.as_view(), name="chatdb"),
    path("chatdbcj/", ChatWithDBCJ.as_view(), name="chatdbcj"),
]
