from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings
from django.urls import re_path as url

from . import views

urlpatterns = [
    path('videofeed/', views.livefeed, name='videofeed'),
    path(r'^vids/(?P<pose>\s+)/$', views.vid, name='vid'),
    path('sessions/', views.sessions, name='sessions'),
    path('index/', views.index, name='landing'),
    path('home/', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('logout/', views.login, name='logout'),
    path('about/', views.about, name='about'),
    path('blog/', views.blog, name='blog'),
    path('singleBlog/', views.singleBlog, name='singleBlog'),
    path('contact/', views.contact, name='contact'),
    path('schedule/', views.schedule, name='schedule'),
    path('account_activation_sent/', views.account_activation_sent, name='account_activation_sent'),
    path(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.activate, name='activate'),

]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
