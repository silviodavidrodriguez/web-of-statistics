from django.contrib import admin
from django.urls import path, include
from django.contrib.sitemaps.views import sitemap
from home.sitemap import StaticSitemap
from django.views.generic import TemplateView

sitemaps = {
    'pages': StaticSitemap,
}

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", include('home.urls')),
    path("", include('descriptive.urls')),
    path("", include('probability.urls')),
    path("", include('inference.urls')),
    path("", include('control.urls')),
    path("", include('anova.urls')),
    path("", include('regression.urls')),
    path("", include('multivariate.urls')),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}),
    path('robots.txt', TemplateView.as_view(
        template_name='home/robots.txt',
        content_type='text/plain'
    )),
]