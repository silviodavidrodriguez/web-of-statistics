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
    #path('i18n/', include('django.conf.urls.i18n')),
    path("", include('home.urls')),
    path("", include('descriptive.urls')),
    path("", include('probability.urls')),
    path("", include('inference.urls')),
    path("", include('control.urls')),
    path("", include('anova.urls')),
    path("", include('regression.urls')),
    path("", include('multivariate.urls')),
    path("", include('classification.urls')),
    path("", include('multivariate_regression.urls')),
    path("", include('tree_models.urls')),
    path("", include('sensory_analysis.urls')),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}),
    path('robots.txt', TemplateView.as_view(
        template_name='home/robots.txt',
        content_type='text/plain'
    )),
]