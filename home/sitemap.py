from django.contrib.sitemaps import Sitemap
from django.urls import reverse

class StaticSitemap(Sitemap):
    protocol = 'https'
    changefreq = 'weekly'
    priority = 0.8

    def items(self):
        return [
            'home',
            'anova',
            'control',
            'descriptive',
            'inference',
            'multivariate',
            'probability',
            'regression',
        ]

    def location(self, item):
        return reverse(item)

    def priority(self, item):
        if item == 'home':
            return 1.0
        return 0.8