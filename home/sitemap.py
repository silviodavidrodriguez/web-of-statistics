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
            'classification',
        ]

    def location(self, item):
        return reverse(item)

    def priority(self, item):

        priorities = {
            'home': 1.0,
            'classification': 0.9,
            'regression': 0.9,
            'anova': 0.8,
            'multivariate': 0.8,
            'probability': 0.8,
            'inference': 0.8,
            'descriptive': 0.8,
            'control': 0.8,
        }

        return priorities.get(item, 0.8)