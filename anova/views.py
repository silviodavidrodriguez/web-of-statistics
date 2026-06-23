from django.shortcuts import render
import pandas as pd
from scipy.stats import kruskal, shapiro, levene, friedmanchisquare, ttest_ind
import scikit_posthocs as sp
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multitest
from scipy.stats import t, f
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io
import base64
import plotly.express as px
from scipy.stats import chi2
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def anova(request):
    tab = request.GET.get('tab', 'one_way_anova')
    subtab = request.GET.get('subtab', None)

    context = {
        'segment': 'anova',
        'active_tab': tab,
        'active_subtab': subtab,
        'data_dca_anova': request.session.get('data_dca_anova', ""),
        'anova_results_dca_anova': request.session.get('anova_results_dca_anova', None),
        'alpha_value_dca_anova': request.session.get('alpha_value_dca_anova', None),
        'post_hoc_test_type_dca_anova': request.session.get('post_hoc_test_type_dca_anova', None),
        'use_first_row_as_header_dca_anova': 'checked' if request.session.get('use_first_row_as_header_dca_anova', False) else '',
        'graph_dca_anova': request.session.get('graph_dca_anova', None),
        'bar_plot_dca_anova': request.session.get('bar_plot_dca_anova', None),
        'residuals_plot_dca_anova': request.session.get('residuals_plot_dca_anova', None),
        'qq_plot_dca_anova': request.session.get('qq_plot_dca_anova', None),
        'tukey_results_dca_anova': request.session.get('tukey_results_dca_anova', None),
        'bonferroni_results_dca_anova': request.session.get('bonferroni_results_dca_anova', None),
        'scheffe_results_dca_anova': request.session.get('scheffe_results_dca_anova', None),
        'dunnett_results_dca_anova': request.session.get('dunnett_results_dca_anova', None),
        'duncan_results_dca_anova': request.session.get('duncan_results_dca_anova', None),
        'holm_results_dca_anova': request.session.get('holm_results_dca_anova', None),
        'fisher_lsd_results_dca_anova': request.session.get('fisher_lsd_results_dca_anova', None),
        'shapiro_results_dca_anova': request.session.get('shapiro_results_dca_anova', None),
        'levene_results_dca_anova': request.session.get('levene_results_dca_anova', None),
        'explanation_dca_anova': request.session.get('explanation_dca_anova', None),
        'warning_1_dca_anova': request.session.get('warning_1_dca_anova', None),
        'warning_2_dca_anova': request.session.get('warning_2_dca_anova', None),
        'warning_3_dca_anova': request.session.get('warning_3_dca_anova', None),
        'warning_4_dca_anova': request.session.get('warning_4_dca_anova', None),
        'data_dba_anova': request.session.get('data_dba_anova', ""),
        'anova_results_dba_anova': request.session.get('anova_results_dba_anova', None),
        'alpha_value_dba_anova': request.session.get('alpha_value_dba_anova', None),
        'post_hoc_test_type_dba_anova': request.session.get('post_hoc_test_type_dba_anova', None),
        'use_first_row_as_header_dba_anova': 'checked' if request.session.get('use_first_row_as_header_dba_anova', False) else '',
        'graph_dba_anova': request.session.get('graph_dba_anova', None),
        'bar_plot_dba_anova': request.session.get('bar_plot_dba_anova', None),
        'residuals_plot_dba_anova': request.session.get('residuals_plot_dba_anova', None),
        'qq_plot_dba_anova': request.session.get('qq_plot_dba_anova', None),
        'tukey_results_dba_anova': request.session.get('tukey_results_dba_anova', None),
        'bonferroni_results_dba_anova': request.session.get('bonferroni_results_dba_anova', None),
        'scheffe_results_dba_anova': request.session.get('scheffe_results_dba_anova', None),
        'dunnett_results_dba_anova': request.session.get('dunnett_results_dba_anova', None),
        'duncan_results_dba_anova': request.session.get('duncan_results_dba_anova', None),
        'holm_results_dba_anova': request.session.get('holm_results_dba_anova', None),
        'fisher_lsd_results_dba_anova': request.session.get('fisher_lsd_results_dba_anova', None),
        'shapiro_results_dba_anova': request.session.get('shapiro_results_dba_anova', None),
        'levene_results_dba_anova': request.session.get('levene_results_dba_anova', None),
        'explanation_dba_anova': request.session.get('explanation_dba_anova', None),
        'warning_1_dba_anova': request.session.get('warning_1_dba_anova', None),
        'warning_2_dba_anova': request.session.get('warning_2_dba_anova', None),
        'warning_3_dba_anova': request.session.get('warning_3_dba_anova', None),
        'warning_4_dba_anova': request.session.get('warning_4_dba_anova', None),
        'data_two_way_anova': request.session.get('data_two_way_anova', ""),
        'anova_results_two_way_anova': request.session.get('anova_results_two_way_anova', None),
        'alpha_value_two_way_anova': request.session.get('alpha_value_two_way_anova', None),
        'post_hoc_test_type_two_way_anova': request.session.get('post_hoc_test_type_two_way_anova', None),
        'use_first_row_as_header_two_way_anova': 'checked' if request.session.get('use_first_row_as_header_two_way_anova', False) else '',
        'graph_two_way_anova': request.session.get('graph_two_way_anova', None),
        'bar_plot_two_way_anova': request.session.get('bar_plot_two_way_anova', None),
        'residuals_plot_two_way_anova': request.session.get('residuals_plot_two_way_anova', None),
        'qq_plot_two_way_anova': request.session.get('qq_plot_two_way_anova', None),
        'interaction_plot_two_way_anova': request.session.get('interaction_plot_two_way_anova', None),
        'tukey_results_two_way_anova': request.session.get('tukey_results_two_way_anova', None),
        'bonferroni_results_two_way_anova': request.session.get('bonferroni_results_two_way_anova', None),
        'scheffe_results_two_way_anova': request.session.get('scheffe_results_two_way_anova', None),
        'holm_results_two_way_anova': request.session.get('holm_results_two_way_anova', None),
        'fisher_lsd_results_two_way_anova': request.session.get('fisher_lsd_results_two_way_anova', None),
        'shapiro_results_two_way_anova': request.session.get('shapiro_results_two_way_anova', None),
        'levene_results_two_way_anova': request.session.get('levene_results_two_way_anova', None),
        'explanation_two_way_anova': request.session.get('explanation_two_way_anova', None),
        'warning_1_two_way_anova': request.session.get('warning_1_two_way_anova', None),
        'warning_3_two_way_anova': request.session.get('warning_3_two_way_anova', None),
        'warning_4_two_way_anova': request.session.get('warning_4_two_way_anova', None),
        'data_kruskal_wallis': request.session.get('data_kruskal_wallis', ""),
        'results_kruskal_wallis': request.session.get('results_kruskal_wallis', None),
        'alpha_value_kruskal_wallis': request.session.get('alpha_value_kruskal_wallis', None),
        'post_hoc_test_type_kruskal_wallis': request.session.get('post_hoc_test_type_kruskal_wallis', None),
        'use_first_row_as_header_kruskal_wallis': 'checked' if request.session.get('use_first_row_as_header_kruskal_wallis', False) else '',
        'graph_kruskal_wallis': request.session.get('graph_kruskal_wallis', None),
        'results_dunn_kruskal_wallis': request.session.get('results_dunn_kruskal_wallis', None),
        'explanation_kruskal_wallis': request.session.get('explanation_kruskal_wallis', None),
        'warning_1_kruskal_wallis': request.session.get('warning_1_kruskal_wallis', None),
        'warning_2_kruskal_wallis': request.session.get('warning_2_kruskal_wallis', None),
        'data_dba_friedman': request.session.get('data_dba_friedman', ""),
        'results_dba_friedman': request.session.get('results_dba_friedman', None),
        'alpha_value_dba_friedman': request.session.get('alpha_value_dba_friedman', None),
        'post_hoc_test_type_dba_friedman': request.session.get('post_hoc_test_type_dba_friedman', None),
        'use_first_row_as_header_dba_friedman': 'checked' if request.session.get('use_first_row_as_header_dba_friedman', False) else '',
        'results_nemenyi_dba_friedman': request.session.get('results_nemenyi_dba_friedman', None),
        'explanation_dba_friedman': request.session.get('explanation_dba_friedman', None),
        'warning_1_dba_friedman': request.session.get('warning_1_dba_friedman', None),
        'warning_2_dba_friedman': request.session.get('warning_2_dba_friedman', None),
        'data_manova': request.session.get('data_manova', ""),
        'manova_results': request.session.get('manova_results', None),
        'univariate_results_manova': request.session.get('univariate_results_manova', None),
        'alpha_value_manova': request.session.get('alpha_value_manova', None),
        'use_first_row_as_header_manova': 'checked' if request.session.get('use_first_row_as_header_manova', False) else '',
        'warning_1_manova': request.session.get('warning_1_manova', None),
        'error_manova': request.session.get('error_manova', None),
        'model_info_manova': request.session.get('model_info_manova', None),
        'boxplots_manova': request.session.get('boxplots_manova', None),
        'boxm_results': request.session.get('boxm_results', None),
        'mardia_results': request.session.get('mardia_results', None),
        'posthoc_manova': request.session.get('posthoc_manova', None),
        'cda_results': request.session.get('cda_results', None),
        'cda_scores_plot': request.session.get('cda_scores_plot', None),
        'cda_centroids': request.session.get('cda_centroids', None),
        }
    
    if request.method == "POST" and request.POST.get("clear_dca_anova") == "true":
        if 'data_dca_anova' in request.session:
            del request.session['data_dca_anova']
        if 'anova_results_dca_anova' in request.session:
            del request.session['anova_results_dca_anova']
        if 'tukey_results_dca_anova' in request.session:
            del request.session['tukey_results_dca_anova']
        if 'bonferroni_results_dca_anova' in request.session:
            del request.session['bonferroni_results_dca_anova']
        if 'scheffe_results_dca_anova' in request.session:
            del request.session['scheffe_results_dca_anova']
        if 'dunnett_results_dca_anova' in request.session:
            del request.session['dunnett_results_dca_anova']
        if 'duncan_results_dca_anova' in request.session:
            del request.session['duncan_results_dca_anova']
        if 'holm_results_dca_anova' in request.session:
            del request.session['holm_results_dca_anova']
        if 'fisher_lsd_results_dca_anova' in request.session:
            del request.session['fisher_lsd_results_dca_anova']
        if 'shapiro_results_dca_anova' in request.session:
            del request.session['shapiro_results_dca_anova']
        if 'levene_results_dca_anova' in request.session:
            del request.session['levene_results_dca_anova']
        if 'graph_dca_anova' in request.session:
            del request.session['graph_dca_anova']
        if 'bar_plot_dca_anova' in request.session:
            del request.session['bar_plot_dca_anova']
        if 'residuals_plot_dca_anova' in request.session:
            del request.session['residuals_plot_dca_anova']
        if 'qq_plot_dca_anova' in request.session:
            del request.session['qq_plot_dca_anova']
        if 'use_first_row_as_header_dca_anova' in request.session:
            request.session.pop('use_first_row_as_header_dca_anova', False)
        if 'alpha_value_dca_anova' in request.session:
            request.session.pop('alpha_value_dca_anova', False)
        if 'post_hoc_test_type_dca_anova' in request.session:
            del request.session['post_hoc_test_type_dca_anova']
        if 'explanation_dca_anova' in request.session:
            del request.session['explanation_dca_anova']
        if 'warning_1_dca_anova' in request.session:
            del request.session['warning_1_dca_anova']
        if 'warning_2_dca_anova' in request.session:
            del request.session['warning_2_dca_anova']
        if 'warning_3_dca_anova' in request.session:
            del request.session['warning_3_dca_anova']
        if 'warning_4_dca_anova' in request.session:
            del request.session['warning_4_dca_anova']
        context['data_dca_anova'] = ""
        context['anova_results_dca_anova'] = None
        context['tukey_results_dca_anova'] = None
        context['bonferroni_results_dca_anova'] = None
        context['scheffe_results_dca_anova'] = None
        context['dunnett_results_dca_anova'] = None
        context['duncan_results_dca_anova'] = None
        context['holm_results_dca_anova'] = None
        context['fisher_lsd_results_dca_anova'] = None
        context['shapiro_results_dca_anova'] = None
        context['levene_results_dca_anova'] = None
        context['graph_dca_anova'] = None
        context['bar_plot_dca_anova'] = None
        context['qq_plot_dca_anova'] = None
        context['residuals_plot_dca_anova'] = None
        context['use_first_row_as_header_dca_anova'] = False
        context['alpha_value_dca_anova'] = False
        context['post_hoc_test_type_dca_anova'] = None
        context['explanation_dca_anova'] = False
        context['warning_1_dca_anova'] = False
        context['warning_2_dca_anova'] = False
        context['warning_3_dca_anova'] = False
        context['warning_4_dca_anova'] = False
        return render(request, "anova/anova.html", context)
    
    if request.method == "POST" and request.POST.get("clear_dba_anova") == "true":
        if 'data_dba_anova' in request.session:
            del request.session['data_dba_anova']
        if 'anova_results_dba_anova' in request.session:
            del request.session['anova_results_dba_anova']
        if 'tukey_results_dba_anova' in request.session:
            del request.session['tukey_results_dba_anova']
        if 'bonferroni_results_dba_anova' in request.session:
            del request.session['bonferroni_results_dba_anova']
        if 'scheffe_results_dba_anova' in request.session:
            del request.session['scheffe_results_dba_anova']
        if 'dunnett_results_dba_anova' in request.session:
            del request.session['dunnett_results_dba_anova']
        if 'duncan_results_dba_anova' in request.session:
            del request.session['duncan_results_dba_anova']
        if 'holm_results_dba_anova' in request.session:
            del request.session['holm_results_dba_anova']
        if 'fisher_lsd_results_dba_anova' in request.session:
            del request.session['fisher_lsd_results_dba_anova']
        if 'shapiro_results_dba_anova' in request.session:
            del request.session['shapiro_results_dba_anova']
        if 'levene_results_dba_anova' in request.session:
            del request.session['levene_results_dba_anova']
        if 'graph_dba_anova' in request.session:
            del request.session['graph_dba_anova']
        if 'bar_plot_dba_anova' in request.session:
            del request.session['bar_plot_dba_anova']
        if 'residuals_plot_dba_anova' in request.session:
            del request.session['residuals_plot_dba_anova']
        if 'qq_plot_dba_anova' in request.session:
            del request.session['qq_plot_dba_anova']
        if 'use_first_row_as_header_dba_anova' in request.session:
            request.session.pop('use_first_row_as_header_dba_anova', False)
        if 'alpha_value_dba_anova' in request.session:
            request.session.pop('alpha_value_dba_anova', False)
        if 'post_hoc_test_type_dba_anova' in request.session:
            del request.session['post_hoc_test_type_dba_anova']
        if 'explanation_dba_anova' in request.session:
            del request.session['explanation_dba_anova']
        if 'warning_1_dba_anova' in request.session:
            del request.session['warning_1_dba_anova']
        if 'warning_2_dba_anova' in request.session:
            del request.session['warning_2_dba_anova']
        if 'warning_3_dba_anova' in request.session:
            del request.session['warning_3_dba_anova']
        if 'warning_4_dba_anova' in request.session:
            del request.session['warning_4_dba_anova']
        context['data_dba_anova'] = ""
        context['anova_results_dba_anova'] = None
        context['tukey_results_dba_anova'] = None
        context['bonferroni_results_dba_anova'] = None
        context['scheffe_results_dba_anova'] = None
        context['dunnett_results_dba_anova'] = None
        context['duncan_results_dba_anova'] = None
        context['holm_results_dba_anova'] = None
        context['fisher_lsd_results_dba_anova'] = None
        context['shapiro_results_dba_anova'] = None
        context['levene_results_dba_anova'] = None
        context['graph_dba_anova'] = None
        context['bar_plot_dba_anova'] = None
        context['qq_plot_dba_anova'] = None
        context['residuals_plot_dba_anova'] = None
        context['use_first_row_as_header_dba_anova'] = False
        context['alpha_value_dba_anova'] = False
        context['post_hoc_test_type_dba_anova'] = None
        context['explanation_dba_anova'] = False
        context['warning_1_dba_anova'] = False
        context['warning_2_dba_anova'] = False
        context['warning_3_dba_anova'] = False
        context['warning_4_dba_anova'] = False
        return render(request, "anova/anova.html", context)
    
    if request.method == "POST" and request.POST.get("clear_two_way_anova") == "true":
        if 'data_two_way_anova' in request.session:
            del request.session['data_two_way_anova']
        if 'anova_results_two_way_anova' in request.session:
            del request.session['anova_results_two_way_anova']
        if 'tukey_results_two_way_anova' in request.session:
            del request.session['tukey_results_two_way_anova']
        if 'bonferroni_results_two_way_anova' in request.session:
            del request.session['bonferroni_results_two_way_anova']
        if 'scheffe_results_two_way_anova' in request.session:
            del request.session['scheffe_results_two_way_anova']
        if 'holm_results_two_way_anova' in request.session:
            del request.session['holm_results_two_way_anova']
        if 'fisher_lsd_results_two_way_anova' in request.session:
            del request.session['fisher_lsd_results_two_way_anova']
        if 'shapiro_results_two_way_anova' in request.session:
            del request.session['shapiro_results_two_way_anova']
        if 'levene_results_two_way_anova' in request.session:
            del request.session['levene_results_two_way_anova']
        if 'graph_two_way_anova' in request.session:
            del request.session['graph_two_way_anova']
        if 'bar_plot_two_way_anova' in request.session:
            del request.session['bar_plot_two_way_anova']
        if 'residuals_plot_two_way_anova' in request.session:
            del request.session['residuals_plot_two_way_anova']
        if 'qq_plot_two_way_anova' in request.session:
            del request.session['qq_plot_two_way_anova']
        if 'interaction_plot_two_way_anova' in request.session:
            del request.session['interaction_plot_two_way_anova']
        if 'use_first_row_as_header_two_way_anova' in request.session:
            request.session.pop('use_first_row_as_header_two_way_anova', False)
        if 'alpha_value_two_way_anova' in request.session:
            request.session.pop('alpha_value_two_way_anova', False)
        if 'post_hoc_test_type_two_way_anova' in request.session:
            del request.session['post_hoc_test_type_two_way_anova']
        if 'explanation_two_way_anova' in request.session:
            del request.session['explanation_two_way_anova']
        if 'warning_1_two_way_anova' in request.session:
            del request.session['warning_1_two_way_anova']
        if 'warning_3_two_way_anova' in request.session:
            del request.session['warning_3_two_way_anova']
        if 'warning_4_two_way_anova' in request.session:
            del request.session['warning_4_two_way_anova']
        context['data_two_way_anova'] = ""
        context['anova_results_two_way_anova'] = None
        context['tukey_results_two_way_anova'] = None
        context['bonferroni_results_two_way_anova'] = None
        context['scheffe_results_two_way_anova'] = None
        context['holm_results_two_way_anova'] = None
        context['fisher_lsd_results_two_way_anova'] = None
        context['shapiro_results_two_way_anova'] = None
        context['levene_results_two_way_anova'] = None
        context['graph_two_way_anova'] = None
        context['bar_plot_two_way_anova'] = None
        context['qq_plot_two_way_anova'] = None
        context['interaction_plot_two_way_anova'] = None
        context['residuals_plot_two_way_anova'] = None
        context['use_first_row_as_header_two_way_anova'] = False
        context['alpha_value_two_way_anova'] = False
        context['post_hoc_test_type_two_way_anova'] = None
        context['explanation_two_way_anova'] = False
        context['warning_1_two_way_anova'] = False
        context['warning_3_two_way_anova'] = False
        context['warning_4_two_way_anova'] = False
        return render(request, "anova/anova.html", context)
    
    if request.method == "POST" and request.POST.get("clear_manova") == "true":
        keys = [
            'data_manova',
            'manova_results',
            'univariate_results_manova',
            'alpha_value_manova',
            'use_first_row_as_header_manova',
            'warning_1_manova',
            'error_manova',
            'model_info_manova',
            'boxplots_manova',
            'boxm_results',
            'mardia_results',
            'posthoc_manova',
            'cda_results',
            'cda_scores_plot',
            'cda_centroids',
        ]

        for key in keys:
            request.session.pop(key, None)

        context['data_manova'] = ""
        context['manova_results'] = None
        context['univariate_results_manova'] = None
        context['alpha_value_manova'] = None
        context['use_first_row_as_header_manova'] = ""
        context['warning_1_manova'] = None
        context['error_manova'] = None
        context['model_info_manova'] = None
        context['boxplots_manova'] = None
        context['boxm_results'] = None
        context['mardia_results'] = None
        context['posthoc_manova'] = None
        context['cda_results'] = None
        context['cda_scores_plot'] = None
        context['cda_centroids'] = None

        return render(request, "anova/anova.html", context)
    
    if request.method == "POST" and request.POST.get("clear_kruskal_wallis") == "true":
        if 'data_kruskal_wallis' in request.session:
            del request.session['data_kruskal_wallis']
        if 'results_kruskal_wallis' in request.session:
            del request.session['results_kruskal_wallis']
        if 'results_dunn_kruskal_wallis' in request.session:
            del request.session['results_dunn_kruskal_wallis']
        if 'graph_kruskal_wallis' in request.session:
            del request.session['graph_kruskal_wallis']
        if 'use_first_row_as_header_kruskal_wallis' in request.session:
            request.session.pop('use_first_row_as_header_kruskal_wallis', False)
        if 'alpha_value_kruskal_wallis' in request.session:
            request.session.pop('alpha_value_kruskal_wallis', False)
        if 'post_hoc_test_type_kruskal_wallis' in request.session:
            del request.session['post_hoc_test_type_kruskal_wallis']
        if 'explanation_kruskal_wallis' in request.session:
            del request.session['explanation_kruskal_wallis']
        if 'warning_1_kruskal_wallis' in request.session:
            del request.session['warning_1_kruskal_wallis']
        if 'warning_2_kruskal_wallis' in request.session:
            del request.session['warning_2_kruskal_wallis']
        context['data_kruskal_wallis'] = ""
        context['results_kruskal_wallis'] = None
        context['results_dunn_kruskal_wallis'] = None
        context['graph_kruskal_wallis'] = None
        context['use_first_row_as_header_kruskal_wallis'] = False
        context['alpha_value_kruskal_wallis'] = False
        context['post_hoc_test_type_kruskal_wallis'] = None
        context['explanation_kruskal_wallis'] = False
        context['warning_1_kruskal_wallis'] = False
        context['warning_2_kruskal_wallis'] = False
        return render(request, "anova/anova.html", context)
    
    if request.method == "POST" and request.POST.get("clear_dba_friedman") == "true":
        if 'data_dba_friedman' in request.session:
            del request.session['data_dba_friedman']
        if 'results_dba_friedman' in request.session:
            del request.session['results_dba_friedman']
        if 'results_nemenyi_dba_friedman' in request.session:
            del request.session['results_nemenyi_dba_friedman']
        if 'use_first_row_as_header_dba_friedman' in request.session:
            request.session.pop('use_first_row_as_header_dba_friedman', False)
        if 'alpha_value_dba_friedman' in request.session:
            request.session.pop('alpha_value_dba_friedman', False)
        if 'post_hoc_test_type_dba_friedman' in request.session:
            del request.session['post_hoc_test_type_dba_friedman']
        if 'explanation_dba_friedman' in request.session:
            del request.session['explanation_dba_friedman']
        if 'warning_1_dba_friedman' in request.session:
            del request.session['warning_1_dba_friedman']
        if 'warning_2_dba_friedman' in request.session:
            del request.session['warning_2_dba_friedman']
        context['data_dba_friedman'] = ""
        context['results_dba_friedman'] = None
        context['results_nemenyi_dba_friedman'] = None
        context['use_first_row_as_header_dba_friedman'] = False
        context['alpha_value_dba_friedman'] = False
        context['post_hoc_test_type_dba_friedman'] = None
        context['explanation_dba_friedman'] = False
        context['warning_1_dba_friedman'] = False
        context['warning_2_dba_friedman'] = False
        return render(request, "anova/anova.html", context)

###########################################################################################################
    if request.method == "POST" and tab == "one_way_anova" and subtab == "dca_anova":
        data_dca_anova = request.POST.get('data_dca_anova')
        use_first_row_as_header_dca_anova = request.POST.get('use_first_row_as_header_dca_anova') == 'on'
        alpha_value_dca_anova = request.POST.get('alpha_value_dca_anova')
        post_hoc_test_type_dca_anova = request.POST.get('post_hoc_test_type_dca_anova')

        context['use_first_row_as_header_dca_anova'] = 'checked' if use_first_row_as_header_dca_anova else ''
        request.session['use_first_row_as_header_dca_anova'] = use_first_row_as_header_dca_anova
        context['data_dca_anova'] = data_dca_anova
        request.session['data_dca_anova'] = data_dca_anova
        context['post_hoc_test_type_dca_anova'] = post_hoc_test_type_dca_anova
        request.session['post_hoc_test_type_dca_anova'] = post_hoc_test_type_dca_anova

        if alpha_value_dca_anova:
            try:
                alpha_value_dca_anova = float(alpha_value_dca_anova)
                context['alpha_value_dca_anova'] = alpha_value_dca_anova
                request.session['alpha_value_dca_anova'] = alpha_value_dca_anova
            except ValueError:
                alpha_value_dca_anova = None
        
        if not alpha_value_dca_anova:
            alpha_value_dca_anova = 95
            context['alpha_value_dca_anova'] = alpha_value_dca_anova
            request.session['alpha_value_dca_anova'] = alpha_value_dca_anova
            warning_1_dca_anova = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_dca_anova'] = warning_1_dca_anova
            request.session['warning_1_dca_anova'] = warning_1_dca_anova

        if not data_dca_anova.strip():
            context['error_dca_anova'] = "Please enter data before calculating."
            context['anova_results_dca_anova'] = None
            context['tukey_results_dca_anova'] = None
            context['bonferroni_results_dca_anova'] = None
            context['scheffe_results_dca_anova'] = None
            context['dunnett_results_dca_anova'] = None
            context['duncan_results_dca_anova'] = None
            context['holm_results_dca_anova'] = None
            context['fisher_lsd_results_dca_anova'] = None
            context['shapiro_results_dca_anova'] = None
            context['levene_results_dca_anova'] = None
            context['graph_dca_anova'] = None
            context['bar_plot_dca_anova'] = None
            context['qq_plot_dca_anova'] = None
            context['residuals_plot_dca_anova'] = None
            context['explanation_dca_anova'] = None
            context['warning_1_dca_anova'] = None
            context['warning_2_dca_anova'] = None
            context['warning_3_dca_anova'] = None
            context['warning_4_dca_anova'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_dca_anova = data_dca_anova.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_dca_anova.split('\n')]

            if use_first_row_as_header_dca_anova:
                headers_dca_anova = rows[0]
                rows = rows[1:]
            else:
                headers_dca_anova = ["Factor", "Response"]

            factors, responses = zip(*rows)
            factors = list(factors)
            responses = [float(r) for r in responses]

            df = pd.DataFrame({"Factor": factors, "Response": responses})

            if df.isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")

            if df["Response"].dtype not in [float, int]:
                raise ValueError("Response column contains non-numeric values.")

            groups = [df[df["Factor"] == level]["Response"] for level in df["Factor"].unique()]

            model = sm.OLS.from_formula("Response ~ C(Factor)", data=df).fit()
            anova_table = anova_lm(model, typ=2)
            anova_df = pd.DataFrame(anova_table)
            total_sum_of_squares = anova_table["sum_sq"].sum()
            total_degrees_of_freedom = anova_table["df"].sum()

            reject_null_dca_anova = "Yes" if anova_table.loc["C(Factor)", "PR(>F)"] <= ((100 - alpha_value_dca_anova) / 100) else "No"

            anova_results_dca_anova = [
                {
                    "Source": row.name if row.name != "C(Factor)" else headers_dca_anova[0],
                    "Sum_of_Squares": "{:.5g}".format(row["sum_sq"]) if pd.notna(row["sum_sq"]) else "",
                    "Degrees_of_Freedom": "{:.5g}".format(row["df"]) if pd.notna(row["df"]) else "",
                    "MS": "{:.5g}".format(row["sum_sq"] / row["df"]) if pd.notna(row["sum_sq"]) and pd.notna(row["df"]) else "",
                    "F_statistic": "{:.5g}".format(row["F"]) if pd.notna(row["F"]) else "",
                    "P_value": "{:.5g}".format(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else "",
                    "Reject_Null": reject_null_dca_anova if row.name == "C(Factor)" else "",
                }
                for _, row in anova_df.iterrows()
            ]

            anova_results_dca_anova.append({
                "Source": "Total",
                "Sum_of_Squares": "{:.5g}".format(total_sum_of_squares),
                "Degrees_of_Freedom": "{:.5g}".format(total_degrees_of_freedom),
                "MS": "",
                "F_statistic": "",
                "P_value": "",
                "Reject_Null": "",
            })

            context['anova_results_dca_anova'] = anova_results_dca_anova
            request.session['anova_results_dca_anova'] = anova_results_dca_anova
            explanation_dca_anova = "Null Hypothesis (H₀): Group means are equal - Alternative Hypothesis (H₁): At least one group mean differs."
            context['explanation_dca_anova'] = explanation_dca_anova
            request.session['explanation_dca_anova'] = explanation_dca_anova

            if reject_null_dca_anova == "No":
                warning_2_dca_anova = "p-value greater than alpha post-hoc test omitted."
                context['warning_2_dca_anova'] = warning_2_dca_anova
                request.session['warning_2_dca_anova'] = warning_2_dca_anova

            if post_hoc_test_type_dca_anova == "tukey" and reject_null_dca_anova == "Yes":
                try:
                    tukey = pairwise_tukeyhsd(endog=df['Response'], groups=df['Factor'], alpha=((100 - alpha_value_dca_anova) / 100))
                    
                    tukey_results_dca_anova = []
                    for i in range(len(tukey.summary().data[1:])):
                        row = tukey.summary().data[i + 1]
                        tukey_results_dca_anova.append({
                            "Group1": row[0],
                            "Group2": row[1],
                            "Mean_Difference": f"{row[2]:.4g}",
                            "P_Value": f"{row[3]:.4g}",
                            "Lower": f"{row[4]:.4g}",
                            "Upper": f"{row[5]:.4g}",
                            "Reject_Null": "Yes" if row[6]==True else "No"
                        })

                    context['tukey_results_dca_anova'] = tukey_results_dca_anova
                    request.session['tukey_results_dca_anova'] = tukey_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Tukey test: {str(e)}"
                    context['tukey_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "bonferroni" and reject_null_dca_anova == "Yes":
                try:
                    unique_groups = df['Factor'].unique()
                    group_pairs = list(combinations(unique_groups, 2))
                    
                    comparisons = []
                    p_values = []

                    for group1, group2 in group_pairs:
                        group1_data = df[df["Factor"] == group1]["Response"]
                        group2_data = df[df["Factor"] == group2]["Response"]

                        t_stat, p_val = ttest_ind(group1_data, group2_data)
                        comparisons.append((group1, group2, t_stat, p_val))
                        p_values.append(p_val)

                    _, bonferroni_corrected_p_values, _, _ = multitest.multipletests(p_values, alpha=((100 - alpha_value_dca_anova) / 100), method="bonferroni")
                    
                    bonferroni_results_dca_anova = []
                    for (group1, group2, t_stat, raw_p_val), corrected_p_val in zip(comparisons, bonferroni_corrected_p_values):
                        bonferroni_results_dca_anova.append({
                            "Group1": group1,
                            "Group2": group2,
                            "T_statistic": f"{t_stat:.4g}",
                            "Raw_P_Value": f"{raw_p_val:.4g}",
                            "Bonferroni_P_Value": f"{corrected_p_val:.4g}",
                            "Reject_Null": "Yes" if corrected_p_val < ((100 - alpha_value_dca_anova) / 100) else "No"
                        })

                    context['bonferroni_results_dca_anova'] = bonferroni_results_dca_anova
                    request.session['bonferroni_results_dca_anova'] = bonferroni_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Bonferroni test: {str(e)}"
                    context['bonferroni_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "scheffe" and reject_null_dca_anova == "Yes":
                try:
                    unique_groups = df['Factor'].unique()
                    group_pairs = list(combinations(unique_groups, 2))
                    
                    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                    df_residual = anova_table.loc["Residual", "df"]
                    k = len(unique_groups)

                    scheffe_results_dca_anova = []

                    for group1, group2 in group_pairs:
                        group1_data = df[df["Factor"] == group1]["Response"]
                        group2_data = df[df["Factor"] == group2]["Response"]

                        n1 = len(group1_data)
                        n2 = len(group2_data)
                        mean1 = group1_data.mean()
                        mean2 = group2_data.mean()

                        mean_diff = mean1 - mean2

                        scheffe_stat = (mean_diff**2) / (mse * (1/n1 + 1/n2))
                        scheffe_critical_value = (k - 1) * f.ppf(1 - ((100 - alpha_value_dca_anova) / 100), k - 1, df_residual)

                        reject_null = "Yes" if scheffe_stat > scheffe_critical_value else "No"

                        scheffe_results_dca_anova.append({
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "Scheffe_Statistic": f"{scheffe_stat:.4g}",
                            "Critical_Value": f"{scheffe_critical_value:.4g}",
                            "Reject_Null": reject_null
                        })

                    context['scheffe_results_dca_anova'] = scheffe_results_dca_anova
                    request.session['scheffe_results_dca_anova'] = scheffe_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Scheffé test: {str(e)}"
                    context['scheffe_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "dunnett" and reject_null_dca_anova == "Yes":
                try:

                    if "control" not in df['Factor'].unique():
                        context['post_hoc_error'] = "Error: No group named 'control' found. Please provide a group named 'control'."
                        context['dunnett_results'] = None
                    else:
                        control_group = df[df["Factor"] == "control"]["Response"]
                        other_groups = df['Factor'].unique()
                        other_groups = [group for group in other_groups if group != "control"]

                        dunnett_results_dca_anova = []
                        p_values = []

                        mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                        df_residual = anova_table.loc["Residual", "df"]
                        n_control = len(control_group)

                        for group in other_groups:
                            group_data = df[df["Factor"] == group]["Response"]
                            n_group = len(group_data)
                            mean_diff = control_group.mean() - group_data.mean()

                            pooled_se = (mse * (1 / n_control + 1 / n_group)) ** 0.5
                            t_stat = mean_diff / pooled_se
                            p_val = 2 * t.sf(abs(t_stat), df_residual)

                            p_values.append(p_val)

                            dunnett_results_dca_anova.append({
                                "Control_Group": "control",
                                "Comparison_Group": group,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "T_Statistic": f"{t_stat:.4g}",
                                "Raw_P_Value": f"{p_val:.4g}",
                            })

                        _, dunnett_corrected_p_values, _, _ = multipletests(p_values, alpha=((100 - alpha_value_dca_anova) / 100), method="bonferroni")

                        for i, corrected_p_val in enumerate(dunnett_corrected_p_values):
                            dunnett_results_dca_anova[i]["Corrected_P_Value"] = f"{corrected_p_val:.4g}"
                            dunnett_results_dca_anova[i]["Reject_Null"] = "Yes" if corrected_p_val < ((100 - alpha_value_dca_anova) / 100) else "No"

                        context['dunnett_results_dca_anova'] = dunnett_results_dca_anova
                        request.session['dunnett_results_dca_anova'] = dunnett_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Dunnett test: {str(e)}"
                    context['dunnett_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "duncan" and  reject_null_dca_anova == "Yes":
                try:
                    group_means = df.groupby("Factor")["Response"].mean()
                    group_sizes = df.groupby("Factor")["Response"].size()
                    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                    df_residual = anova_table.loc["Residual", "df"]

                    sorted_groups = group_means.sort_values().index
                    sorted_means = group_means[sorted_groups]
                    sorted_sizes = group_sizes[sorted_groups]

                    duncan_results_dca_anova = []
                    group_pairs = []

                    for i in range(len(sorted_groups)):
                        for j in range(i + 1, len(sorted_groups)):
                            group1 = sorted_groups[i]
                            group2 = sorted_groups[j]

                            mean_diff = sorted_means[group2] - sorted_means[group1]
                            pooled_se = (mse * (1 / sorted_sizes[group1] + 1 / sorted_sizes[group2])) ** 0.5
                            q_stat = mean_diff / pooled_se

                            num_groups = j - i + 1
                            critical_q = t.ppf(1 - ((100 - alpha_value_dca_anova) / 100) / 2, df_residual) * (num_groups ** 0.5)

                            reject_null = "Yes" if abs(q_stat) > critical_q else "No"

                            duncan_results_dca_anova.append({
                                "Group1": group1,
                                "Group2": group2,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "Q_Statistic": f"{q_stat:.4g}",
                                "Critical_Q": f"{critical_q:.4g}",
                                "Reject_Null": reject_null
                            })

                    context['duncan_results_dca_anova'] = duncan_results_dca_anova
                    request.session['duncan_results_dca_anova'] = duncan_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Duncan test: {str(e)}"
                    context['duncan_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "holm" and reject_null_dca_anova == "Yes":
                try:
                    group_names = df['Factor'].unique()
                    pairwise_results = []
                    p_values = []
                    comparisons = []

                    for i, group1 in enumerate(group_names):
                        for group2 in group_names[i + 1:]:
                            responses1 = df[df['Factor'] == group1]['Response']
                            responses2 = df[df['Factor'] == group2]['Response']
                            
                            t_stat, p_value = ttest_ind(responses1, responses2)
                            p_values.append(p_value)
                            comparisons.append((group1, group2))
                    
                    sorted_indices = np.argsort(p_values)
                    sorted_p_values = np.array(p_values)[sorted_indices]
                    sorted_comparisons = np.array(comparisons)[sorted_indices]

                    m = len(sorted_p_values)
                    holm_results_dca_anova = []
                    for rank, (comparison, p_value) in enumerate(zip(sorted_comparisons, sorted_p_values), start=1):
                        adjusted_alpha = ((100 - alpha_value_dca_anova) / 100) / (m - rank + 1)
                        reject_null = "Yes" if p_value < adjusted_alpha else "No"

                        holm_results_dca_anova.append({
                            "Group1": comparison[0],
                            "Group2": comparison[1],
                            "Original_P_Value": f"{p_value:.4g}",
                            "Adjusted_Alpha": f"{adjusted_alpha:.4g}",
                            "Reject_Null": reject_null
                        })

                    context['holm_results_dca_anova'] = holm_results_dca_anova
                    request.session['holm_results_dca_anova'] = holm_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Holm test: {str(e)}"
                    context['holm_results_dca_anova'] = None

            elif post_hoc_test_type_dca_anova == "fisher_lsd" and reject_null_dca_anova == "Yes":
                try:
                    comparisons = list(itertools.combinations(df['Factor'].unique(), 2))
                    fisher_lsd_results_dca_anova = []

                    group_data = {factor: df[df['Factor'] == factor]['Response'] for factor in df['Factor'].unique()}

                    for group1, group2 in comparisons:
                        data1 = group_data[group1]
                        data2 = group_data[group2]

                        t_stat, p_value = ttest_ind(data1, data2, equal_var=True)

                        fisher_lsd_results_dca_anova.append({
                            "Group1": group1,
                            "Group2": group2,
                            "T_Statistic": f"{t_stat:.4g}",
                            "P_Value": f"{p_value:.4g}",
                            "Reject_Null": "Yes" if p_value < ((100 - alpha_value_dca_anova) / 100) else "No"
                        })

                    context['fisher_lsd_results_dca_anova'] = fisher_lsd_results_dca_anova
                    request.session['fisher_lsd_results_dca_anova'] = fisher_lsd_results_dca_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Fisher LSD test: {str(e)}"
                    context['fisher_lsd_results_dca_anova'] = None

            model = sm.OLS.from_formula("Response ~ C(Factor)", data=df).fit()
            residuals = model.resid

            shapiro_results_dca_anova = shapiro(residuals)

            context['shapiro_results_dca_anova'] = {
                "W_statistic": f"{shapiro_results_dca_anova.statistic:.4g}",
                "p_value": f"{shapiro_results_dca_anova.pvalue:.4g}",
                "normality": "Yes" if shapiro_results_dca_anova.pvalue > ((100 - alpha_value_dca_anova) / 100) else "No"
            }
            request.session['shapiro_results_dca_anova'] = shapiro_results_dca_anova

            if shapiro_results_dca_anova.pvalue <= ((100 - alpha_value_dca_anova) / 100):
                warning_3_dca_anova = "Shapiro-Wilk Test for normality failed. Data may not be normally distributed."
                context['warning_3_dca_anova'] = warning_3_dca_anova
                request.session['warning_3_dca_anova'] = warning_3_dca_anova

            levene_results_dca_anova = levene(*groups)

            context['levene_results_dca_anova'] = {
                "W_statistic": f"{levene_results_dca_anova.statistic:.4g}",
                "p_value": f"{levene_results_dca_anova.pvalue:.4g}",
                "homogeneity": "Yes" if levene_results_dca_anova.pvalue > ((100 - alpha_value_dca_anova) / 100) else "No"
            }
            request.session['levene_results_dca_anova'] = levene_results_dca_anova

            if levene_results_dca_anova.pvalue <= ((100 - alpha_value_dca_anova) / 100):
                warning_4_dca_anova = "Levene Test for homocedasticity failed. Data may not have homogenous variance."
                context['warning_4_dca_anova'] = warning_4_dca_anova
                request.session['warning_4_dca_anova'] = warning_4_dca_anova

            group_means = df.groupby('Factor')['Response'].mean()
            group_sems = df.groupby('Factor')['Response'].sem()
            ci_95 = 1.96 * group_sems

            factor_name = headers_dca_anova[0] if headers_dca_anova else "Factor"
            response_name = headers_dca_anova[1] if headers_dca_anova else "Response"
            
            fig, ax = plt.subplots()
            bars = ax.bar(group_means.index, group_means.values, yerr=ci_95.values, capsize=5, alpha=0.7, color='skyblue')
            
            ax.set_title('Group Means with 95% Confidence Intervals', fontsize=16)
            ax.set_xlabel(factor_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['bar_plot_dca_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['bar_plot_dca_anova'] = context['bar_plot_dca_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x="Factor", y="Response", data=df, ax=ax)
            ax.set_title("ANOVA - Boxplot", fontsize=16)
            ax.set_xlabel(factor_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['graph_dca_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['graph_dca_anova'] = context['graph_dca_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=residuals, lowess=False, line_kws={"color": "red"}, ax=ax)
            ax.axhline(0, color='black', linestyle='--')
            ax.set_title("Residuals vs Fitted", fontsize=16)
            ax.set_xlabel("Fitted Values", fontsize=14)
            ax.set_ylabel("Residuals", fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['residuals_plot_dca_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['residuals_plot_dca_anova'] = context['residuals_plot_dca_anova']
            buf.close()
            plt.close(fig)

            fig = plt.figure()
            sm.qqplot(residuals, line='45', fit=True, ax=fig.add_subplot(111))
            plt.title("Normal QQ-Plot", fontsize=16)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['qq_plot_dca_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['qq_plot_dca_anova'] = context['qq_plot_dca_anova']
            buf.close()
            plt.close(fig)

        except Exception as e:
            context['error_dca_anova'] = str(e)
            context['anova_results_dca_anova'] = None
            context['tukey_results_dca_anova'] = None
            context['bonferroni_results_dca_anova'] = None
            context['scheffe_results_dca_anova'] = None
            context['dunnett_results_dca_anova'] = None
            context['duncan_results_dca_anova'] = None
            context['holm_results_dca_anova'] = None
            context['fisher_lsd_results_dca_anova'] = None
            context['shapiro_results_dca_anova'] = None
            context['levene_results_dca_anova'] = None
            context['graph_dca_anova'] = None
            context['residuals_plot_dca_anova'] = None
            context['qq_plot_dca_anova'] = None
            context['bar_plot_dca_anova'] = None
            context['explanation_dca_anova'] = None
            context['warning_1_dca_anova'] = None
            context['warning_2_dca_anova'] = None
            context['warning_3_dca_anova'] = None
            context['warning_4_dca_anova'] = None

###############################################################################################################
    if request.method == "POST" and tab == "one_way_anova" and subtab == "dba_anova":
        data_dba_anova = request.POST.get('data_dba_anova')
        use_first_row_as_header_dba_anova = request.POST.get('use_first_row_as_header_dba_anova') == 'on'
        alpha_value_dba_anova = request.POST.get('alpha_value_dba_anova')
        post_hoc_test_type_dba_anova = request.POST.get('post_hoc_test_type_dba_anova')

        context['use_first_row_as_header_dba_anova'] = 'checked' if use_first_row_as_header_dba_anova else ''
        request.session['use_first_row_as_header_dba_anova'] = use_first_row_as_header_dba_anova
        context['data_dba_anova'] = data_dba_anova
        request.session['data_dba_anova'] = data_dba_anova
        context['post_hoc_test_type_dba_anova'] = post_hoc_test_type_dba_anova
        request.session['post_hoc_test_type_dba_anova'] = post_hoc_test_type_dba_anova

        if alpha_value_dba_anova:
            try:
                alpha_value_dba_anova = float(alpha_value_dba_anova)
                context['alpha_value_dba_anova'] = alpha_value_dba_anova
                request.session['alpha_value_dba_anova'] = alpha_value_dba_anova
            except ValueError:
                alpha_value_dba_anova = None
        
        if not alpha_value_dba_anova:
            alpha_value_dba_anova = 95
            context['alpha_value_dba_anova'] = alpha_value_dba_anova
            request.session['alpha_value_dba_anova'] = alpha_value_dba_anova
            warning_1_dba_anova = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_dba_anova'] = warning_1_dba_anova
            request.session['warning_1_dba_anova'] = warning_1_dba_anova

        if not data_dba_anova.strip():
            context['error_dba_anova'] = "Please enter data before calculating."
            context['anova_results_dba_anova'] = None
            context['tukey_results_dba_anova'] = None
            context['bonferroni_results_dba_anova'] = None
            context['scheffe_results_dba_anova'] = None
            context['dunnett_results_dba_anova'] = None
            context['duncan_results_dba_anova'] = None
            context['holm_results_dba_anova'] = None
            context['fisher_lsd_results_dba_anova'] = None
            context['shapiro_results_dba_anova'] = None
            context['levene_results_dba_anova'] = None
            context['graph_dba_anova'] = None
            context['bar_plot_dba_anova'] = None
            context['qq_plot_dba_anova'] = None
            context['residuals_plot_dba_anova'] = None
            context['explanation_dba_anova'] = None
            context['warning_1_dba_anova'] = None
            context['warning_2_dba_anova'] = None
            context['warning_3_dba_anova'] = None
            context['warning_4_dba_anova'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_dba_anova = data_dba_anova.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_dba_anova.split('\n')]

            if use_first_row_as_header_dba_anova:
                headers_dba_anova = rows[0]
                rows = rows[1:]
            else:
                headers_dba_anova = ["Block", "Factor", "Response"]

            blocks, factors, responses = zip(*rows)
            blocks = list(blocks)
            factors = list(factors)
            responses = [float(r) for r in responses]

            df = pd.DataFrame({"Block": blocks, "Factor": factors, "Response": responses})

            if df.isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")

            if df["Response"].dtype not in [float, int]:
                raise ValueError("Response column contains non-numeric values.")

            model = sm.OLS.from_formula("Response ~ C(Factor) + C(Block)", data=df).fit()
            anova_table = anova_lm(model, typ=2)

            reject_null_dba_anova = "Yes" if anova_table.loc["C(Factor)", "PR(>F)"] <= ((100 - alpha_value_dba_anova) / 100) else "No"

            anova_results_dba_anova = [
                {
                    "Source": (
                        headers_dba_anova[1] if row.name == "C(Factor)" else
                        headers_dba_anova[0] if row.name == "C(Block)" else
                        row.name
                    ),
                    "Sum_of_Squares": "{:.5g}".format(row["sum_sq"]) if pd.notna(row["sum_sq"]) else "",
                    "Degrees_of_Freedom": "{:.5g}".format(row["df"]) if pd.notna(row["df"]) else "",
                    "MS": "{:.5g}".format(row["sum_sq"] / row["df"]) if pd.notna(row["sum_sq"]) and pd.notna(row["df"]) else "",
                    "F_statistic": "{:.5g}".format(row["F"]) if pd.notna(row["F"]) else "",
                    "P_value": "{:.5g}".format(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else "",
                    "Reject_Null": "Yes" if row.name == "C(Factor)" and reject_null_dba_anova == "Yes" else "",
                }
                for _, row in anova_table.iterrows()
            ]

            context['anova_results_dba_anova'] = anova_results_dba_anova
            request.session['anova_results_dba_anova'] = anova_results_dba_anova
            explanation_dba_anova = "Null Hypothesis (H₀): Group means are equal - Alternative Hypothesis (H₁): At least one group mean differs."
            context['explanation_dba_anova'] = explanation_dba_anova
            request.session['explanation_dba_anova'] = explanation_dba_anova

            if reject_null_dba_anova == "No":
                warning_2_dba_anova = "p-value greater than alpha post-hoc test omitted."
                context['warning_2_dba_anova'] = warning_2_dba_anova
                request.session['warning_2_dba_anova'] = warning_2_dba_anova

            if post_hoc_test_type_dba_anova == "tukey" and reject_null_dba_anova == "Yes":
                try:
                    tukey = pairwise_tukeyhsd(endog=df['Response'], groups=df['Factor'], alpha=((100 - alpha_value_dba_anova) / 100))

                    tukey_results_dba_anova = []
                    for i in range(len(tukey.summary().data[1:])):
                        row = tukey.summary().data[i + 1]
                        tukey_results_dba_anova.append({
                            "Group1": row[0],
                            "Group2": row[1],
                            "Mean_Difference": f"{row[2]:.4g}",
                            "P_Value": f"{row[3]:.4g}",
                            "Lower": f"{row[4]:.4g}",
                            "Upper": f"{row[5]:.4g}",
                            "Reject_Null": "Yes" if row[6]==True else "No"
                        })
                    context['tukey_results_dba_anova'] = tukey_results_dba_anova
                    request.session['tukey_results_dba_anova'] = tukey_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Tukey test: {str(e)}"
                    context['tukey_results_dba_anova'] = None

            elif post_hoc_test_type_dba_anova == "bonferroni" and reject_null_dba_anova == "Yes":
                try:
                    unique_groups = df['Factor'].unique()
                    group_pairs = list(combinations(unique_groups, 2))
                    comparisons, p_values = [], []

                    for group1, group2 in group_pairs:
                        group1_data = df[df["Factor"] == group1]["Response"]
                        group2_data = df[df["Factor"] == group2]["Response"]
                        t_stat, p_val = ttest_ind(group1_data, group2_data)
                        comparisons.append((group1, group2, t_stat, p_val))
                        p_values.append(p_val)

                    _, bonferroni_corrected_p_values, _, _ = multipletests(p_values, alpha=((100 - alpha_value_dba_anova) / 100), method="bonferroni")

                    bonferroni_results_dba_anova = [
                        {
                            "Group1": group1,
                            "Group2": group2,
                            "T_statistic": f"{t_stat:.4g}",
                            "Raw_P_Value": f"{raw_p_val:.4g}",
                            "Bonferroni_P_Value": f"{corrected_p_val:.4g}",
                            "Reject_Null": "Yes" if corrected_p_val < ((100 - alpha_value_dba_anova) / 100) else "No"
                        }
                        for (group1, group2, t_stat, raw_p_val), corrected_p_val in zip(comparisons, bonferroni_corrected_p_values)
                    ]

                    context['bonferroni_results_dba_anova'] = bonferroni_results_dba_anova
                    request.session['bonferroni_results_dba_anova'] = bonferroni_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Bonferroni test: {str(e)}"            

            elif post_hoc_test_type_dba_anova == "scheffe" and reject_null_dba_anova == "Yes":
                try:
                    unique_groups = df['Factor'].unique()
                    group_pairs = list(combinations(unique_groups, 2))
                    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                    df_residual = anova_table.loc["Residual", "df"]
                    k = len(unique_groups)

                    scheffe_results_dba_anova = []
                    for group1, group2 in group_pairs:
                        group1_data = df[df["Factor"] == group1]["Response"]
                        group2_data = df[df["Factor"] == group2]["Response"]
                        n1, n2 = len(group1_data), len(group2_data)
                        mean_diff = group1_data.mean() - group2_data.mean()
                        scheffe_stat = (mean_diff**2) / (mse * (1/n1 + 1/n2))
                        scheffe_critical_value = (k - 1) * f.ppf(1 - ((100 - alpha_value_dba_anova) / 100), k - 1, df_residual)
                        reject_null = "Yes" if scheffe_stat > scheffe_critical_value else "No"

                        scheffe_results_dba_anova.append({
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "Scheffe_Statistic": f"{scheffe_stat:.4g}",
                            "Critical_Value": f"{scheffe_critical_value:.4g}",
                            "Reject_Null": reject_null
                        })

                    context['scheffe_results_dba_anova'] = scheffe_results_dba_anova
                    request.session['scheffe_results_dba_anova'] = scheffe_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Scheffé test: {str(e)}"

            elif post_hoc_test_type_dba_anova == "dunnett" and reject_null_dba_anova == "Yes":
                try:
                    if "control" not in df['Factor'].unique():
                        context['post_hoc_error'] = "No group named 'control' found. Please provide a 'control' group."
                    else:
                        control_group = df[df["Factor"] == "control"]["Response"]
                        other_groups = [g for g in df['Factor'].unique() if g != "control"]

                        dunnett_results_dba_anova = []
                        mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                        df_residual = anova_table.loc["Residual", "df"]
                        n_control = len(control_group)

                        for group in other_groups:
                            group_data = df[df["Factor"] == group]["Response"]
                            n_group = len(group_data)
                            mean_diff = control_group.mean() - group_data.mean()
                            pooled_se = (mse * (1 / n_control + 1 / n_group)) ** 0.5
                            t_stat = mean_diff / pooled_se
                            p_val = 2 * t.sf(abs(t_stat), df_residual)

                            dunnett_results_dba_anova.append({
                                "Control_Group": "control",
                                "Comparison_Group": group,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "T_Statistic": f"{t_stat:.4g}",
                                "Raw_P_Value": f"{p_val:.4g}",
                                "Reject_Null": "Yes" if p_val < ((100 - alpha_value_dba_anova) / 100) else "No"
                            })

                        context['dunnett_results_dba_anova'] = dunnett_results_dba_anova
                        request.session['dunnett_results_dba_anova'] = dunnett_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Dunnett test: {str(e)}"

            elif post_hoc_test_type_dba_anova == "duncan" and reject_null_dba_anova == "Yes":
                try:
                    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
                    df_residual = anova_table.loc["Residual", "df"]

                    group_means = df.groupby("Factor")["Response"].mean()
                    group_sizes = df.groupby("Factor")["Response"].size()

                    sorted_groups = group_means.sort_values().index
                    sorted_means = group_means[sorted_groups]
                    sorted_sizes = group_sizes[sorted_groups]

                    duncan_results_dba_anova = []

                    for i in range(len(sorted_groups)):
                        for j in range(i + 1, len(sorted_groups)):
                            group1 = sorted_groups[i]
                            group2 = sorted_groups[j]

                            mean_diff = sorted_means[group2] - sorted_means[group1]
                            pooled_se = (mse * (1 / sorted_sizes[group1] + 1 / sorted_sizes[group2])) ** 0.5
                            q_stat = mean_diff / pooled_se

                            num_groups = j - i + 1
                            critical_q = t.ppf(1 - ((100 - alpha_value_dba_anova) / 100) / 2, df_residual) * (num_groups ** 0.5)

                            reject_null = "Yes" if abs(q_stat) > critical_q else "No"

                            duncan_results_dba_anova.append({
                                "Group1": group1,
                                "Group2": group2,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "Q_Statistic": f"{q_stat:.4g}",
                                "Critical_Q": f"{critical_q:.4g}",
                                "Reject_Null": reject_null
                            })

                    context['duncan_results_dba_anova'] = duncan_results_dba_anova
                    request.session['duncan_results_dba_anova'] = duncan_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Duncan test: {str(e)}"
                    context['duncan_results_dba_anova'] = None

            elif post_hoc_test_type_dba_anova == "holm" and reject_null_dba_anova == "Yes":
                try:
                    group_names = df['Factor'].unique()
                    p_values = []
                    comparisons = []

                    for i, group1 in enumerate(group_names):
                        for group2 in group_names[i + 1:]:
                            responses1 = model.fittedvalues[df['Factor'] == group1]
                            responses2 = model.fittedvalues[df['Factor'] == group2]

                            t_stat, p_value = ttest_ind(responses1, responses2, equal_var=True)
                            p_values.append(p_value)
                            comparisons.append((group1, group2))

                    sorted_indices = np.argsort(p_values)
                    sorted_p_values = np.array(p_values)[sorted_indices]
                    sorted_comparisons = np.array(comparisons)[sorted_indices]

                    m = len(sorted_p_values)
                    holm_results_dba_anova = []

                    for rank, (comparison, p_value) in enumerate(zip(sorted_comparisons, sorted_p_values), start=1):
                        adjusted_alpha = ((100 - alpha_value_dba_anova) / 100) / (m - rank + 1)
                        reject_null = "Yes" if p_value < adjusted_alpha else "No"

                        holm_results_dba_anova.append({
                            "Group1": comparison[0],
                            "Group2": comparison[1],
                            "Original_P_Value": f"{p_value:.4g}",
                            "Adjusted_Alpha": f"{adjusted_alpha:.4g}",
                            "Reject_Null": reject_null
                        })

                    context['holm_results_dba_anova'] = holm_results_dba_anova
                    request.session['holm_results_dba_anova'] = holm_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Holm test: {str(e)}"
                    context['holm_results_dba_anova'] = None

            elif post_hoc_test_type_dba_anova == "fisher_lsd" and reject_null_dba_anova == "Yes":
                try:
                    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]

                    comparisons = list(itertools.combinations(df['Factor'].unique(), 2))
                    fisher_lsd_results_dba_anova = []

                    group_data = {factor: model.fittedvalues[df['Factor'] == factor] for factor in df['Factor'].unique()}

                    for group1, group2 in comparisons:
                        data1 = group_data[group1]
                        data2 = group_data[group2]

                        mean_diff = np.mean(data2) - np.mean(data1)
                        pooled_se = (mse * (1 / len(data1) + 1 / len(data2))) ** 0.5
                        t_stat = mean_diff / pooled_se

                        p_value = 2 * (1 - t.cdf(abs(t_stat), df=anova_table.loc["Residual", "df"]))

                        fisher_lsd_results_dba_anova.append({
                            "Group1": group1,
                            "Group2": group2,
                            "T_Statistic": f"{t_stat:.4g}",
                            "P_Value": f"{p_value:.4g}",
                            "Reject_Null": "Yes" if p_value < ((100 - alpha_value_dba_anova) / 100) else "No"
                        })

                    context['fisher_lsd_results_dba_anova'] = fisher_lsd_results_dba_anova
                    request.session['fisher_lsd_results_dba_anova'] = fisher_lsd_results_dba_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Fisher LSD test: {str(e)}"
                    context['fisher_lsd_results_dba_anova'] = None

            residuals = model.resid
            shapiro_results_dba_anova = shapiro(residuals)
            levene_results_dba_anova = levene(*[df[df["Factor"] == level]["Response"] for level in df["Factor"].unique()])

            context['shapiro_results_dba_anova'] = {
                "W_statistic": f"{shapiro_results_dba_anova.statistic:.4g}",
                "p_value": f"{shapiro_results_dba_anova.pvalue:.4g}",
                "normality": "Yes" if shapiro_results_dba_anova.pvalue > ((100 - alpha_value_dba_anova) / 100) else "No"
            }
            request.session['shapiro_results_dba_anova'] = shapiro_results_dba_anova

            if shapiro_results_dba_anova.pvalue <= ((100 - alpha_value_dba_anova) / 100):
                warning_3_dba_anova = "Shapiro-Wilk Test for normality failed. Data may not be normally distributed."
                context['warning_3_dba_anova'] = warning_3_dba_anova
                request.session['warning_3_dba_anova'] = warning_3_dba_anova

            context['levene_results_dba_anova'] = {
                "W_statistic": f"{levene_results_dba_anova.statistic:.4g}",
                "p_value": f"{levene_results_dba_anova.pvalue:.4g}",
                "homogeneity": "Yes" if levene_results_dba_anova.pvalue > ((100 - alpha_value_dba_anova) / 100) else "No"
            }
            request.session['levene_results_dba_anova'] = levene_results_dba_anova

            if levene_results_dba_anova.pvalue <= ((100 - alpha_value_dba_anova) / 100):
                warning_4_dba_anova = "Levene Test for homocedasticity failed. Data may not have homogenous variance."
                context['warning_4_dba_anova'] = warning_4_dba_anova
                request.session['warning_4_dba_anova'] = warning_4_dba_anova

            group_means = df.groupby('Factor')['Response'].mean()
            group_sems = df.groupby('Factor')['Response'].sem()
            ci_95 = 1.96 * group_sems

            factor_name = headers_dba_anova[1] if headers_dba_anova else "Factor"
            response_name = headers_dba_anova[2] if headers_dba_anova else "Response"

            fig, ax = plt.subplots()
            bars = ax.bar(group_means.index, group_means.values, yerr=ci_95.values, capsize=5, alpha=0.7, color='skyblue')

            ax.set_title('Group Means with 95% Confidence Intervals', fontsize=16)
            ax.set_xlabel(factor_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['bar_plot_dba_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['bar_plot_dba_anova'] = context['bar_plot_dba_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x="Factor", y="Response", data=df, ax=ax)
            ax.set_title("ANOVA - Boxplot", fontsize=16)
            ax.set_xlabel(factor_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['graph_dba_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['graph_dba_anova'] = context['graph_dba_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=residuals, lowess=False, line_kws={"color": "red"}, ax=ax)
            ax.axhline(0, color='black', linestyle='--')
            ax.set_title("Residuals vs Fitted", fontsize=16)
            ax.set_xlabel("Fitted Values", fontsize=14)
            ax.set_ylabel("Residuals", fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['residuals_plot_dba_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['residuals_plot_dba_anova'] = context['residuals_plot_dba_anova']
            buf.close()
            plt.close(fig)

            fig = plt.figure()
            sm.qqplot(residuals, line='45', fit=True, ax=fig.add_subplot(111))
            plt.title("Normal QQ-Plot", fontsize=16)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['qq_plot_dba_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['qq_plot_dba_anova'] = context['qq_plot_dba_anova']
            buf.close()
            plt.close(fig)

        except Exception as e:
            context['error_dba_anova'] = str(e)
            context['anova_results_dba_anova'] = None
            context['tukey_results_dba_anova'] = None
            context['bonferroni_results_dba_anova'] = None
            context['scheffe_results_dba_anova'] = None
            context['dunnett_results_dba_anova'] = None
            context['duncan_results_dba_anova'] = None
            context['holm_results_dba_anova'] = None
            context['fisher_lsd_results_dba_anova'] = None
            context['shapiro_results_dba_anova'] = None
            context['levene_results_dba_anova'] = None
            context['graph_dba_anova'] = None
            context['residuals_plot_dba_anova'] = None
            context['qq_plot_dba_anova'] = None
            context['bar_plot_dba_anova'] = None
            context['explanation_dba_anova'] = None
            context['warning_1_dba_anova'] = None
            context['warning_2_dba_anova'] = None
            context['warning_3_dba_anova'] = None
            context['warning_4_dba_anova'] = None

#########################################################################################################
    if request.method == "POST" and tab == "two_way_anova":
        data_two_way_anova = request.POST.get('data_two_way_anova')
        use_first_row_as_header_two_way_anova = request.POST.get('use_first_row_as_header_two_way_anova') == 'on'
        alpha_value_two_way_anova = request.POST.get('alpha_value_two_way_anova')
        post_hoc_test_type_two_way_anova = request.POST.get('post_hoc_test_type_two_way_anova')

        context['use_first_row_as_header_two_way_anova'] = 'checked' if use_first_row_as_header_two_way_anova else ''
        request.session['use_first_row_as_header_two_way_anova'] = use_first_row_as_header_two_way_anova
        context['data_two_way_anova'] = data_two_way_anova
        request.session['data_two_way_anova'] = data_two_way_anova
        context['post_hoc_test_type_two_way_anova'] = post_hoc_test_type_two_way_anova
        request.session['post_hoc_test_type_two_way_anova'] = post_hoc_test_type_two_way_anova

        if alpha_value_two_way_anova:
            try:
                alpha_value_two_way_anova = float(alpha_value_two_way_anova)
                context['alpha_value_two_way_anova'] = alpha_value_two_way_anova
                request.session['alpha_value_two_way_anova'] = alpha_value_two_way_anova
            except ValueError:
                alpha_value_two_way_anova = None

        if not alpha_value_two_way_anova:
            alpha_value_two_way_anova = 95
            context['alpha_value_two_way_anova'] = alpha_value_two_way_anova
            request.session['alpha_value_two_way_anova'] = alpha_value_two_way_anova
            warning_1_two_way_anova = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_two_way_anova'] = warning_1_two_way_anova
            request.session['warning_1_two_way_anova'] = warning_1_two_way_anova

        if not data_two_way_anova.strip():
            context['error_two_way_anova'] = "Please enter data before calculating."
            context['anova_results_two_way_anova'] = None
            context['tukey_results_two_way_anova'] = None
            context['bonferroni_results_two_way_anova'] = None
            context['scheffe_results_two_way_anova'] = None
            context['holm_results_two_way_anova'] = None
            context['fisher_lsd_results_two_way_anova'] = None
            context['shapiro_results_two_way_anova'] = None
            context['levene_results_two_way_anova'] = None
            context['graph_two_way_anova'] = None
            context['bar_plot_two_way_anova'] = None
            context['qq_plot_two_way_anova'] = None
            context['residuals_plot_two_way_anova'] = None
            context['interaction_plot_two_way_anova'] = None
            context['explanation_two_way_anova'] = None
            context['warning_1_two_way_anova'] = None
            context['warning_3_two_way_anova'] = None
            context['warning_4_two_way_anova'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_two_way_anova = data_two_way_anova.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_two_way_anova.split('\n')]

            if use_first_row_as_header_two_way_anova:
                headers_two_way_anova = rows[0]
                rows = rows[1:]
            else:
                headers_two_way_anova = ["Factor1", "Factor2", "Response"]

            factors1, factors2, responses = zip(*rows)
            factors1 = list(factors1)
            factors2 = list(factors2)
            responses = [float(r) for r in responses]

            df = pd.DataFrame({"Factor1": factors1, "Factor2": factors2, "Response": responses})

            if df.isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")

            if df["Response"].dtype not in [float, int]:
                raise ValueError("Response column contains non-numeric values.")

            model = sm.OLS.from_formula("Response ~ C(Factor1) * C(Factor2)", data=df).fit()
            anova_table = anova_lm(model, typ=2)
            print(anova_table.index)
            reject_null_two_way_anova = {
                "C(Factor1)": "Yes" if anova_table.loc["C(Factor1)", "PR(>F)"] <= ((100 - alpha_value_two_way_anova) / 100) else "No",
                "C(Factor2)": "Yes" if anova_table.loc["C(Factor2)", "PR(>F)"] <= ((100 - alpha_value_two_way_anova) / 100) else "No",
                "C(Factor1):C(Factor2)": "Yes" if anova_table.loc["C(Factor1):C(Factor2)", "PR(>F)"] <= ((100 - alpha_value_two_way_anova) / 100) else "No",
            }

            anova_results_two_way_anova = [
                {
                    "Source": (
                        headers_two_way_anova[0] if row.name == f"C(Factor1)" else
                        headers_two_way_anova[1] if row.name == f"C(Factor2)" else
                        f"{headers_two_way_anova[0]}:{headers_two_way_anova[1]}" if row.name == f"C(Factor1):C(Factor2)" else
                        row.name
                    ),
                    "Sum_of_Squares": "{:.5g}".format(row["sum_sq"]) if pd.notna(row["sum_sq"]) else "",
                    "Degrees_of_Freedom": "{:.5g}".format(row["df"]) if pd.notna(row["df"]) else "",
                    "MS": "{:.5g}".format(row["sum_sq"] / row["df"]) if pd.notna(row["sum_sq"]) and pd.notna(row["df"]) else "",
                    "F_statistic": "{:.5g}".format(row["F"]) if pd.notna(row["F"]) else "",
                    "P_value": "{:.5g}".format(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else "",
                    "Reject_Null": reject_null_two_way_anova.get(row.name, ""),
                }
                for _, row in anova_table.iterrows()
            ]

            context['anova_results_two_way_anova'] = anova_results_two_way_anova
            request.session['anova_results_two_way_anova'] = anova_results_two_way_anova
            explanation_two_way_anova = "Null Hypothesis (H₀): All group means are equal across factors and their interaction - Alternative Hypothesis (H₁): At least one group mean differs, or there is significant interaction between factors."
            context['explanation_two_way_anova'] = explanation_two_way_anova
            request.session['explanation_two_way_anova'] = explanation_two_way_anova

            if post_hoc_test_type_two_way_anova == "tukey":
                try:
                    tukey_results_two_way_anova = []

                    tukey_factor1 = pairwise_tukeyhsd(
                        endog=df['Response'], 
                        groups=df['Factor1'], 
                        alpha=((100 - alpha_value_two_way_anova) / 100)
                    )
                    for i in range(len(tukey_factor1.summary().data[1:])):
                        row = tukey_factor1.summary().data[i + 1]
                        tukey_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[0] if headers_two_way_anova else f"C(Factor1)",
                            "Group1": row[0],
                            "Group2": row[1],
                            "Mean_Difference": f"{row[2]:.4g}",
                            "P_Value": f"{row[3]:.4g}",
                            "Lower": f"{row[4]:.4g}",
                            "Upper": f"{row[5]:.4g}",
                            "Reject_Null": "Yes" if row[6] else "No"
                        })

                    tukey_factor2 = pairwise_tukeyhsd(
                        endog=df['Response'], 
                        groups=df['Factor2'], 
                        alpha=((100 - alpha_value_two_way_anova) / 100)
                    )
                    for i in range(len(tukey_factor2.summary().data[1:])):
                        row = tukey_factor2.summary().data[i + 1]
                        tukey_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[1] if headers_two_way_anova else f"C(Factor2)",
                            "Group1": row[0],
                            "Group2": row[1],
                            "Mean_Difference": f"{row[2]:.4g}",
                            "P_Value": f"{row[3]:.4g}",
                            "Lower": f"{row[4]:.4g}",
                            "Upper": f"{row[5]:.4g}",
                            "Reject_Null": "Yes" if row[6] else "No"
                        })

                    df['Combined_Factors'] = df['Factor1'].astype(str) + " x " + df['Factor2'].astype(str)
                    tukey_combined = pairwise_tukeyhsd(
                        endog=df['Response'], 
                        groups=df['Combined_Factors'], 
                        alpha=((100 - alpha_value_two_way_anova) / 100)
                    )
                    for i in range(len(tukey_combined.summary().data[1:])):
                        row = tukey_combined.summary().data[i + 1]
                        tukey_results_two_way_anova.append({
                            "Comparison_Type": "Interaction",
                            "Group1": row[0],
                            "Group2": row[1],
                            "Mean_Difference": f"{row[2]:.4g}",
                            "P_Value": f"{row[3]:.4g}",
                            "Lower": f"{row[4]:.4g}",
                            "Upper": f"{row[5]:.4g}",
                            "Reject_Null": "Yes" if row[6] else "No"
                        })

                    context['tukey_results_two_way_anova'] = tukey_results_two_way_anova
                    request.session['tukey_results_two_way_anova'] = tukey_results_two_way_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Tukey test: {str(e)}"
                    context['tukey_results_two_way_anova'] = None

            elif post_hoc_test_type_two_way_anova == "bonferroni":
                try:
                    bonferroni_results_two_way_anova = []

                    levels_factor1 = df['Factor1'].unique()
                    comparisons_factor1 = list(combinations(levels_factor1, 2))
                    for group1, group2 in comparisons_factor1:
                        response1 = df[df['Factor1'] == group1]['Response']
                        response2 = df[df['Factor1'] == group2]['Response']
                        t_stat, p_value = ttest_ind(response1, response2, equal_var=True)
                        adjusted_p = p_value * len(comparisons_factor1)
                        bonferroni_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[0] if headers_two_way_anova else f"C(Factor1)",
                            "Group1": group1,
                            "Group2": group2,
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{min(adjusted_p, 1):.4g}",
                            "Reject_Null": "Yes" if adjusted_p <= ((100 - alpha_value_two_way_anova) / 100) else "No"
                        })

                    levels_factor2 = df['Factor2'].unique()
                    comparisons_factor2 = list(combinations(levels_factor2, 2))
                    for group1, group2 in comparisons_factor2:
                        response1 = df[df['Factor2'] == group1]['Response']
                        response2 = df[df['Factor2'] == group2]['Response']
                        t_stat, p_value = ttest_ind(response1, response2, equal_var=True)
                        adjusted_p = p_value * len(comparisons_factor2)
                        bonferroni_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[1] if headers_two_way_anova else f"C(Factor2)",
                            "Group1": group1,
                            "Group2": group2,
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{min(adjusted_p, 1):.4g}",
                            "Reject_Null": "Yes" if adjusted_p <= ((100 - alpha_value_two_way_anova) / 100) else "No"
                        })

                    df['Combined_Factors'] = df['Factor1'].astype(str) + " x " + df['Factor2'].astype(str)
                    levels_combined = df['Combined_Factors'].unique()
                    comparisons_combined = list(combinations(levels_combined, 2))
                    for group1, group2 in comparisons_combined:
                        response1 = df[df['Combined_Factors'] == group1]['Response']
                        response2 = df[df['Combined_Factors'] == group2]['Response']
                        t_stat, p_value = ttest_ind(response1, response2, equal_var=True)
                        adjusted_p = p_value * len(comparisons_combined)
                        bonferroni_results_two_way_anova.append({
                            "Comparison_Type": "Interaction",
                            "Group1": group1,
                            "Group2": group2,
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{min(adjusted_p, 1):.4g}",
                            "Reject_Null": "Yes" if adjusted_p <= ((100 - alpha_value_two_way_anova) / 100) else "No"
                        })

                    context['bonferroni_results_two_way_anova'] = bonferroni_results_two_way_anova
                    request.session['bonferroni_results_two_way_anova'] = bonferroni_results_two_way_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Bonferroni test: {str(e)}"
                    context['bonferroni_results_two_way_anova'] = None

            elif post_hoc_test_type_two_way_anova == "scheffe":
                try:
                    scheffe_results_two_way_anova = []
                    n_total = len(df)
                    n_groups_factor1 = df['Factor1'].nunique()
                    n_groups_factor2 = df['Factor2'].nunique()
                    df_error = anova_table.loc['Residual', 'df']
                    ms_error = anova_table.loc['Residual', 'sum_sq'] / df_error

                    def scheffe_critical_value(num_groups, df_error, alpha):
                        return (num_groups - 1) * f.ppf(1 - alpha, num_groups - 1, df_error)

                    levels_factor1 = df['Factor1'].unique()
                    group_means_factor1 = df.groupby('Factor1')['Response'].mean()
                    n_per_group_factor1 = df.groupby('Factor1').size()
                    comparisons_factor1 = list(itertools.combinations(levels_factor1, 2))

                    for group1, group2 in comparisons_factor1:
                        mean_diff = abs(group_means_factor1[group1] - group_means_factor1[group2])
                        pooled_variance = ms_error * (1 / n_per_group_factor1[group1] + 1 / n_per_group_factor1[group2])
                        scheffe_stat = mean_diff**2 / pooled_variance
                        critical_value = scheffe_critical_value(n_groups_factor1, df_error, ((100 - alpha_value_two_way_anova) / 100))
                        reject_null = "Yes" if scheffe_stat > critical_value else "No"
                        scheffe_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[0] if headers_two_way_anova else f"C(Factor1)",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "Scheffe_Statistic": f"{scheffe_stat:.4g}",
                            "Critical_Value": f"{critical_value:.4g}",
                            "Reject_Null": reject_null
                        })

                    levels_factor2 = df['Factor2'].unique()
                    group_means_factor2 = df.groupby('Factor2')['Response'].mean()
                    n_per_group_factor2 = df.groupby('Factor2').size()
                    comparisons_factor2 = list(itertools.combinations(levels_factor2, 2))

                    for group1, group2 in comparisons_factor2:
                        mean_diff = abs(group_means_factor2[group1] - group_means_factor2[group2])
                        pooled_variance = ms_error * (1 / n_per_group_factor2[group1] + 1 / n_per_group_factor2[group2])
                        scheffe_stat = mean_diff**2 / pooled_variance
                        critical_value = scheffe_critical_value(n_groups_factor2, df_error, ((100 - alpha_value_two_way_anova) / 100))
                        reject_null = "Yes" if scheffe_stat > critical_value else "No"
                        scheffe_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[1] if headers_two_way_anova else f"C(Factor2)",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "Scheffe_Statistic": f"{scheffe_stat:.4g}",
                            "Critical_Value": f"{critical_value:.4g}",
                            "Reject_Null": reject_null
                        })

                    levels_combined = df['Factor1'].astype(str) + " x " + df['Factor2'].astype(str)
                    df['Combined_Factors'] = levels_combined
                    unique_levels_combined = df['Combined_Factors'].unique()

                    group_means_combined = df.groupby('Combined_Factors')['Response'].mean()
                    n_per_group_combined = df.groupby('Combined_Factors').size()
                    comparisons_combined = set(itertools.combinations(unique_levels_combined, 2))

                    for group1, group2 in comparisons_combined:
                        mean_diff = abs(group_means_combined[group1] - group_means_combined[group2])
                        pooled_variance = ms_error * (1 / n_per_group_combined[group1] + 1 / n_per_group_combined[group2])
                        scheffe_stat = mean_diff**2 / pooled_variance
                        critical_value = scheffe_critical_value(len(unique_levels_combined), df_error, ((100 - alpha_value_two_way_anova) / 100))
                        reject_null = "Yes" if scheffe_stat > critical_value else "No"
                        scheffe_results_two_way_anova.append({
                            "Comparison_Type": "Interaction",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "Scheffe_Statistic": f"{scheffe_stat:.4g}",
                            "Critical_Value": f"{critical_value:.4g}",
                            "Reject_Null": reject_null
                        })

                    context['scheffe_results_two_way_anova'] = scheffe_results_two_way_anova
                    request.session['scheffe_results_two_way_anova'] = scheffe_results_two_way_anova

                except Exception as e:
                    context['post_hoc_error'] = f"Error during Scheffé test: {str(e)}"
                    context['scheffe_results_two_way_anova'] = None

            elif post_hoc_test_type_two_way_anova == "holm":
                try:
                    holm_results_two_way_anova = []
                    alpha = ((100 - alpha_value_two_way_anova) / 100)
                    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
                    df_error = anova_table.loc['Residual', 'df']

                    def holm_adjust(p_values):
                        """Ajuste de Holm para múltiples comparaciones."""
                        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
                        adjusted_p = [0] * len(p_values)
                        m = len(p_values)

                        for rank, (original_idx, p_val) in enumerate(sorted_p):
                            adjusted_p[original_idx] = min(p_val * (m - rank), 1)

                        return adjusted_p

                    levels_factor1 = df['Factor1'].unique()
                    group_means_factor1 = df.groupby('Factor1')['Response'].mean()
                    n_per_group_factor1 = df.groupby('Factor1').size()

                    comparisons_factor1 = []
                    p_values_factor1 = []

                    for group1, group2 in combinations(levels_factor1, 2):
                        mean_diff = group_means_factor1[group1] - group_means_factor1[group2]
                        pooled_variance = ms_error * (1 / n_per_group_factor1[group1] + 1 / n_per_group_factor1[group2])
                        se = np.sqrt(pooled_variance)
                        t_stat = mean_diff / se
                        p_value = t.sf(np.abs(t_stat), df_error) * 2  # Dos colas
                        comparisons_factor1.append((group1, group2, mean_diff, t_stat, p_value))
                        p_values_factor1.append(p_value)

                    adjusted_p_values_factor1 = holm_adjust(p_values_factor1)

                    for (group1, group2, mean_diff, t_stat, p_value), adj_p in zip(comparisons_factor1, adjusted_p_values_factor1):
                        holm_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[0] if headers_two_way_anova else f"C(Factor1)",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "T_Statistic": f"{t_stat:.4g}",
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{adj_p:.4g}",
                            "Reject_Null": "Yes" if adj_p < alpha else "No"
                        })

                    levels_factor2 = df['Factor2'].unique()
                    group_means_factor2 = df.groupby('Factor2')['Response'].mean()
                    n_per_group_factor2 = df.groupby('Factor2').size()

                    comparisons_factor2 = []
                    p_values_factor2 = []

                    for group1, group2 in combinations(levels_factor2, 2):
                        mean_diff = group_means_factor2[group1] - group_means_factor2[group2]
                        pooled_variance = ms_error * (1 / n_per_group_factor2[group1] + 1 / n_per_group_factor2[group2])
                        se = np.sqrt(pooled_variance)
                        t_stat = mean_diff / se
                        p_value = t.sf(np.abs(t_stat), df_error) * 2  # Dos colas
                        comparisons_factor2.append((group1, group2, mean_diff, t_stat, p_value))
                        p_values_factor2.append(p_value)

                    adjusted_p_values_factor2 = holm_adjust(p_values_factor2)

                    for (group1, group2, mean_diff, t_stat, p_value), adj_p in zip(comparisons_factor2, adjusted_p_values_factor2):
                        holm_results_two_way_anova.append({
                            "Comparison_Type": headers_two_way_anova[1] if headers_two_way_anova else f"C(Factor2)",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "T_Statistic": f"{t_stat:.4g}",
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{adj_p:.4g}",
                            "Reject_Null": "Yes" if adj_p < alpha else "No"
                        })

                    df['Combined_Factors'] = df['Factor1'].astype(str) + " x " + df['Factor2'].astype(str)
                    levels_combined = df['Combined_Factors'].unique()
                    group_means_combined = df.groupby('Combined_Factors')['Response'].mean()
                    n_per_group_combined = df.groupby('Combined_Factors').size()

                    comparisons_combined = []
                    p_values_combined = []

                    for group1, group2 in combinations(levels_combined, 2):
                        mean_diff = group_means_combined[group1] - group_means_combined[group2]
                        pooled_variance = ms_error * (1 / n_per_group_combined[group1] + 1 / n_per_group_combined[group2])
                        se = np.sqrt(pooled_variance)
                        t_stat = mean_diff / se
                        p_value = t.sf(np.abs(t_stat), df_error) * 2  # Dos colas
                        comparisons_combined.append((group1, group2, mean_diff, t_stat, p_value))
                        p_values_combined.append(p_value)

                    adjusted_p_values_combined = holm_adjust(p_values_combined)

                    for (group1, group2, mean_diff, t_stat, p_value), adj_p in zip(comparisons_combined, adjusted_p_values_combined):
                        holm_results_two_way_anova.append({
                            "Comparison_Type": "Interaction",
                            "Group1": group1,
                            "Group2": group2,
                            "Mean_Difference": f"{mean_diff:.4g}",
                            "T_Statistic": f"{t_stat:.4g}",
                            "P_Value": f"{p_value:.4g}",
                            "Adjusted_P_Value": f"{adj_p:.4g}",
                            "Reject_Null": "Yes" if adj_p < alpha else "No"
                        })

                    context['holm_results_two_way_anova'] = holm_results_two_way_anova
                    request.session['holm_results_two_way_anova'] = holm_results_two_way_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Holm test: {str(e)}"
                    context['holm_results_two_way_anova'] = None

            elif post_hoc_test_type_two_way_anova == "fisher_lsd":
                try:
                    fisher_lsd_results_two_way_anova = []
                    alpha = ((100 - alpha_value_two_way_anova) / 100)
                    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
                    df_error = anova_table.loc['Residual', 'df']
                    t_critical = t.ppf(1 - alpha / 2, df_error)

                    levels_factor1 = df['Factor1'].unique()
                    group_means_factor1 = df.groupby('Factor1')['Response'].mean()
                    n_per_group_factor1 = df.groupby('Factor1').size()

                    for i, group1 in enumerate(levels_factor1):
                        for group2 in levels_factor1[i + 1:]:
                            mean_diff = group_means_factor1[group1] - group_means_factor1[group2]
                            pooled_variance = ms_error * (1 / n_per_group_factor1[group1] + 1 / n_per_group_factor1[group2])
                            se = np.sqrt(pooled_variance)
                            t_stat = mean_diff / se
                            p_value = t.sf(np.abs(t_stat), df_error) * 2  # Dos colas
                            reject_null = "Yes" if p_value < alpha else "No"

                            fisher_lsd_results_two_way_anova.append({
                                "Comparison_Type": headers_two_way_anova[0] if headers_two_way_anova else f"C(Factor1)",
                                "Group1": group1,
                                "Group2": group2,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "T_Statistic": f"{t_stat:.4g}",
                                "P_Value": f"{p_value:.4g}",
                                "Reject_Null": reject_null
                            })

                    levels_factor2 = df['Factor2'].unique()
                    group_means_factor2 = df.groupby('Factor2')['Response'].mean()
                    n_per_group_factor2 = df.groupby('Factor2').size()

                    for i, group1 in enumerate(levels_factor2):
                        for group2 in levels_factor2[i + 1:]:
                            mean_diff = group_means_factor2[group1] - group_means_factor2[group2]
                            pooled_variance = ms_error * (1 / n_per_group_factor2[group1] + 1 / n_per_group_factor2[group2])
                            se = np.sqrt(pooled_variance)
                            t_stat = mean_diff / se
                            p_value = t.sf(np.abs(t_stat), df_error) * 2
                            reject_null = "Yes" if p_value < alpha else "No"

                            fisher_lsd_results_two_way_anova.append({
                                "Comparison_Type": headers_two_way_anova[1] if headers_two_way_anova else f"C(Factor2)",
                                "Group1": group1,
                                "Group2": group2,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "T_Statistic": f"{t_stat:.4g}",
                                "P_Value": f"{p_value:.4g}",
                                "Reject_Null": reject_null
                            })

                    df['Combined_Factors'] = df['Factor1'].astype(str) + " x " + df['Factor2'].astype(str)
                    levels_combined = df['Combined_Factors'].unique()
                    group_means_combined = df.groupby('Combined_Factors')['Response'].mean()
                    n_per_group_combined = df.groupby('Combined_Factors').size()

                    for i, group1 in enumerate(levels_combined):
                        for group2 in levels_combined[i + 1:]:
                            mean_diff = group_means_combined[group1] - group_means_combined[group2]
                            pooled_variance = ms_error * (1 / n_per_group_combined[group1] + 1 / n_per_group_combined[group2])
                            se = np.sqrt(pooled_variance)
                            t_stat = mean_diff / se
                            p_value = t.sf(np.abs(t_stat), df_error) * 2  # Dos colas
                            reject_null = "Yes" if p_value < alpha else "No"

                            fisher_lsd_results_two_way_anova.append({
                                "Comparison_Type": "Interaction",
                                "Group1": group1,
                                "Group2": group2,
                                "Mean_Difference": f"{mean_diff:.4g}",
                                "T_Statistic": f"{t_stat:.4g}",
                                "P_Value": f"{p_value:.4g}",
                                "Reject_Null": reject_null
                            })

                    context['fisher_lsd_results_two_way_anova'] = fisher_lsd_results_two_way_anova
                    request.session['fisher_lsd_results_two_way_anova'] = fisher_lsd_results_two_way_anova
                except Exception as e:
                    context['post_hoc_error'] = f"Error during Fisher LSD test: {str(e)}"
                    context['fisher_lsd_results_two_way_anova'] = None

            residuals = model.resid
            shapiro_results_two_way_anova = shapiro(residuals)

            levene_results_two_way_anova = levene(*[
                df[(df["Factor1"] == level1) & (df["Factor2"] == level2)]["Response"]
                for level1 in df["Factor1"].unique()
                for level2 in df["Factor2"].unique()
            ])

            context['shapiro_results_two_way_anova'] = {
                "W_statistic": f"{shapiro_results_two_way_anova.statistic:.4g}",
                "p_value": f"{shapiro_results_two_way_anova.pvalue:.4g}",
                "normality": "Yes" if shapiro_results_two_way_anova.pvalue > ((100 - alpha_value_two_way_anova) / 100) else "No"
            }
            request.session['shapiro_results_two_way_anova'] = shapiro_results_two_way_anova

            if shapiro_results_two_way_anova.pvalue <= ((100 - alpha_value_two_way_anova) / 100):
                warning_3_two_way_anova = "Shapiro-Wilk Test for normality failed. Data may not be normally distributed."
                context['warning_3_two_way_anova'] = warning_3_two_way_anova
                request.session['warning_3_two_way_anova'] = warning_3_two_way_anova

            context['levene_results_two_way_anova'] = {
                "W_statistic": f"{levene_results_two_way_anova.statistic:.4g}",
                "p_value": f"{levene_results_two_way_anova.pvalue:.4g}",
                "homogeneity": "Yes" if levene_results_two_way_anova.pvalue > ((100 - alpha_value_two_way_anova) / 100) else "No"
            }
            request.session['levene_results_two_way_anova'] = levene_results_two_way_anova

            if levene_results_two_way_anova.pvalue <= ((100 - alpha_value_two_way_anova) / 100):
                warning_4_two_way_anova = "Levene Test for homocedasticity failed. Data may not have homogenous variance."
                context['warning_4_two_way_anova'] = warning_4_two_way_anova
                request.session['warning_4_two_way_anova'] = warning_4_two_way_anova

            group_means = df.groupby(['Factor1', 'Factor2'])['Response'].mean()
            group_sems = df.groupby(['Factor1', 'Factor2'])['Response'].sem()
            ci_95 = 1.96 * group_sems

            all_combinations = pd.MultiIndex.from_product(
                [df['Factor1'].unique(), df['Factor2'].unique()],
                names=['Factor1', 'Factor2']
            )
            group_means = group_means.reindex(all_combinations)
            ci_95 = ci_95.reindex(all_combinations)
            group_means_unstacked = group_means.unstack()
            ci_95_unstacked = ci_95.unstack()
            ci_95_aligned = ci_95_unstacked.reindex_like(group_means_unstacked)

            factor1_name = headers_two_way_anova[0] if headers_two_way_anova else "Factor 1"
            factor2_name = headers_two_way_anova[1] if headers_two_way_anova else "Factor 2"
            response_name = headers_two_way_anova[2] if headers_two_way_anova else "Response"

            fig, ax = plt.subplots()
            group_means_unstacked.plot(
                kind='bar',
                yerr=ci_95_aligned,
                ax=ax,
                capsize=5,
                alpha=0.7,
                color=['skyblue', 'orange']
            )
            ax.set_title('Group Means with 95% Confidence Intervals', fontsize=16)
            ax.set_xlabel(factor1_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.legend(title=factor2_name)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['bar_plot_two_way_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['bar_plot_two_way_anova'] = context['bar_plot_two_way_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x="Factor1", y="Response", hue="Factor2", data=df, ax=ax)
            ax.set_title("Boxplot for Two-Factor ANOVA", fontsize=16)
            ax.set_xlabel(factor1_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.legend(title=factor2_name)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['graph_two_way_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['graph_two_way_anova'] = context['graph_two_way_anova']
            buf.close()
            plt.close(fig)

            residuals = model.resid
            fitted_values = model.fittedvalues

            fig, ax = plt.subplots()
            sns.residplot(x=fitted_values, y=residuals, lowess=False, line_kws={"color": "red"}, ax=ax)
            ax.axhline(0, color='black', linestyle='--')
            ax.set_title("Residuals vs Fitted", fontsize=16)
            ax.set_xlabel("Fitted Values", fontsize=14)
            ax.set_ylabel("Residuals", fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['residuals_plot_two_way_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['residuals_plot_two_way_anova'] = context['residuals_plot_two_way_anova']
            buf.close()
            plt.close(fig)

            fig = plt.figure()
            sm.qqplot(residuals, line='45', fit=True, ax=fig.add_subplot(111))
            plt.title("Normal QQ-Plot", fontsize=16)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            context['qq_plot_two_way_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['qq_plot_two_way_anova'] = context['qq_plot_two_way_anova']
            buf.close()
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.pointplot(x="Factor1", y="Response", hue="Factor2", data=df, dodge=True, markers=["o", "s"], ax=ax, capsize=0.1, err_kws={"linewidth": 1})
            ax.set_title("Interaction Plot", fontsize=16)
            ax.set_xlabel(factor1_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.legend(title=factor2_name)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['interaction_plot_two_way_anova'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['interaction_plot_two_way_anova'] = context['interaction_plot_two_way_anova']
            buf.close()
            plt.close(fig)

        except Exception as e:
            context['error_two_way_anova'] = str(e)
            context['anova_results_two_way_anova'] = None
            context['tukey_results_two_way_anova'] = None
            context['bonferroni_results_two_way_anova'] = None
            context['scheffe_results_two_way_anova'] = None
            context['holm_results_two_way_anova'] = None
            context['fisher_lsd_results_two_way_anova'] = None
            context['shapiro_results_two_way_anova'] = None
            context['levene_results_two_way_anova'] = None
            context['graph_two_way_anova'] = None
            context['bar_plot_two_way_anova'] = None
            context['qq_plot_two_way_anova'] = None
            context['residuals_plot_two_way_anova'] = None
            context['interaction_plot_two_way_anova'] = None
            context['explanation_two_way_anova'] = None
            context['warning_1_two_way_anova'] = None
            context['warning_3_two_way_anova'] = None
            context['warning_4_two_way_anova'] = None

#####################################################################################################
    if request.method == "POST" and tab == "manova":

        data_manova = request.POST.get('data_manova')
        use_first_row_as_header_manova = request.POST.get('use_first_row_as_header_manova') == 'on'
        alpha_value_manova = request.POST.get('alpha_value_manova')

        if alpha_value_manova:
            alpha_value_manova = alpha_value_manova.replace(",", ".")

        context['data_manova'] = data_manova
        request.session['data_manova'] = data_manova

        context['use_first_row_as_header_manova'] = 'checked' if use_first_row_as_header_manova else ''
        request.session['use_first_row_as_header_manova'] = use_first_row_as_header_manova

        if alpha_value_manova:
            try:
                alpha_value_manova = float(alpha_value_manova)
                context['alpha_value_manova'] = alpha_value_manova
                request.session['alpha_value_manova'] = alpha_value_manova
            except ValueError:
                alpha_value_manova = None

        if not alpha_value_manova:
            alpha_value_manova = 95
            context['alpha_value_manova'] = alpha_value_manova
            request.session['alpha_value_manova'] = alpha_value_manova

            warning_1_manova = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_manova'] = warning_1_manova
            request.session['warning_1_manova'] = warning_1_manova

        if not data_manova.strip():
            context['error_manova'] = "Please enter data before calculating."
            context['manova_results'] = None
            context['univariate_results_manova'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_manova = data_manova.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_manova.split('\n') if row.strip()]

            if use_first_row_as_header_manova:
                headers_manova = rows[0]
                rows = rows[1:]
            else:
                headers_manova = ["Factor"] + [f"Y{i}" for i in range(1, len(rows[0]))]

            if len(headers_manova) < 3:
                raise ValueError("MANOVA requires one factor column and at least two response variables.")

            factor_name = headers_manova[0]
            response_names = headers_manova[1:]

            factors = []
            response_data = []

            for row in rows:
                if len(row) != len(headers_manova):
                    raise ValueError("All rows must have the same number of columns.")

                factors.append(row[0])
                response_data.append([float(value) for value in row[1:]])

            df = pd.DataFrame(response_data, columns=response_names)
            df[factor_name] = factors

            if df[response_names].isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")

            if df[factor_name].nunique() < 2:
                raise ValueError("MANOVA requires at least two factor levels.")

            if len(response_names) < 2:
                raise ValueError("MANOVA requires at least two response variables.")

            formula = " + ".join(response_names) + f" ~ C({factor_name})"

            manova_model = MANOVA.from_formula(formula, data=df)
            manova_table = manova_model.mv_test()

            stat_table = manova_table.results[f"C({factor_name})"]["stat"]

            alpha = (100 - alpha_value_manova) / 100

            model_info_manova = {
                "method": "MANOVA",
                "factor": factor_name,
                "responses": len(response_names),
                "response_names": ", ".join(response_names),
                "groups": df[factor_name].nunique(),
                "samples": df.shape[0],
                "confidence_level": f"{alpha_value_manova:g}%",
            }

            # ==========================
            # Box's M Test
            # ==========================

            groups = sorted(df[factor_name].unique())

            group_covs = []
            group_ns = []

            for grp in groups:
                subset = df[df[factor_name] == grp][response_names]

                group_covs.append(np.cov(subset.T, bias=False))
                group_ns.append(len(subset))

            p = len(response_names)
            g = len(groups)
            N = sum(group_ns)

            # pooled covariance
            spo_num = np.zeros((p, p))

            for n_i, S_i in zip(group_ns, group_covs):
                spo_num += (n_i - 1) * S_i

            Spooled = spo_num / (N - g)

            M = (N - g) * np.log(np.linalg.det(Spooled))

            for n_i, S_i in zip(group_ns, group_covs):
                M -= (n_i - 1) * np.log(np.linalg.det(S_i))

            correction_factor = (
                (
                    (2 * p**2 + 3 * p - 1)
                    / (6 * (p + 1) * (g - 1))
                )
                *
                (
                    sum(
                        1 / (n_i - 1)
                        for n_i in group_ns
                    )
                    - 1 / (N - g)
                )
            )

            chi_square = M * (1 - correction_factor)

            df_box = ((g - 1) * p * (p + 1)) / 2

            p_box = 1 - chi2.cdf(chi_square, df_box)

            boxm_results = {
                "statistic": f"{chi_square:.5g}",
                "df": f"{df_box:.0f}",
                "p_value": f"{p_box:.5g}",
                "conclusion": (
                    "Assumption satisfied"
                    if p_box > alpha
                    else "Assumption violated"
                ),
            }

            # ==========================
            # Mardia Multivariate Normality
            # ==========================

            X_mardia = df[response_names].values

            n = X_mardia.shape[0]
            p = X_mardia.shape[1]

            mean_vec = np.mean(X_mardia, axis=0)

            S = np.cov(X_mardia.T, bias=False)
            S_inv = np.linalg.inv(S)

            centered = X_mardia - mean_vec

            D = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    D[i, j] = (
                        centered[i]
                        @ S_inv
                        @ centered[j].T
                    )

            # ---------
            # Skewness
            # ---------

            b1p = np.sum(D**3) / (n**2)

            chi_skew = n * b1p / 6

            df_skew = p * (p + 1) * (p + 2) / 6

            p_skew = 1 - chi2.cdf(chi_skew, df_skew)

            # ---------
            # Kurtosis
            # ---------

            diagonal_D = np.diag(D)

            b2p = np.mean(diagonal_D**2)

            expected_b2p = p * (p + 2)

            z_kurt = (
                b2p - expected_b2p
            ) / np.sqrt((8 * p * (p + 2)) / n)

            p_kurt = 2 * (1 - norm.cdf(abs(z_kurt)))

            mardia_results = {
                "skewness_statistic": f"{b1p:.5g}",
                "skewness_p_value": f"{p_skew:.5g}",
                "kurtosis_statistic": f"{b2p:.5g}",
                "kurtosis_z": f"{z_kurt:.5g}",
                "kurtosis_p_value": f"{p_kurt:.5g}",
                "conclusion": (
                    "Assumption satisfied"
                    if p_skew > alpha and p_kurt > alpha
                    else "Assumption violated"
                )
            }

            manova_results = []

            for test_name, row in stat_table.iterrows():
                p_value = row["Pr > F"]

                manova_results.append({
                    "effect": factor_name,
                    "test": test_name,
                    "value": f"{row['Value']:.5g}",
                    "f_value": f"{row['F Value']:.5g}" if pd.notna(row["F Value"]) else "",
                    "num_df": f"{row['Num DF']:.5g}" if pd.notna(row["Num DF"]) else "",
                    "den_df": f"{row['Den DF']:.5g}" if pd.notna(row["Den DF"]) else "",
                    "p_value": f"{p_value:.5g}" if pd.notna(p_value) else "",
                    "reject_null": "Yes" if pd.notna(p_value) and p_value <= alpha else "No",
                })

            univariate_results_manova = []

            for response in response_names:
                model = sm.OLS.from_formula(f"{response} ~ C({factor_name})", data=df).fit()
                anova_table = anova_lm(model, typ=2)

                for source, row in anova_table.iterrows():
                    if source == "Residual":
                        continue

                    p_value = row["PR(>F)"]

                    univariate_results_manova.append({
                        "response": response,
                        "source": factor_name,
                        "sum_sq": f"{row['sum_sq']:.5g}",
                        "df": f"{row['df']:.5g}",
                        "f_statistic": f"{row['F']:.5g}" if pd.notna(row["F"]) else "",
                        "p_value": f"{p_value:.5g}" if pd.notna(p_value) else "",
                        "reject_null": "Yes" if pd.notna(p_value) and p_value <= alpha else "No",
                    })

            boxplots_manova = []
            for response in response_names:
                fig = px.box(
                    df,
                    x=factor_name,
                    y=response,
                    points="all",
                    title=f"{response} by {factor_name}",
                    labels={
                        factor_name: factor_name,
                        response: response,
                    }
                )

                fig.update_layout(
                    title_x=0.5,
                    height=500
                )

                boxplots_manova.append({
                    "response": response,
                    "plot": fig.to_html(full_html=False),
                })

            # ==========================
            # Post-hoc Tukey
            # ==========================

            posthoc_manova = []

            for response in response_names:

                tukey = pairwise_tukeyhsd(
                    endog=df[response],
                    groups=df[factor_name],
                    alpha=alpha
                )

                tukey_table = []

                for row in tukey.summary().data[1:]:

                    tukey_table.append({
                        "group1": str(row[0]),
                        "group2": str(row[1]),
                        "mean_diff": f"{float(row[2]):.5g}",
                        "p_value": f"{float(row[3]):.5g}",
                        "lower": f"{float(row[4]):.5g}",
                        "upper": f"{float(row[5]):.5g}",
                        "significant": "Yes" if bool(row[6]) else "No",
                    })

                posthoc_manova.append({
                    "response": response,
                    "results": tukey_table,
                })

            # ==========================
            # Canonical Discriminant Analysis
            # ==========================

            cda_results = None
            cda_scores_plot = None
            cda_centroids = None

            X_cda = df[response_names].values
            y_cda = df[factor_name].values

            max_cda_components = min(
                len(np.unique(y_cda)) - 1,
                X_cda.shape[1]
            )

            if max_cda_components >= 1:

                lda_cda = LinearDiscriminantAnalysis(
                    n_components=max_cda_components
                )

                scores_cda = lda_cda.fit_transform(
                    X_cda,
                    y_cda
                )

                cda_columns = [
                    f"CD{i + 1}"
                    for i in range(scores_cda.shape[1])
                ]

                cda_df = pd.DataFrame(
                    scores_cda,
                    columns=cda_columns
                )

                cda_df[factor_name] = y_cda

                explained = lda_cda.explained_variance_ratio_

                cda_results = []

                for i, value in enumerate(explained):
                    canonical_correlation = np.sqrt(value)

                    cda_results.append({
                        "function": f"CD{i + 1}",
                        "explained": f"{value * 100:.2f}",
                        "canonical_correlation": f"{canonical_correlation:.4f}",
                    })

                cda_centroids = []

                centroid_df = cda_df.groupby(factor_name)[cda_columns].mean()

                for group_name, row in centroid_df.iterrows():
                    item = {
                        "group": group_name,
                    }

                    for col in cda_columns:
                        item[col.lower()] = f"{row[col]:.4f}"

                    cda_centroids.append(item)

                if scores_cda.shape[1] >= 2:
                    fig = px.scatter(
                        cda_df,
                        x="CD1",
                        y="CD2",
                        color=factor_name,
                        title="Canonical Discriminant Analysis Scores Plot",
                        labels={
                            "CD1": f"CD1 ({explained[0] * 100:.2f}%)",
                            "CD2": f"CD2 ({explained[1] * 100:.2f}%)",
                            factor_name: factor_name,
                        }
                    )

                else:
                    fig = px.scatter(
                        cda_df,
                        x="CD1",
                        y=[0] * len(cda_df),
                        color=factor_name,
                        title="Canonical Discriminant Analysis Scores Plot",
                        labels={
                            "CD1": f"CD1 ({explained[0] * 100:.2f}%)",
                            "y": "",
                            factor_name: factor_name,
                        }
                    )

                fig.update_layout(
                    title_x=0.5,
                    height=550
                )

                cda_scores_plot = fig.to_html(full_html=False)

            context['manova_results'] = manova_results
            context['univariate_results_manova'] = univariate_results_manova
            context['error_manova'] = None
            context['model_info_manova'] = model_info_manova
            context['boxplots_manova'] = boxplots_manova
            context['boxm_results'] = boxm_results
            context['mardia_results'] = mardia_results
            context['posthoc_manova'] = posthoc_manova
            context['cda_results'] = cda_results
            context['cda_scores_plot'] = cda_scores_plot
            context['cda_centroids'] = cda_centroids

            request.session['manova_results'] = manova_results
            request.session['univariate_results_manova'] = univariate_results_manova
            request.session['error_manova'] = None
            request.session['model_info_manova'] = model_info_manova
            request.session['boxplots_manova'] = boxplots_manova
            request.session['boxm_results'] = boxm_results
            request.session['mardia_results'] = mardia_results
            request.session['posthoc_manova'] = posthoc_manova
            request.session['cda_results'] = cda_results
            request.session['cda_scores_plot'] = cda_scores_plot
            request.session['cda_centroids'] = cda_centroids

        except Exception as e:
            context['error_manova'] = str(e)
            context['manova_results'] = None
            context['univariate_results_manova'] = None
            context['model_info_manova'] = None
            context['boxplots_manova'] = None
            context['boxm_results'] = None
            context['mardia_results'] = None
            context['posthoc_manova'] = None
            context['cda_results'] = None
            context['cda_scores_plot'] = None
            context['cda_centroids'] = None

            request.session['error_manova'] = str(e)
            request.session['manova_results'] = None
            request.session['univariate_results_manova'] = None
            request.session['model_info_manova'] = None
            request.session['boxplots_manova'] = None
            request.session['boxm_results'] = None
            request.session['mardia_results'] = None
            request.session['posthoc_manova'] = None
            request.session['cda_results'] = None
            request.session['cda_scores_plot'] = None
            request.session['cda_centroids'] = None

#####################################################################################################
    if request.method == "POST" and tab == "non_parametrics" and subtab == "kruskal_wallis":
        data_kruskal_wallis = request.POST.get('data_kruskal_wallis')
        use_first_row_as_header_kruskal_wallis = request.POST.get('use_first_row_as_header_kruskal_wallis') == 'on'
        alpha_value_kruskal_wallis = request.POST.get('alpha_value_kruskal_wallis')
        post_hoc_test_type_kruskal_wallis = request.POST.get('post_hoc_test_type_kruskal_wallis')

        context['use_first_row_as_header_kruskal_wallis'] = 'checked' if use_first_row_as_header_kruskal_wallis else ''
        request.session['use_first_row_as_header_kruskal_wallis'] = use_first_row_as_header_kruskal_wallis
        context['data_kruskal_wallis'] = data_kruskal_wallis
        request.session['data_kruskal_wallis'] = data_kruskal_wallis
        context['post_hoc_test_type_kruskal_wallis'] = post_hoc_test_type_kruskal_wallis
        request.session['post_hoc_test_type_kruskal_wallis'] = post_hoc_test_type_kruskal_wallis

        if alpha_value_kruskal_wallis:
            try:
                alpha_value_kruskal_wallis = float(alpha_value_kruskal_wallis)
                context['alpha_value_kruskal_wallis'] = alpha_value_kruskal_wallis
                request.session['alpha_value_kruskal_wallis'] = alpha_value_kruskal_wallis
            except ValueError:
                alpha_value_kruskal_wallis = None

        if not alpha_value_kruskal_wallis:
            alpha_value_kruskal_wallis = 95
            context['alpha_value_kruskal_wallis'] = alpha_value_kruskal_wallis
            request.session['alpha_value_kruskal_wallis'] = alpha_value_kruskal_wallis
            warning_1_kruskal_wallis = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_kruskal_wallis'] = warning_1_kruskal_wallis
            request.session['warning_1_kruskal_wallis'] = warning_1_kruskal_wallis

        if not data_kruskal_wallis.strip():
            context['error_kruskal_wallis'] = "Please enter data before calculating."
            context['results_kruskal_wallis'] = None
            context['results_dunn_kruskal_wallis'] = None
            context['graph_kruskal_wallis'] = None
            context['explanation_kruskal_wallis'] = None
            context['warning_1_kruskal_wallis'] = None
            context['warning_2_kruskal_wallis'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_kruskal_wallis = data_kruskal_wallis.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_kruskal_wallis.split('\n')]
            if use_first_row_as_header_kruskal_wallis:
                headers_kruskal_wallis = rows[0]
                rows = rows[1:]
            else:
                headers_kruskal_wallis = ["Factor", "Response"]

            factors, responses = zip(*rows)
            factors = list(factors)
            responses = [float(r) for r in responses]

            df = pd.DataFrame({"Factor": factors, "Response": responses})

            if df.isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")
            if df["Response"].dtype not in [float, int]:
                raise ValueError("Response column contains non-numeric values.")

            groups = [df[df["Factor"] == level]["Response"] for level in df["Factor"].unique()]
            kw_result = kruskal(*groups)

            context['results_kruskal_wallis'] = {
                "H_statistic": f"{kw_result.statistic:.4g}",
                "p_value": f"{kw_result.pvalue:.4g}",
                "significant": "Yes" if kw_result.pvalue < ((100 - alpha_value_kruskal_wallis) / 100) else "No"
            }

            request.session['results_kruskal_wallis'] = context['results_kruskal_wallis']
            explanation_kruskal_wallis = "Null Hypothesis (H₀): There are no differences between the population medians of the groups. Alternative Hypothesis (H₁): At least one group has a different population median."
            context['explanation_kruskal_wallis'] = explanation_kruskal_wallis
            request.session['explanation_kruskal_wallis'] = explanation_kruskal_wallis

            if kw_result.pvalue >= ((100 - alpha_value_kruskal_wallis) / 100):
                warning_2_kruskal_wallis = "p-value greater than alpha; post-hoc test omitted."
                context['warning_2_kruskal_wallis'] = warning_2_kruskal_wallis
                request.session['warning_2_kruskal_wallis'] = warning_2_kruskal_wallis

            if post_hoc_test_type_kruskal_wallis == "dunn" and kw_result.pvalue < ((100 - alpha_value_kruskal_wallis) / 100):
                dunn_results = sp.posthoc_dunn(df, val_col="Response", group_col="Factor", p_adjust="bonferroni")
                dunn_results_table = dunn_results.reset_index().melt(id_vars='index', var_name='Group2', value_name='P_Value')
                dunn_results_table.columns = ["Group1", "Group2", "P_Value"]
                dunn_results_table = dunn_results_table[~(dunn_results_table['Group1'] == dunn_results_table['Group2'])]
                dunn_results_table["P_Value"] = dunn_results_table["P_Value"].apply(lambda p: f"{p:.4g}")
                dunn_results_table["Reject_Null"] = dunn_results_table["P_Value"].apply(lambda p: "Yes" if float(p) < ((100 - alpha_value_kruskal_wallis) / 100) else "No")
                context['results_dunn_kruskal_wallis'] = dunn_results_table.to_dict(orient="records")
                request.session['results_dunn_kruskal_wallis'] = context['results_dunn_kruskal_wallis']

            factor_name = headers_kruskal_wallis[0] if headers_kruskal_wallis else "Factor"
            response_name = headers_kruskal_wallis[1] if headers_kruskal_wallis else "Response"

            fig, ax = plt.subplots()
            sns.boxplot(x="Factor", y="Response", data=df, ax=ax)
            ax.set_title("Boxplot", fontsize=16)
            ax.set_xlabel(factor_name, fontsize=14)
            ax.set_ylabel(response_name, fontsize=14)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            context['graph_kruskal_wallis'] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            request.session['graph_kruskal_wallis'] = context['graph_kruskal_wallis']
            buf.close()
            plt.close(fig)

        except Exception as e:
            context['error_kruskal_wallis'] = str(e)
            context['results_kruskal_wallis'] = None
            context['graph_kruskal_wallis'] = None
            context['results_dunn_kruskal_wallis'] = None
            context['explanation_kruskal_wallis'] = None
            context['warning_1_kruskal_wallis'] = None
            context['warning_2_kruskal_wallis'] = None

#####################################################################################################
    if request.method == "POST" and tab == "non_parametrics" and subtab == "dba_friedman":
        data_dba_friedman = request.POST.get('data_dba_friedman')
        use_first_row_as_header_dba_friedman = request.POST.get('use_first_row_as_header_dba_friedman') == 'on'
        alpha_value_dba_friedman = request.POST.get('alpha_value_dba_friedman')
        post_hoc_test_type_dba_friedman = request.POST.get('post_hoc_test_type_dba_friedman')

        context['use_first_row_as_header_dba_friedman'] = 'checked' if use_first_row_as_header_dba_friedman else ''
        request.session['use_first_row_as_header_dba_friedman'] = use_first_row_as_header_dba_friedman
        context['data_dba_friedman'] = data_dba_friedman
        request.session['data_dba_friedman'] = data_dba_friedman
        context['post_hoc_test_type_dba_friedman'] = post_hoc_test_type_dba_friedman
        request.session['post_hoc_test_type_dba_friedman'] = post_hoc_test_type_dba_friedman

        if alpha_value_dba_friedman:
            try:
                alpha_value_dba_friedman = float(alpha_value_dba_friedman)
                context['alpha_value_dba_friedman'] = alpha_value_dba_friedman
                request.session['alpha_value_dba_friedman'] = alpha_value_dba_friedman
            except ValueError:
                alpha_value_dba_friedman = None
        
        if not alpha_value_dba_friedman:
            alpha_value_dba_friedman = 95
            context['alpha_value_dba_friedman'] = alpha_value_dba_friedman
            request.session['alpha_value_dba_friedman'] = alpha_value_dba_friedman
            warning_1_dba_friedman = "Confidence Level not defined, defaulting to 95%."
            context['warning_1_dba_friedman'] = warning_1_dba_friedman
            request.session['warning_1_dba_friedman'] = warning_1_dba_friedman

        if not data_dba_friedman.strip():
            context['error_dba_friedman'] = "Please enter data before calculating."
            context['results_dba_friedman'] = None
            context['results_nemenyi_dba_friedman'] = None
            context['graph_dba_friedman'] = None
            context['explanation_dba_friedman'] = None
            context['warning_1_dba_friedman'] = None
            context['warning_2_dba_friedman'] = None
            return render(request, "anova/anova.html", context)

        try:
            data_dba_friedman = data_dba_friedman.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_dba_friedman.split('\n')]

            if use_first_row_as_header_dba_friedman:
                headers_dba_friedman = rows[0]
                rows = rows[1:]
            else:
                headers_dba_friedman = ["Block", "Factor", "Response"]

            blocks, factors, responses = zip(*rows)
            blocks = list(blocks)
            factors = list(factors)
            responses = [float(r) for r in responses]

            df = pd.DataFrame({"Block": blocks, "Factor": factors, "Response": responses})

            df = df.groupby(["Block", "Factor"]).agg({"Response": "mean"}).reset_index()

            if df.isnull().values.any():
                raise ValueError("Data contains missing values. Please provide complete data.")

            pivot_df = df.pivot(index="Block", columns="Factor", values="Response")

            if pivot_df.isnull().values.any():
                raise ValueError("Some blocks do not contain all factor levels. Ensure data is complete.")

            friedman_result = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
            reject_null_dba_friedman = "Yes" if friedman_result.pvalue <= (1 - ((alpha_value_dba_friedman)/100)) else "No"

            context['results_dba_friedman'] = {
                "Chi_square": f"{friedman_result.statistic:.4g}",
                "p_value": f"{friedman_result.pvalue:.4g}",
                "Reject_Null": reject_null_dba_friedman
            }

            request.session['results_dba_friedman'] = context['results_dba_friedman']
            explanation_friedman = "Null Hypothesis (H₀): The distributions of all groups are the same. Alternative Hypothesis (H₁): At least one group differs in its distribution."
            context['explanation_dba_friedman'] = explanation_friedman
            request.session['explanation_dba_friedman'] = explanation_friedman

            if reject_null_dba_friedman == "No":
                warning_2_dba_friedman = "p-value greater than alpha; post-hoc test omitted."
                context['warning_2_dba_friedman'] = warning_2_dba_friedman
                request.session['warning_2_dba_friedman'] = warning_2_dba_friedman

            nemenyi_result = None
            
            if post_hoc_test_type_dba_friedman == "nemenyi" and reject_null_dba_friedman == "Yes":
                
                nemenyi_result = posthoc_nemenyi_friedman(pivot_df.values)
                nemenyi_matrix = nemenyi_result.round(4).values.tolist()
                nemenyi_headers = list(pivot_df.columns)

                nemenyi_results_list = []
                for i, group1 in enumerate(nemenyi_headers):
                    for j, group2 in enumerate(nemenyi_headers):
                        if i < j:
                            nemenyi_results_list.append({
                                "Group1": group1,
                                "Group2": group2,
                                "P_Value": nemenyi_matrix[i][j],
                                "Reject_Null": "Yes" if nemenyi_matrix[i][j] <= (1 - ((alpha_value_dba_friedman)/100)) else "No"
                            })

                context['results_nemenyi_dba_friedman'] = nemenyi_results_list
                request.session['results_nemenyi_dba_friedman'] = nemenyi_results_list

        except Exception as e:
            context['error_dba_friedman'] = str(e)
            context['results_dba_friedman'] = None
            context['results_nemenyi_dba_friedman'] = None
            context['graph_dba_friedman'] = None
            context['explanation_dba_friedman'] = None
            context['warning_1_dba_friedman'] = None
            context['warning_2_dba_friedman'] = None
#####################################################################################################

    return render(request, "anova/anova.html", context)