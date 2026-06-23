from django.shortcuts import render
import numpy as np
from scipy.stats import norm, t, chi2, f, mannwhitneyu, wilcoxon, shapiro, kstest
import matplotlib.pyplot as plt
import io
import base64
import re
from io import BytesIO

def inference(request):
    tab = request.GET.get('tab', 'confidence')
    subtab = request.GET.get('subtab', None)
    subsubtab = request.GET.get('subsubtab', None)

    context = {
        'segment': 'inference',
        'active_tab': tab,
        'active_subtab': subtab,
        'active_subsubtab': subsubtab,
        'data_mean_1': request.session.get('data_mean_1', ""),
        'results_mean_1': request.session.get('results_mean_1', None),
        'graph_mean_1': request.session.get('graph_mean_1', None),
        'headers_mean_1': request.session.get('headers_mean_1', None),
        'use_first_row_as_header_mean_1': 'checked' if request.session.get('use_first_row_as_header_mean_1', False) else '',
        'data_mean_2': request.session.get('data_mean_2', ""),
        'results_mean_2': request.session.get('results_mean_2', None),
        'graph_mean_2': request.session.get('graph_mean_2', None),
        'headers_mean_2': request.session.get('headers_mean_2', None),
        'use_first_row_as_header_mean_2': 'checked' if request.session.get('use_first_row_as_header_mean_2', False) else '',
        'data_var_1': request.session.get('data_var_1', ""),
        'results_var_1': request.session.get('results_var_1', None),
        'graph_var_1': request.session.get('graph_var_1', None),
        'headers_var_1': request.session.get('headers_var_1', None),
        'use_first_row_as_header_var_1': 'checked' if request.session.get('use_first_row_as_header_var_1', False) else '',
        'data_var_2': request.session.get('data_var_2', ""),
        'results_var_2': request.session.get('results_var_2', None),
        'graph_var_2': request.session.get('graph_var_2', None),
        'headers_var_2': request.session.get('headers_var_2', None),
        'use_first_row_as_header_var_2': 'checked' if request.session.get('use_first_row_as_header_var_2', False) else '',
        'data_prop': request.session.get('data_prop', ""),
        'results_prop': request.session.get('results_prop', None),
        'graph_prop': request.session.get('graph_prop', None),
        'headers_prop': request.session.get('headers_prop', None),
        'use_first_row_as_header_prop': 'checked' if request.session.get('use_first_row_as_header_prop', False) else '',
        'data_mean_diff_1': request.session.get('data_mean_diff_1', ""),
        'results_mean_diff_1': request.session.get('results_mean_diff_1', None),
        'graph_mean_diff_1': request.session.get('graph_mean_diff_1', None),
        'headers_mean_diff_1': request.session.get('headers_mean_diff_1', None),
        'use_first_row_as_header_mean_diff_1': 'checked' if request.session.get('use_first_row_as_header_mean_diff_1', False) else '',
        'alpha_value_mean_diff_1': request.session.get('alpha_value_mean_diff_1', None),
        'data_mean_diff_2': request.session.get('data_mean_diff_2', ""),
        'results_mean_diff_2': request.session.get('results_mean_diff_2', None),
        'graph_mean_diff_2': request.session.get('graph_mean_diff_2', None),
        'headers_mean_diff_2': request.session.get('headers_mean_diff_2', None),
        'use_first_row_as_header_mean_diff_2': 'checked' if request.session.get('use_first_row_as_header_mean_diff_2', False) else '',
        'alpha_value_mean_diff_2': request.session.get('alpha_value_mean_diff_2', None),
        'data_mean_diff_3': request.session.get('data_mean_diff_3', ""),
        'results_mean_diff_3': request.session.get('results_mean_diff_3', None),
        'graph_mean_diff_3': request.session.get('graph_mean_diff_3', None),
        'headers_mean_diff_3': request.session.get('headers_mean_diff_3', None),
        'use_first_row_as_header_mean_diff_3': 'checked' if request.session.get('use_first_row_as_header_mean_diff_3', False) else '',
        'alpha_value_mean_diff_3': request.session.get('alpha_value_mean_diff_3', None),
        'data_mean_diff_4': request.session.get('data_mean_diff_4', ""),
        'results_mean_diff_4': request.session.get('results_mean_diff_4', None),
        'graph_mean_diff_4': request.session.get('graph_mean_diff_4', None),
        'headers_mean_diff_4': request.session.get('headers_mean_diff_4', None),
        'use_first_row_as_header_mean_diff_4': 'checked' if request.session.get('use_first_row_as_header_mean_diff_4', False) else '',
        'alpha_value_mean_diff_4': request.session.get('alpha_value_mean_diff_4', None),
        'data_var_quo': request.session.get('data_var_quo', ""),
        'results_var_quo': request.session.get('results_var_quo', None),
        'graph_var_quo': request.session.get('graph_var_quo', None),
        'headers_var_quo': request.session.get('headers_var_quo', None),
        'use_first_row_as_header_var_quo': 'checked' if request.session.get('use_first_row_as_header_var_quo', False) else '',
        'alpha_value_var_quo': request.session.get('alpha_value_var_quo', None),
        'data_prop_diff': request.session.get('data_prop_diff', ""),
        'results_prop_diff': request.session.get('results_prop_diff', None),
        'graph_prop_diff': request.session.get('graph_prop_diff', None),
        'headers_prop_diff': request.session.get('headers_prop_diff', None),
        'use_first_row_as_header_prop_diff': 'checked' if request.session.get('use_first_row_as_header_prop_diff', False) else '',
        'alpha_value_prop_diff': request.session.get('alpha_value_prop_diff', None),
        'data_mean_test_1': request.session.get('data_mean_test_1', ""),
        'results_mean_test_1': request.session.get('results_mean_test_1', None),
        'graph_mean_test_1': request.session.get('graph_mean_test_1', None),
        'headers_mean_test_1': request.session.get('headers_mean_test_1', None),
        'use_first_row_as_header_mean_test_1': 'checked' if request.session.get('use_first_row_as_header_mean_test_1', False) else '',
        'test_type_mean_test_1': request.session.get('test_type_mean_test_1', None),
        'explanation_mean_test_1': request.session.get('explanation_mean_test_1', None),
        'data_mean_test_2': request.session.get('data_mean_test_2', ""),
        'results_mean_test_2': request.session.get('results_mean_test_2', None),
        'graph_mean_test_2': request.session.get('graph_mean_test_2', None),
        'headers_mean_test_2': request.session.get('headers_mean_test_2', None),
        'use_first_row_as_header_mean_test_2': 'checked' if request.session.get('use_first_row_as_header_mean_test_2', False) else '',
        'test_type_mean_test_2': request.session.get('test_type_mean_test_2', None),
        'explanation_mean_test_2': request.session.get('explanation_mean_test_2', None),
        'data_var_test': request.session.get('data_var_test', ""),
        'results_var_test': request.session.get('results_var_test', None),
        'graph_var_test': request.session.get('graph_var_test', None),
        'headers_var_test': request.session.get('headers_var_test', None),
        'use_first_row_as_header_var_test': 'checked' if request.session.get('use_first_row_as_header_var_test', False) else '',
        'test_type_var_test': request.session.get('test_type_var_test', None),
        'explanation_var_test': request.session.get('explanation_var_test', None),
        'data_mean_diff_test_1': request.session.get('data_mean_diff_test_1', ""),
        'results_mean_diff_test_1': request.session.get('results_mean_diff_test_1', None),
        'graph_mean_diff_test_1': request.session.get('graph_mean_diff_test_1', None),
        'headers_mean_diff_test_1': request.session.get('headers_mean_diff_test_1', None),
        'use_first_row_as_header_mean_diff_test_1': 'checked' if request.session.get('use_first_row_as_header_mean_diff_test_1', False) else '',
        'test_type_mean_diff_test_1': request.session.get('test_type_mean_diff_test_1', None),
        'explanation_mean_diff_test_1': request.session.get('explanation_mean_diff_test_1', None),
        'alpha_value_mean_diff_test_1': request.session.get('alpha_value_mean_diff_test_1', None),
        'data_mean_diff_test_2': request.session.get('data_mean_diff_test_2', ""),
        'results_mean_diff_test_2': request.session.get('results_mean_diff_test_2', None),
        'graph_mean_diff_test_2': request.session.get('graph_mean_diff_test_2', None),
        'headers_mean_diff_test_2': request.session.get('headers_mean_diff_test_2', None),
        'use_first_row_as_header_mean_diff_test_2': 'checked' if request.session.get('use_first_row_as_header_mean_diff_test_2', False) else '',
        'test_type_mean_diff_test_2': request.session.get('test_type_mean_diff_test_2', None),
        'explanation_mean_diff_test_2': request.session.get('explanation_mean_diff_test_2', None),
        'alpha_value_mean_diff_test_2': request.session.get('alpha_value_mean_diff_test_2', None),
        'data_mean_diff_test_3': request.session.get('data_mean_diff_test_3', ""),
        'results_mean_diff_test_3': request.session.get('results_mean_diff_test_3', None),
        'graph_mean_diff_test_3': request.session.get('graph_mean_diff_test_3', None),
        'headers_mean_diff_test_3': request.session.get('headers_mean_diff_test_3', None),
        'use_first_row_as_header_mean_diff_test_3': 'checked' if request.session.get('use_first_row_as_header_mean_diff_test_3', False) else '',
        'test_type_mean_diff_test_3': request.session.get('test_type_mean_diff_test_3', None),
        'explanation_mean_diff_test_3': request.session.get('explanation_mean_diff_test_3', None),
        'alpha_value_mean_diff_test_3': request.session.get('alpha_value_mean_diff_test_3', None),
        'data_mean_diff_test_4': request.session.get('data_mean_diff_test_4', ""),
        'results_mean_diff_test_4': request.session.get('results_mean_diff_test_4', None),
        'graph_mean_diff_test_4': request.session.get('graph_mean_diff_test_4', None),
        'headers_mean_diff_test_4': request.session.get('headers_mean_diff_test_4', None),
        'use_first_row_as_header_mean_diff_test_4': 'checked' if request.session.get('use_first_row_as_header_mean_diff_test_4', False) else '',
        'test_type_mean_diff_test_4': request.session.get('test_type_mean_diff_test_4', None),
        'explanation_mean_diff_test_4': request.session.get('explanation_mean_diff_test_4', None),
        'alpha_value_mean_diff_test_4': request.session.get('alpha_value_mean_diff_test_4', None),
        'data_prop_test': request.session.get('data_prop_test', ""),
        'results_prop_test': request.session.get('results_prop_test', None),
        'graph_prop_test': request.session.get('graph_prop_test', None),
        'headers_prop_test': request.session.get('headers_prop_test', None),
        'use_first_row_as_header_prop_test': 'checked' if request.session.get('use_first_row_as_header_prop_test', False) else '',
        'test_type_prop_test': request.session.get('test_type_prop_test', None),
        'explanation_prop_test': request.session.get('explanation_prop_test', None),
        'data_prop_diff_test': request.session.get('data_prop_diff_test', ""),
        'results_prop_diff_test': request.session.get('results_prop_diff_test', None),
        'graph_prop_diff_test': request.session.get('graph_prop_diff_test', None),
        'headers_prop_diff_test': request.session.get('headers_prop_diff_test', None),
        'use_first_row_as_header_prop_diff_test': 'checked' if request.session.get('use_first_row_as_header_prop_diff_test', False) else '',
        'test_type_prop_diff_test': request.session.get('test_type_prop_diff_test', None),
        'explanation_prop_diff_test': request.session.get('explanation_prop_diff_test', None),
        'alpha_value_prop_diff_test': request.session.get('alpha_value_prop_diff_test', None),
        'data_var_quo_test': request.session.get('data_var_quo_test', ""),
        'results_var_quo_test': request.session.get('results_var_quo_test', None),
        'graph_var_quo_test': request.session.get('graph_var_quo_test', None),
        'headers_var_quo_test': request.session.get('headers_var_quo_test', None),
        'use_first_row_as_header_var_quo_test': 'checked' if request.session.get('use_first_row_as_header_var_quo_test', False) else '',
        'test_type_var_quo_test': request.session.get('test_type_var_quo_test', None),
        'explanation_var_quo_test': request.session.get('explanation_var_quo_test', None),
        'alpha_value_var_quo_test': request.session.get('alpha_value_var_quo_test', None),
        'data_U_test': request.session.get('data_U_test', ""),
        'results_U_test': request.session.get('results_U_test', None),
        'headers_U_test': request.session.get('headers_U_test', None),
        'use_first_row_as_header_U_test': 'checked' if request.session.get('use_first_row_as_header_U_test', False) else '',
        'test_type_U_test': request.session.get('test_type_U_test', None),
        'explanation_U_test': request.session.get('explanation_U_test', None),
        'alpha_value_U_test': request.session.get('alpha_value_U_test', None),
        'data_U_signed_test': request.session.get('data_U_signed_test', ""),
        'results_U_signed_test': request.session.get('results_U_signed_test', None),
        'headers_U_signed_test': request.session.get('headers_U_signed_test', None),
        'use_first_row_as_header_U_signed_test': 'checked' if request.session.get('use_first_row_as_header_U_signed_test', False) else '',
        'test_type_U_signed_test': request.session.get('test_type_U_signed_test', None),
        'explanation_U_signed_test': request.session.get('explanation_U_signed_test', None),
        'alpha_value_U_signed_test': request.session.get('alpha_value_U_signed_test', None),
        'data_shap_test': request.session.get('data_shap_test', ""),
        'results_shap_test': request.session.get('results_shap_test', None),
        'headers_shap_test': request.session.get('headers_shap_test', None),
        'use_first_row_as_header_shap_test': 'checked' if request.session.get('use_first_row_as_header_shap_test', False) else '',
        'explanation_shap_test': request.session.get('explanation_shap_test', None),
        'alpha_value_shap_test': request.session.get('alpha_value_shap_test', None),
        'data_kol_test': request.session.get('data_kol_test', ""),
        'results_kol_test': request.session.get('results_kol_test', None),
        'headers_kol_test': request.session.get('headers_kol_test', None),
        'use_first_row_as_header_kol_test': 'checked' if request.session.get('use_first_row_as_header_kol_test', False) else '',
        'explanation_kol_test': request.session.get('explanation_kol_test', None),
        'alpha_value_kol_test': request.session.get('alpha_value_kol_test', None),
    }

    if request.method == "POST" and request.POST.get("clear_mean_1") == "true":
        if 'data_mean_1' in request.session:
            del request.session['data_mean_1']
        if 'results_mean_1' in request.session:
            del request.session['results_mean_1']
        if 'graph_mean_1' in request.session:
            del request.session['graph_mean_1']
        if 'headers_mean_1' in request.session:
            del request.session['headers_mean_1']
        if 'use_first_row_as_header_mean_1' in request.session:
            request.session.pop('use_first_row_as_header_mean_1', False)
        context['data_mean_1'] = ""
        context['results_mean_1'] = None
        context['graph_mean_1'] = None
        context['use_first_row_as_header_mean_1'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_2") == "true":
        if 'data_mean_2' in request.session:
            del request.session['data_mean_2']
        if 'results_mean_2' in request.session:
            del request.session['results_mean_2']
        if 'graph_mean_2' in request.session:
            del request.session['graph_mean_2']
        if 'headers_mean_2' in request.session:
            del request.session['headers_mean_2']
        if 'use_first_row_as_header_mean_2' in request.session:
            request.session.pop('use_first_row_as_header_mean_2', False)
        context['data_mean_2'] = ""
        context['results_mean_2'] = None
        context['graph_mean_2'] = None
        context['use_first_row_as_header_mean_2'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_var_1") == "true":
        if 'data_var_1' in request.session:
            del request.session['data_var_1']
        if 'results_var_1' in request.session:
            del request.session['results_var_1']
        if 'graph_var_1' in request.session:
            del request.session['graph_var_1']
        if 'headers_var_1' in request.session:
            del request.session['headers_var_1']
        if 'use_first_row_as_header_var_1' in request.session:
            request.session.pop('use_first_row_as_header_var_1', False)
        context['data_var_1'] = ""
        context['results_var_1'] = None
        context['graph_var_1'] = None
        context['use_first_row_as_header_var_1'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_var_2") == "true":
        if 'data_var_2' in request.session:
            del request.session['data_var_2']
        if 'results_var_2' in request.session:
            del request.session['results_var_2']
        if 'graph_var_2' in request.session:
            del request.session['graph_var_2']
        if 'headers_var_2' in request.session:
            del request.session['headers_var_2']
        if 'use_first_row_as_header_var_2' in request.session:
            request.session.pop('use_first_row_as_header_var_2', False)
        context['data_var_2'] = ""
        context['results_var_2'] = None
        context['graph_var_2'] = None
        context['use_first_row_as_header_var_2'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_prop") == "true":
        if 'data_prop' in request.session:
            del request.session['data_prop']
        if 'results_prop' in request.session:
            del request.session['results_prop']
        if 'graph_prop' in request.session:
            del request.session['graph_prop']
        if 'headers_prop' in request.session:
            del request.session['headers_prop']
        if 'use_first_row_as_header_prop' in request.session:
            request.session.pop('use_first_row_as_header_prop', False)
        context['data_prop'] = ""
        context['results_prop'] = None
        context['graph_prop'] = None
        context['use_first_row_as_header_prop'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_1") == "true":
        if 'data_mean_diff_1' in request.session:
            del request.session['data_mean_diff_1']
        if 'results_mean_diff_1' in request.session:
            del request.session['results_mean_diff_1']
        if 'graph_mean_diff_1' in request.session:
            del request.session['graph_mean_diff_1']
        if 'headers_mean_diff_1' in request.session:
            del request.session['headers_mean_diff_1']
        if 'use_first_row_as_header_mean_diff_1' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_1', False)
        if 'alpha_value_mean_diff_1' in request.session:
            request.session.pop('alpha_value_mean_diff_1', False)
        context['data_mean_diff_1'] = ""
        context['results_mean_diff_1'] = None
        context['graph_mean_diff_1'] = None
        context['use_first_row_as_header_mean_diff_1'] = False
        context['alpha_value_mean_diff_1'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_2") == "true":
        if 'data_mean_diff_2' in request.session:
            del request.session['data_mean_diff_2']
        if 'results_mean_diff_2' in request.session:
            del request.session['results_mean_diff_2']
        if 'graph_mean_diff_2' in request.session:
            del request.session['graph_mean_diff_2']
        if 'headers_mean_diff_2' in request.session:
            del request.session['headers_mean_diff_2']
        if 'use_first_row_as_header_mean_diff_2' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_2', False)
        if 'alpha_value_mean_diff_2' in request.session:
            request.session.pop('alpha_value_mean_diff_2', False)
        context['data_mean_diff_2'] = ""
        context['results_mean_diff_2'] = None
        context['graph_mean_diff_2'] = None
        context['use_first_row_as_header_mean_diff_2'] = False
        context['alpha_value_mean_diff_2'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_3") == "true":
        if 'data_mean_diff_3' in request.session:
            del request.session['data_mean_diff_3']
        if 'results_mean_diff_3' in request.session:
            del request.session['results_mean_diff_3']
        if 'graph_mean_diff_3' in request.session:
            del request.session['graph_mean_diff_3']
        if 'headers_mean_diff_3' in request.session:
            del request.session['headers_mean_diff_3']
        if 'use_first_row_as_header_mean_diff_3' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_3', False)
        if 'alpha_value_mean_diff_3' in request.session:
            request.session.pop('alpha_value_mean_diff_3', False)
        context['data_mean_diff_3'] = ""
        context['results_mean_diff_3'] = None
        context['graph_mean_diff_3'] = None
        context['use_first_row_as_header_mean_diff_3'] = False
        context['alpha_value_mean_diff_3'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_4") == "true":
        if 'data_mean_diff_4' in request.session:
            del request.session['data_mean_diff_4']
        if 'results_mean_diff_4' in request.session:
            del request.session['results_mean_diff_4']
        if 'graph_mean_diff_4' in request.session:
            del request.session['graph_mean_diff_4']
        if 'headers_mean_diff_4' in request.session:
            del request.session['headers_mean_diff_4']
        if 'use_first_row_as_header_mean_diff_4' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_4', False)
        if 'alpha_value_mean_diff_4' in request.session:
            request.session.pop('alpha_value_mean_diff_4', False)
        context['data_mean_diff_4'] = ""
        context['results_mean_diff_4'] = None
        context['graph_mean_diff_4'] = None
        context['use_first_row_as_header_mean_diff_4'] = False
        context['alpha_value_mean_diff_4'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_var_quo") == "true":
        if 'data_var_quo' in request.session:
            del request.session['data_var_quo']
        if 'results_var_quo' in request.session:
            del request.session['results_var_quo']
        if 'graph_var_quo' in request.session:
            del request.session['graph_var_quo']
        if 'headers_var_quo' in request.session:
            del request.session['headers_var_quo']
        if 'use_first_row_as_header_var_quo' in request.session:
            request.session.pop('use_first_row_as_header_var_quo', False)
        if 'alpha_value_var_quo' in request.session:
            request.session.pop('alpha_value_var_quo', False)
        context['data_var_quo'] = ""
        context['results_var_quo'] = None
        context['graph_var_quo'] = None
        context['use_first_row_as_header_var_quo'] = False
        context['alpha_value_var_quo'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_prop_diff") == "true":
        if 'data_prop_diff' in request.session:
            del request.session['data_prop_diff']
        if 'results_prop_diff' in request.session:
            del request.session['results_prop_diff']
        if 'graph_prop_diff' in request.session:
            del request.session['graph_prop_diff']
        if 'headers_prop_diff' in request.session:
            del request.session['headers_prop_diff']
        if 'use_first_row_as_header_prop_diff' in request.session:
            request.session.pop('use_first_row_as_header_prop_diff', False)
        if 'alpha_value_prop_diff' in request.session:
            request.session.pop('alpha_value_prop_diff', False)
        context['data_prop_diff'] = ""
        context['results_prop_diff'] = None
        context['graph_prop_diff'] = None
        context['use_first_row_as_header_prop_diff'] = False
        context['alpha_value_prop_diff'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_test_1") == "true":
        if 'data_mean_test_1' in request.session:
            del request.session['data_mean_test_1']
        if 'results_mean_test_1' in request.session:
            del request.session['results_mean_test_1']
        if 'graph_mean_test_1' in request.session:
            del request.session['graph_mean_test_1']
        if 'headers_mean_test_1' in request.session:
            del request.session['headers_mean_test_1']
        if 'use_first_row_as_header_mean_test_1' in request.session:
            request.session.pop('use_first_row_as_header_mean_test_1', False)
        if 'test_type_mean_test_1' in request.session:
            del request.session['test_type_mean_test_1']
        if 'explanation_mean_test_1' in request.session:
            del request.session['explanation_mean_test_1']
        context['data_mean_test_1'] = ""
        context['results_mean_test_1'] = None
        context['graph_mean_test_1'] = None
        context['test_type_mean_test_1'] = None
        context['use_first_row_as_header_mean_test_1'] = False
        context['explanation_mean_test_1'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_test_2") == "true":
        if 'data_mean_test_2' in request.session:
            del request.session['data_mean_test_2']
        if 'results_mean_test_2' in request.session:
            del request.session['results_mean_test_2']
        if 'graph_mean_test_2' in request.session:
            del request.session['graph_mean_test_2']
        if 'headers_mean_test_2' in request.session:
            del request.session['headers_mean_test_2']
        if 'use_first_row_as_header_mean_test_2' in request.session:
            request.session.pop('use_first_row_as_header_mean_test_2', False)
        if 'test_type_mean_test_2' in request.session:
            del request.session['test_type_mean_test_2']
        if 'explanation_mean_test_2' in request.session:
            del request.session['explanation_mean_test_2']
        context['data_mean_test_2'] = ""
        context['results_mean_test_2'] = None
        context['graph_mean_test_2'] = None
        context['test_type_mean_test_2'] = None
        context['use_first_row_as_header_mean_test_2'] = False
        context['explanation_mean_test_2'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_var_test") == "true":
        if 'data_var_test' in request.session:
            del request.session['data_var_test']
        if 'results_var_test' in request.session:
            del request.session['results_var_test']
        if 'graph_var_test' in request.session:
            del request.session['graph_var_test']
        if 'headers_var_test' in request.session:
            del request.session['headers_var_test']
        if 'use_first_row_as_header_var_test' in request.session:
            request.session.pop('use_first_row_as_header_var_test', False)
        if 'test_type_var_test' in request.session:
            del request.session['test_type_var_test']
        if 'explanation_var_test' in request.session:
            del request.session['explanation_var_test']
        context['data_var_test'] = ""
        context['results_var_test'] = None
        context['graph_var_test'] = None
        context['test_type_var_test'] = None
        context['use_first_row_as_header_var_test'] = False
        context['explanation_var_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_test_1") == "true":
        if 'data_mean_diff_test_1' in request.session:
            del request.session['data_mean_diff_test_1']
        if 'results_mean_diff_test_1' in request.session:
            del request.session['results_mean_diff_test_1']
        if 'graph_mean_diff_test_1' in request.session:
            del request.session['graph_mean_diff_test_1']
        if 'headers_mean_diff_test_1' in request.session:
            del request.session['headers_mean_diff_test_1']
        if 'use_first_row_as_header_mean_diff_test_1' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_test_1', False)
        if 'test_type_mean_diff_test_1' in request.session:
            del request.session['test_type_mean_diff_test_1']
        if 'explanation_mean_diff_test_1' in request.session:
            del request.session['explanation_mean_diff_test_1']
        if 'alpha_value_mean_diff_test_1' in request.session:
            request.session.pop('alpha_value_mean_diff_test_1', False)
        context['data_mean_diff_test_1'] = ""
        context['results_mean_diff_test_1'] = None
        context['graph_mean_diff_test_1'] = None
        context['test_type_mean_diff_test_1'] = None
        context['use_first_row_as_header_mean_diff_test_1'] = False
        context['explanation_mean_diff_test_1'] = False
        context['alpha_value_mean_diff_test_1'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_test_2") == "true":
        if 'data_mean_diff_test_2' in request.session:
            del request.session['data_mean_diff_test_2']
        if 'results_mean_diff_test_2' in request.session:
            del request.session['results_mean_diff_test_2']
        if 'graph_mean_diff_test_2' in request.session:
            del request.session['graph_mean_diff_test_2']
        if 'headers_mean_diff_test_2' in request.session:
            del request.session['headers_mean_diff_test_2']
        if 'use_first_row_as_header_mean_diff_test_2' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_test_2', False)
        if 'test_type_mean_diff_test_2' in request.session:
            del request.session['test_type_mean_diff_test_2']
        if 'explanation_mean_diff_test_2' in request.session:
            del request.session['explanation_mean_diff_test_2']
        if 'alpha_value_mean_diff_test_2' in request.session:
            request.session.pop('alpha_value_mean_diff_test_2', False)
        context['data_mean_diff_test_2'] = ""
        context['results_mean_diff_test_2'] = None
        context['graph_mean_diff_test_2'] = None
        context['test_type_mean_diff_test_2'] = None
        context['use_first_row_as_header_mean_diff_test_2'] = False
        context['explanation_mean_diff_test_2'] = False
        context['alpha_value_mean_diff_test_2'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_test_3") == "true":
        if 'data_mean_diff_test_3' in request.session:
            del request.session['data_mean_diff_test_3']
        if 'results_mean_diff_test_3' in request.session:
            del request.session['results_mean_diff_test_3']
        if 'graph_mean_diff_test_3' in request.session:
            del request.session['graph_mean_diff_test_3']
        if 'headers_mean_diff_test_3' in request.session:
            del request.session['headers_mean_diff_test_3']
        if 'use_first_row_as_header_mean_diff_test_3' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_test_3', False)
        if 'test_type_mean_diff_test_3' in request.session:
            del request.session['test_type_mean_diff_test_3']
        if 'explanation_mean_diff_test_3' in request.session:
            del request.session['explanation_mean_diff_test_3']
        if 'alpha_value_mean_diff_test_3' in request.session:
            request.session.pop('alpha_value_mean_diff_test_3', False)
        context['data_mean_diff_test_3'] = ""
        context['results_mean_diff_test_3'] = None
        context['graph_mean_diff_test_3'] = None
        context['test_type_mean_diff_test_3'] = None
        context['use_first_row_as_header_mean_diff_test_3'] = False
        context['explanation_mean_diff_test_3'] = False
        context['alpha_value_mean_diff_test_3'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mean_diff_test_4") == "true":
        if 'data_mean_diff_test_4' in request.session:
            del request.session['data_mean_diff_test_4']
        if 'results_mean_diff_test_4' in request.session:
            del request.session['results_mean_diff_test_4']
        if 'graph_mean_diff_test_4' in request.session:
            del request.session['graph_mean_diff_test_4']
        if 'headers_mean_diff_test_4' in request.session:
            del request.session['headers_mean_diff_test_4']
        if 'use_first_row_as_header_mean_diff_test_4' in request.session:
            request.session.pop('use_first_row_as_header_mean_diff_test_4', False)
        if 'test_type_mean_diff_test_4' in request.session:
            del request.session['test_type_mean_diff_test_4']
        if 'explanation_mean_diff_test_4' in request.session:
            del request.session['explanation_mean_diff_test_4']
        if 'alpha_value_mean_diff_test_4' in request.session:
            request.session.pop('alpha_value_mean_diff_test_4', False)
        context['data_mean_diff_test_4'] = ""
        context['results_mean_diff_test_4'] = None
        context['graph_mean_diff_test_4'] = None
        context['test_type_mean_diff_test_4'] = None
        context['use_first_row_as_header_mean_diff_test_4'] = False
        context['explanation_mean_diff_test_4'] = False
        context['alpha_value_mean_diff_test_4'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_prop_test") == "true":
        if 'data_prop_test' in request.session:
            del request.session['data_prop_test']
        if 'results_prop_test' in request.session:
            del request.session['results_prop_test']
        if 'graph_prop_test' in request.session:
            del request.session['graph_prop_test']
        if 'headers_prop_test' in request.session:
            del request.session['headers_prop_test']
        if 'use_first_row_as_header_prop_test' in request.session:
            request.session.pop('use_first_row_as_header_prop_test', False)
        if 'test_type_prop_test' in request.session:
            del request.session['test_type_prop_test']
        if 'explanation_prop_test' in request.session:
            del request.session['explanation_prop_test']
        context['data_prop_test'] = ""
        context['results_prop_test'] = None
        context['graph_prop_test'] = None
        context['test_type_prop_test'] = None
        context['use_first_row_as_header_prop_test'] = False
        context['explanation_prop_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_prop_diff_test") == "true":
        if 'data_prop_diff_test' in request.session:
            del request.session['data_prop_diff_test']
        if 'results_prop_diff_test' in request.session:
            del request.session['results_prop_diff_test']
        if 'graph_prop_diff_test' in request.session:
            del request.session['graph_prop_diff_test']
        if 'headers_prop_diff_test' in request.session:
            del request.session['headers_prop_diff_test']
        if 'use_first_row_as_header_prop_diff_test' in request.session:
            request.session.pop('use_first_row_as_header_prop_diff_test', False)
        if 'test_type_prop_diff_test' in request.session:
            del request.session['test_type_prop_diff_test']
        if 'explanation_prop_diff_test' in request.session:
            del request.session['explanation_prop_diff_test']
        if 'alpha_value_prop_diff_test' in request.session:
            request.session.pop('alpha_value_prop_diff_test', False)
        context['data_prop_diff_test'] = ""
        context['results_prop_diff_test'] = None
        context['graph_prop_diff_test'] = None
        context['test_type_prop_diff_test'] = None
        context['use_first_row_as_header_prop_diff_test'] = False
        context['explanation_prop_diff_test'] = False
        context['alpha_value_prop_diff_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_var_quo_test") == "true":
        if 'data_var_quo_test' in request.session:
            del request.session['data_var_quo_test']
        if 'results_var_quo_test' in request.session:
            del request.session['results_var_quo_test']
        if 'graph_var_quo_test' in request.session:
            del request.session['graph_var_quo_test']
        if 'headers_var_quo_test' in request.session:
            del request.session['headers_var_quo_test']
        if 'use_first_row_as_header_var_quo_test' in request.session:
            request.session.pop('use_first_row_as_header_var_quo_test', False)
        if 'test_type_var_quo_test' in request.session:
            del request.session['test_type_var_quo_test']
        if 'explanation_var_quo_test' in request.session:
            del request.session['explanation_var_quo_test']
        if 'alpha_value_var_quo_test' in request.session:
            request.session.pop('alpha_value_var_quo_test', False)
        context['data_var_quo_test'] = ""
        context['results_var_quo_test'] = None
        context['graph_var_quo_test'] = None
        context['test_type_var_quo_test'] = None
        context['use_first_row_as_header_var_quo_test'] = False
        context['explanation_var_quo_test'] = False
        context['alpha_value_var_quo_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_U_test") == "true":
        if 'data_U_test' in request.session:
            del request.session['data_U_test']
        if 'results_U_test' in request.session:
            del request.session['results_U_test']
        if 'headers_U_test' in request.session:
            del request.session['headers_U_test']
        if 'use_first_row_as_header_U_test' in request.session:
            request.session.pop('use_first_row_as_header_U_test', False)
        if 'test_type_U_test' in request.session:
            del request.session['test_type_U_test']
        if 'explanation_U_test' in request.session:
            del request.session['explanation_U_test']
        if 'alpha_value_U_test' in request.session:
            request.session.pop('alpha_value_U_test', False)
        context['data_U_test'] = ""
        context['results_U_test'] = None
        context['test_type_U_test'] = None
        context['use_first_row_as_header_U_test'] = False
        context['explanation_U_test'] = False
        context['alpha_value_U_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_U_signed_test") == "true":
        if 'data_U_signed_test' in request.session:
            del request.session['data_U_signed_test']
        if 'results_U_signed_test' in request.session:
            del request.session['results_U_signed_test']
        if 'headers_U_signed_test' in request.session:
            del request.session['headers_U_signed_test']
        if 'use_first_row_as_header_U_signed_test' in request.session:
            request.session.pop('use_first_row_as_header_U_signed_test', False)
        if 'test_type_U_signed_test' in request.session:
            del request.session['test_type_U_signed_test']
        if 'explanation_U_signed_test' in request.session:
            del request.session['explanation_U_signed_test']
        if 'alpha_value_U_signed_test' in request.session:
            request.session.pop('alpha_value_U_signed_test', False)
        context['data_U_signed_test'] = ""
        context['results_U_signed_test'] = None
        context['test_type_U_signed_test'] = None
        context['use_first_row_as_header_U_signed_test'] = False
        context['explanation_U_signed_test'] = False
        context['alpha_value_U_signed_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_shap_test") == "true":
        if 'data_shap_test' in request.session:
            del request.session['data_shap_test']
        if 'results_shap_test' in request.session:
            del request.session['results_shap_test']
        if 'headers_shap_test' in request.session:
            del request.session['headers_shap_test']
        if 'use_first_row_as_header_shap_test' in request.session:
            request.session.pop('use_first_row_as_header_shap_test', False)
        if 'explanation_shap_test' in request.session:
            del request.session['explanation_shap_test']
        if 'alpha_value_shap_test' in request.session:
            request.session.pop('alpha_value_shap_test', False)
        context['data_shap_test'] = ""
        context['results_shap_test'] = None
        context['use_first_row_as_header_shap_test'] = False
        context['explanation_shap_test'] = False
        context['alpha_value_shap_test'] = False
        return render(request, "inference/inference.html", context)
    
    if request.method == "POST" and request.POST.get("clear_kol_test") == "true":
        if 'data_kol_test' in request.session:
            del request.session['data_kol_test']
        if 'results_kol_test' in request.session:
            del request.session['results_kol_test']
        if 'headers_kol_test' in request.session:
            del request.session['headers_kol_test']
        if 'use_first_row_as_header_kol_test' in request.session:
            request.session.pop('use_first_row_as_header_kol_test', False)
        if 'explanation_kol_test' in request.session:
            del request.session['explanation_kol_test']
        if 'alpha_value_kol_test' in request.session:
            request.session.pop('alpha_value_kol_test', False)
        context['data_kol_test'] = ""
        context['results_kol_test'] = None
        context['use_first_row_as_header_kol_test'] = False
        context['explanation_kol_test'] = False
        context['alpha_value_kol_test'] = False
        return render(request, "inference/inference.html", context)

####################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "mean" and subsubtab == "mean_1":
        data_mean_1 = request.POST.get('data_mean_1')
        use_first_row_as_header_mean_1 = request.POST.get('use_first_row_as_header_mean_1') == 'on'

        if not data_mean_1.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_1'] = None
            context['graph_mean_1'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_1 = data_mean_1.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_1.split('\n')]
        columns = []
        headers_mean_1 = []

        if use_first_row_as_header_mean_1 and rows:
            headers_mean_1 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_1:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_1'] = None
            context['results_mean_1'] = None
            return render(request, "inference/inference.html", context)
        
        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results_mean_1 = []
        means = []
        confidence_intervals = []
        n_vector = []
        variances = []
        alphas = []
        valid_columns = []
        invalid_columns = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) < 4:
                invalid_columns.append(i)
                continue

            n, mean, variance, alpha = valid_col[:4]
            std_error = np.sqrt(variance) / np.sqrt(n)
            z = norm.ppf(1 - (alpha / 100) / 2)
            margin_error = z * std_error
            conf_interval = (mean - margin_error, mean + margin_error)
            
            means.append(mean)
            confidence_intervals.append(conf_interval)
            n_vector.append(n)
            variances.append(variance)
            alphas.append(alpha)
            variable_name = headers_mean_1[i] if headers_mean_1 and i < len(headers_mean_1) else f"var. {i + 1}"
            
            results_mean_1.append({
                'variable': variable_name,
                'mean': str(mean),
                'confidence_interval': (float(format_str.format(conf_interval[0])), float(format_str.format(conf_interval[1]))),
                'n_elements': str(n),
                'variance': str(variance),
                'alpha': str(alpha),
            })

            valid_columns.append(i)
        
        if not valid_columns:
            context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, population variance, and significance level values."
            context['results_mean_1'] = None
            context['graph_mean_1'] = None
            return render(request, "inference/inference.html", context)

        if invalid_columns:
            context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, population variance, and significance level values."
        else:
            context['warning'] = None
        
        context['results_mean_1'] = results_mean_1
        context['data_mean_1'] = data_mean_1
        context['headers_mean_1'] = headers_mean_1 if use_first_row_as_header_mean_1 else None
        context['use_first_row_as_header_mean_1'] = 'checked' if use_first_row_as_header_mean_1 else ''

        request.session['results_mean_1'] = results_mean_1
        request.session['data_mean_1'] = data_mean_1
        request.session['headers_mean_1'] = headers_mean_1 if use_first_row_as_header_mean_1 else None
        request.session['use_first_row_as_header_mean_1'] = use_first_row_as_header_mean_1

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(means))
        
        ax.bar(index, means, bar_width, label='Mean', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            error_lower = means[i] - ci_lower
            error_upper = ci_upper - means[i]
            ax.errorbar(index[i], means[i], yerr=[[error_lower], [error_upper]], 
                        fmt='o', color='black', capsize=3)
        
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Mean Plot with Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results_mean_1], rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_1 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_1'] = f'data:image/png;base64,{graph_mean_1}'
        request.session['graph_mean_1'] = context['graph_mean_1']

###################################################################################################3
    if request.method == "POST" and tab == "confidence" and subtab == "mean" and subsubtab == "mean_2":
        data_mean_2 = request.POST.get('data_mean_2')
        use_first_row_as_header_mean_2 = request.POST.get('use_first_row_as_header_mean_2') == 'on'

        if not data_mean_2.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_2'] = None
            context['graph_mean_2'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_2 = data_mean_2.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_2.split('\n')]
        columns = []
        headers_mean_2 = []

        if use_first_row_as_header_mean_2 and rows:
            headers_mean_2 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_2:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_2'] = None
            context['results_mean_2'] = None
            return render(request, "inference/inference.html", context)
        
        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results_mean_2 = []
        means = []
        confidence_intervals = []
        n_vector = []
        variances = []
        alphas = []
        valid_columns = []
        invalid_columns = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) < 4:
                invalid_columns.append(i)
                continue

            n, mean, sample_variance, alpha = valid_col[:4]
            std_error = np.sqrt(sample_variance) / np.sqrt(n)
            t_critical = t.ppf(1 - (1 - alpha / 100) / 2, df=n - 1)
            margin_error = t_critical * std_error
            conf_interval = (mean - margin_error, mean + margin_error)
            
            means.append(mean)
            confidence_intervals.append(conf_interval)
            n_vector.append(n)
            variances.append(sample_variance)
            alphas.append(alpha)
            variable_name = headers_mean_2[i] if headers_mean_2 and i < len(headers_mean_2) else f"var. {i + 1}"
            
            results_mean_2.append({
                'variable': variable_name,
                'mean': str(mean),
                'confidence_interval': (float(format_str.format(conf_interval[0])), float(format_str.format(conf_interval[1]))),
                'n_elements': str(n),
                'variance': str(sample_variance),
                'alpha': str(alpha),
            })

            valid_columns.append(i)
        
        if not valid_columns:
            context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
            context['results_mean_2'] = None
            context['graph_mean_2'] = None
            return render(request, "inference/inference.html", context)

        if invalid_columns:
            context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
        else:
            context['warning'] = None
        
        context['results_mean_2'] = results_mean_2
        context['data_mean_2'] = data_mean_2
        context['headers_mean_2'] = headers_mean_2 if use_first_row_as_header_mean_2 else None
        context['use_first_row_as_header_mean_2'] = 'checked' if use_first_row_as_header_mean_2 else ''

        request.session['results_mean_2'] = results_mean_2
        request.session['data_mean_2'] = data_mean_2
        request.session['headers_mean_2'] = headers_mean_2 if use_first_row_as_header_mean_2 else None
        request.session['use_first_row_as_header_mean_2'] = use_first_row_as_header_mean_2

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(means))
        
        ax.bar(index, means, bar_width, label='Mean', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            error_lower = means[i] - ci_lower
            error_upper = ci_upper - means[i]
            ax.errorbar(index[i], means[i], yerr=[[error_lower], [error_upper]], 
                        fmt='o', color='black', capsize=3)
        
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Mean Plot with Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results_mean_2], rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_2 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_2'] = f'data:image/png;base64,{graph_mean_2}'
        request.session['graph_mean_2'] = context['graph_mean_2']

##########################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "var" and subsubtab == "var_1":
        data_var_1 = request.POST.get('data_var_1')
        use_first_row_as_header_var_1 = request.POST.get('use_first_row_as_header_var_1') == 'on'

        if not data_var_1.strip():
            context['error'] = "Please enter data before calculating."
            context['results_var_1'] = None
            context['graph_var_1'] = None
            return render(request, "inference/inference.html", context)
        
        data_var_1 = data_var_1.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_var_1.split('\n')]
        columns = []
        headers_var_1 = []

        if use_first_row_as_header_var_1 and rows:
            headers_var_1 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_var_1:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_var_1'] = None
            context['results_var_1'] = None
            return render(request, "inference/inference.html", context)
        
        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results_var_1 = []
        confidence_intervals = []
        n_vector = []
        variances = []
        alphas = []
        valid_columns = []
        invalid_columns = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) < 3:
                invalid_columns.append(i)
                continue

            n, variance, alpha = valid_col[:3]
            gl = n-1
            chi2_inf = chi2.ppf((1 - alpha / 100) / 2, gl)
            chi2_sup = chi2.ppf(1 - (1 - alpha / 100) / 2, gl)
            lim_inf = (gl * variance) / chi2_sup
            lim_sup = (gl * variance) / chi2_inf
            conf_interval = (lim_inf, lim_sup)
            
            confidence_intervals.append(conf_interval)
            n_vector.append(n)
            variances.append(variance)
            alphas.append(alpha)
            variable_name = headers_var_1[i] if headers_var_1 and i < len(headers_var_1) else f"var. {i + 1}"
            
            results_var_1.append({
                'variable': variable_name,
                'confidence_interval': (float(format_str.format(conf_interval[0])), float(format_str.format(conf_interval[1]))),
                'n_elements': str(n),
                'variance': str(variance),
                'alpha': str(alpha),
            })

            valid_columns.append(i)
        
        if not valid_columns:
            context['error'] = "Error: Incomplete data. Please ensure each column has n, sample variance, and significance level values."
            context['results_var_1'] = None
            context['graph_var_1'] = None
            return render(request, "inference/inference.html", context)

        if invalid_columns:
            context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, sample variance, and significance level values."
        else:
            context['warning'] = None
        
        context['results_var_1'] = results_var_1
        context['data_var_1'] = data_var_1
        context['headers_var_1'] = headers_var_1 if use_first_row_as_header_var_1 else None
        context['use_first_row_as_header_var_1'] = 'checked' if use_first_row_as_header_var_1 else ''

        request.session['results_var_1'] = results_var_1
        request.session['data_var_1'] = data_var_1
        request.session['headers_var_1'] = headers_var_1 if use_first_row_as_header_var_1 else None
        request.session['use_first_row_as_header_var_1'] = use_first_row_as_header_var_1

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(variances))
        
        ax.bar(index, variances, bar_width, label='Variance', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            error_lower = variances[i] - ci_lower
            error_upper = ci_upper - variances[i]
            ax.errorbar(index[i], variances[i], yerr=[[error_lower], [error_upper]], 
                        fmt='o', color='black', capsize=3)
        
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Variance Plot with Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results_var_1], rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_var_1 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_var_1'] = f'data:image/png;base64,{graph_var_1}'
        request.session['graph_var_1'] = context['graph_var_1']

##############################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "var" and subsubtab == "var_2":
        data_var_2 = request.POST.get('data_var_2')
        use_first_row_as_header_var_2 = request.POST.get('use_first_row_as_header_var_2') == 'on'

        if not data_var_2.strip():
            context['error'] = "Please enter data before calculating."
            context['results_var_2'] = None
            context['graph_var_2'] = None
            return render(request, "inference/inference.html", context)
        
        data_var_2 = data_var_2.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_var_2.split('\n')]
        columns = []
        headers_var_2 = []

        if use_first_row_as_header_var_2 and rows:
            headers_var_2 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_var_2:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_var_2'] = None
            context['results_var_2'] = None
            return render(request, "inference/inference.html", context)
        
        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results_var_2 = []
        confidence_intervals = []
        n_vector = []
        variances = []
        alphas = []
        valid_columns = []
        invalid_columns = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) < 3:
                invalid_columns.append(i)
                continue

            n, variance, alpha = valid_col[:3]
            gl = n
            chi2_inf = chi2.ppf((1 - alpha / 100) / 2, gl)
            chi2_sup = chi2.ppf(1 - (1 - alpha / 100) / 2, gl)
            lim_inf = (gl * variance) / chi2_sup
            lim_sup = (gl * variance) / chi2_inf
            conf_interval = (lim_inf, lim_sup)
            
            confidence_intervals.append(conf_interval)
            n_vector.append(n)
            variances.append(variance)
            alphas.append(alpha)
            variable_name = headers_var_2[i] if headers_var_2 and i < len(headers_var_2) else f"var. {i + 1}"
            
            results_var_2.append({
                'variable': variable_name,
                'confidence_interval': (float(format_str.format(conf_interval[0])), float(format_str.format(conf_interval[1]))),
                'n_elements': str(n),
                'variance': str(variance),
                'alpha': str(alpha),
            })

            valid_columns.append(i)
        
        if not valid_columns:
            context['error'] = "Error: Incomplete data. Please ensure each column has n, sample variance, and significance level values."
            context['results_var_2'] = None
            context['graph_var_2'] = None
            return render(request, "inference/inference.html", context)

        if invalid_columns:
            context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, sample variance, and significance level values."
        else:
            context['warning'] = None
        
        context['results_var_2'] = results_var_2
        context['data_var_2'] = data_var_2
        context['headers_var_2'] = headers_var_2 if use_first_row_as_header_var_2 else None
        context['use_first_row_as_header_var_2'] = 'checked' if use_first_row_as_header_var_2 else ''

        request.session['results_var_2'] = results_var_2
        request.session['data_var_2'] = data_var_2
        request.session['headers_var_2'] = headers_var_2 if use_first_row_as_header_var_2 else None
        request.session['use_first_row_as_header_var_2'] = use_first_row_as_header_var_2

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(variances))
        
        ax.bar(index, variances, bar_width, label='Variance', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            error_lower = variances[i] - ci_lower
            error_upper = ci_upper - variances[i]
            ax.errorbar(index[i], variances[i], yerr=[[error_lower], [error_upper]], 
                        fmt='o', color='black', capsize=3)
        
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Variance Plot with Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results_var_2], rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_var_2 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_var_2'] = f'data:image/png;base64,{graph_var_2}'
        request.session['graph_var_2'] = context['graph_var_2']

##########################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "prop":
        data_prop = request.POST.get('data_prop')
        use_first_row_as_header_prop = request.POST.get('use_first_row_as_header_prop') == 'on'

        if not data_prop.strip():
            context['error'] = "Please enter data before calculating."
            context['results_prop'] = None
            context['graph_prop'] = None
            return render(request, "inference/inference.html", context)
        
        data_prop = data_prop.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_prop.split('\n')]
        columns = []
        headers_prop = []

        if use_first_row_as_header_prop and rows:
            headers_prop = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_prop:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_prop'] = None
            context['results_prop'] = None
            return render(request, "inference/inference.html", context)
        
        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results_prop = []
        props = []
        confidence_intervals = []
        n_vector = []
        alphas = []
        valid_columns = []
        invalid_columns = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) < 3:
                invalid_columns.append(i)
                continue

            n, prop, alpha = valid_col[:3]
            std_error = np.sqrt((prop * (1 - prop)) / n)
            z = norm.ppf(1 - (1 - alpha / 100) / 2)
            margin_error = z * std_error
            conf_interval = (prop - margin_error, prop + margin_error)
            
            props.append(prop)
            confidence_intervals.append(conf_interval)
            n_vector.append(n)
            alphas.append(alpha)
            variable_name = headers_prop[i] if headers_prop and i < len(headers_prop) else f"var. {i + 1}"
            
            results_prop.append({
                'variable': variable_name,
                'prop': str(prop),
                'confidence_interval': (float(format_str.format(conf_interval[0])), float(format_str.format(conf_interval[1]))),
                'n_elements': str(n),
                'alpha': str(alpha),
            })

            valid_columns.append(i)
        
        if not valid_columns:
            context['error'] = "Error: Incomplete data. Please ensure each column has n, proportion, and significance level values."
            context['results_prop'] = None
            context['graph_prop'] = None
            return render(request, "inference/inference.html", context)

        if invalid_columns:
            context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, proportion, and significance level values."
        else:
            context['warning'] = None
        
        context['results_prop'] = results_prop
        context['data_prop'] = data_prop
        context['headers_prop'] = headers_prop if use_first_row_as_header_prop else None
        context['use_first_row_as_header_prop'] = 'checked' if use_first_row_as_header_prop else ''

        request.session['results_prop'] = results_prop
        request.session['data_prop'] = data_prop
        request.session['headers_prop'] = headers_prop if use_first_row_as_header_prop else None
        request.session['use_first_row_as_header_prop'] = use_first_row_as_header_prop

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(props))
        
        ax.bar(index, props, bar_width, label='Proportion', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            error_lower = props[i] - ci_lower
            error_upper = ci_upper - props[i]
            ax.errorbar(index[i], props[i], yerr=[[error_lower], [error_upper]], 
                        fmt='o', color='black', capsize=3)
        
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Proportion Plot with Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results_prop], rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_prop = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_prop'] = f'data:image/png;base64,{graph_prop}'
        request.session['graph_prop'] = context['graph_prop']

######################################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "mean_diff" and subsubtab == "mean_diff_1":
        data_mean_diff_1 = request.POST.get('data_mean_diff_1')
        use_first_row_as_header_mean_diff_1 = request.POST.get('use_first_row_as_header_mean_diff_1') == 'on'
        alpha_value_mean_diff_1 = request.POST.get('alpha_value_mean_diff_1')
    
        if alpha_value_mean_diff_1:
            try:
                alpha_value_mean_diff_1 = float(alpha_value_mean_diff_1)
            except ValueError:
                alpha_value_mean_diff_1 = None


        if not data_mean_diff_1.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_1'] = None
            context['graph_mean_diff_1'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_diff_1 = data_mean_diff_1.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_1.split('\n')]
        columns = []
        headers_mean_diff_1 = []

        if use_first_row_as_header_mean_diff_1 and rows:
            headers_mean_diff_1 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_1:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_1'] = None
            context['results_mean_diff_1'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_1:
            alpha_value_mean_diff_1 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
            n1 = int(data_columns[0][0])
            mean1 = float(data_columns[0][1])
            variance1 = float(data_columns[0][2])
            n2 = int(data_columns[1][0])
            mean2 = float(data_columns[1][1])
            variance2 = float(data_columns[1][2])
            mean_difference = mean1 - mean2
            std_error_diff = np.sqrt(variance1 / n1 + variance2 / n2)
            z = norm.ppf(1 - (1 - float(alpha_value_mean_diff_1) / 100) / 2)
            margin_error_diff = z * std_error_diff
            conf_interval = (mean_difference - margin_error_diff, mean_difference + margin_error_diff)
            n_vector = [n1, n2]
            means = [mean1, mean2]
            variances = [variance1, variance2]

        else:
            context['error'] = "The data entered is incorrect."

        variable_name = [headers_mean_diff_1[0],headers_mean_diff_1[1]] if headers_mean_diff_1 else ["var. 1", "var. 2"]
            
        results_mean_diff_1 = [{
            'variable': f"{variable_name[0]}, {variable_name[1]}",
            'n_elements': f"{n1}, {n2}",
            'variance': f"{variance1}, {variance2}",
            'alpha': alpha_value_mean_diff_1,
            'mean': f"{mean1}, {mean2}",
            'mean_diff': f"{mean_difference:.4f}",
            'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
        }]

        context['results_mean_diff_1'] = results_mean_diff_1
        context['data_mean_diff_1'] = data_mean_diff_1
        context['alpha_value_mean_diff_1'] = alpha_value_mean_diff_1
        context['headers_mean_diff_1'] = headers_mean_diff_1 if use_first_row_as_header_mean_diff_1 else None
        context['use_first_row_as_header_mean_diff_1'] = 'checked' if use_first_row_as_header_mean_diff_1 else ''

        request.session['results_mean_diff_1'] = results_mean_diff_1
        request.session['data_mean_diff_1'] = data_mean_diff_1
        request.session['alpha_value_mean_diff_1'] = alpha_value_mean_diff_1
        request.session['headers_mean_diff_1'] = headers_mean_diff_1 if use_first_row_as_header_mean_diff_1 else None
        request.session['use_first_row_as_header_mean_diff_1'] = use_first_row_as_header_mean_diff_1

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, mean_difference, 'o', color='#9368E9', markersize=8, label='Mean Difference')
        error_lower = mean_difference - conf_interval[0]
        error_upper = conf_interval[1] - mean_difference
        ax.errorbar(index[0], mean_difference, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='y = 0')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Difference of Means with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Mean Difference'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_diff_1 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_diff_1'] = f'data:image/png;base64,{graph_mean_diff_1}'
        request.session['graph_mean_diff_1'] = context['graph_mean_diff_1']

#################################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "mean_diff" and subsubtab == "mean_diff_2":
        data_mean_diff_2 = request.POST.get('data_mean_diff_2')
        use_first_row_as_header_mean_diff_2 = request.POST.get('use_first_row_as_header_mean_diff_2') == 'on'
        alpha_value_mean_diff_2 = request.POST.get('alpha_value_mean_diff_2')
    
        if alpha_value_mean_diff_2:
            try:
                alpha_value_mean_diff_2 = float(alpha_value_mean_diff_2)
            except ValueError:
                alpha_value_mean_diff_2 = None


        if not data_mean_diff_2.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_2'] = None
            context['graph_mean_diff_2'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_diff_2 = data_mean_diff_2.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_2.split('\n')]
        columns = []
        headers_mean_diff_2 = []

        if use_first_row_as_header_mean_diff_2 and rows:
            headers_mean_diff_2 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_2:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_2'] = None
            context['results_mean_diff_2'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_2:
            alpha_value_mean_diff_2 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
            n1 = int(data_columns[0][0])
            mean1 = float(data_columns[0][1])
            variance1 = float(data_columns[0][2])
            n2 = int(data_columns[1][0])
            mean2 = float(data_columns[1][1])
            variance2 = float(data_columns[1][2])
            pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
            mean_difference = mean1 - mean2
            std_error_diff = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
            degrees_of_freedom = n1 + n2 - 2
            t_critical = t.ppf(1 - (1 - float(alpha_value_mean_diff_2) / 100) / 2, df=degrees_of_freedom)
            margin_error_diff = t_critical * std_error_diff
            conf_interval = (mean_difference - margin_error_diff, mean_difference + margin_error_diff)
            n_vector = [n1, n2]
            means = [mean1, mean2]
            variances = [variance1, variance2]

        else:
            context['error'] = "The data entered is incorrect."

        variable_name = [headers_mean_diff_2[0],headers_mean_diff_2[1]] if headers_mean_diff_2 else ["var. 1", "var. 2"]
            
        results_mean_diff_2 = [{
            'variable': f"{variable_name[0]}, {variable_name[1]}",
            'n_elements': f"{n1}, {n2}",
            'variance': f"{variance1}, {variance2}",
            'alpha': alpha_value_mean_diff_2,
            'mean': f"{mean1}, {mean2}",
            'mean_diff': f"{mean_difference:.4f}",
            'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
        }]

        context['results_mean_diff_2'] = results_mean_diff_2
        context['data_mean_diff_2'] = data_mean_diff_2
        context['alpha_value_mean_diff_2'] = alpha_value_mean_diff_2
        context['headers_mean_diff_2'] = headers_mean_diff_2 if use_first_row_as_header_mean_diff_2 else None
        context['use_first_row_as_header_mean_diff_2'] = 'checked' if use_first_row_as_header_mean_diff_2 else ''

        request.session['results_mean_diff_2'] = results_mean_diff_2
        request.session['data_mean_diff_2'] = data_mean_diff_2
        request.session['alpha_value_mean_diff_2'] = alpha_value_mean_diff_2
        request.session['headers_mean_diff_2'] = headers_mean_diff_2 if use_first_row_as_header_mean_diff_2 else None
        request.session['use_first_row_as_header_mean_diff_2'] = use_first_row_as_header_mean_diff_2

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, mean_difference, 'o', color='#9368E9', markersize=8, label='Mean Difference')
        error_lower = mean_difference - conf_interval[0]
        error_upper = conf_interval[1] - mean_difference
        ax.errorbar(index[0], mean_difference, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='y = 0')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Difference of Means with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Mean Difference'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_diff_2 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_diff_2'] = f'data:image/png;base64,{graph_mean_diff_2}'
        request.session['graph_mean_diff_2'] = context['graph_mean_diff_2']

#########################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "mean_diff" and subsubtab == "mean_diff_3":
        data_mean_diff_3 = request.POST.get('data_mean_diff_3')
        use_first_row_as_header_mean_diff_3 = request.POST.get('use_first_row_as_header_mean_diff_3') == 'on'
        alpha_value_mean_diff_3 = request.POST.get('alpha_value_mean_diff_3')
    
        if alpha_value_mean_diff_3:
            try:
                alpha_value_mean_diff_3 = float(alpha_value_mean_diff_3)
            except ValueError:
                alpha_value_mean_diff_3 = None


        if not data_mean_diff_3.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_3'] = None
            context['graph_mean_diff_3'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_diff_3 = data_mean_diff_3.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_3.split('\n')]
        columns = []
        headers_mean_diff_3 = []

        if use_first_row_as_header_mean_diff_3 and rows:
            headers_mean_diff_3 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_3:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_3'] = None
            context['results_mean_diff_3'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_3:
            alpha_value_mean_diff_3 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
            n1 = int(data_columns[0][0])
            mean1 = float(data_columns[0][1])
            variance1 = float(data_columns[0][2])
            n2 = int(data_columns[1][0])
            mean2 = float(data_columns[1][1])
            variance2 = float(data_columns[1][2])
            mean_difference = mean1 - mean2
            std_error_diff = np.sqrt((variance1 / n1) + (variance2 / n2))
            numerator = (variance1 / n1 + variance2 / n2) ** 2
            denominator = ((variance1 / n1) ** 2 / (n1 - 1)) + ((variance2 / n2) ** 2 / (n2 - 1))
            degrees_of_freedom = numerator / denominator
            t_critical = t.ppf(1 - (1 - float(alpha_value_mean_diff_3) / 100) / 2, df=degrees_of_freedom)
            margin_error_diff = t_critical * std_error_diff
            conf_interval = (mean_difference - margin_error_diff, mean_difference + margin_error_diff)
            n_vector = [n1, n2]
            means = [mean1, mean2]
            variances = [variance1, variance2]

        else:
            context['error'] = "The data entered is incorrect."

        variable_name = [headers_mean_diff_3[0],headers_mean_diff_3[1]] if headers_mean_diff_3 else ["var. 1", "var. 2"]
            
        results_mean_diff_3 = [{
            'variable': f"{variable_name[0]}, {variable_name[1]}",
            'n_elements': f"{n1}, {n2}",
            'variance': f"{variance1}, {variance2}",
            'alpha': alpha_value_mean_diff_3,
            'mean': f"{mean1}, {mean2}",
            'mean_diff': f"{mean_difference:.4f}",
            'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
        }]

        context['results_mean_diff_3'] = results_mean_diff_3
        context['data_mean_diff_3'] = data_mean_diff_3
        context['alpha_value_mean_diff_3'] = alpha_value_mean_diff_3
        context['headers_mean_diff_3'] = headers_mean_diff_3 if use_first_row_as_header_mean_diff_3 else None
        context['use_first_row_as_header_mean_diff_3'] = 'checked' if use_first_row_as_header_mean_diff_3 else ''

        request.session['results_mean_diff_3'] = results_mean_diff_3
        request.session['data_mean_diff_3'] = data_mean_diff_3
        request.session['alpha_value_mean_diff_3'] = alpha_value_mean_diff_3
        request.session['headers_mean_diff_3'] = headers_mean_diff_3 if use_first_row_as_header_mean_diff_3 else None
        request.session['use_first_row_as_header_mean_diff_3'] = use_first_row_as_header_mean_diff_3

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, mean_difference, 'o', color='#9368E9', markersize=8, label='Mean Difference')
        error_lower = mean_difference - conf_interval[0]
        error_upper = conf_interval[1] - mean_difference
        ax.errorbar(index[0], mean_difference, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='y = 0')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Difference of Means with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Mean Difference'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_diff_3 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_diff_3'] = f'data:image/png;base64,{graph_mean_diff_3}'
        request.session['graph_mean_diff_3'] = context['graph_mean_diff_3']

###################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "mean_diff" and subsubtab == "mean_diff_4":
        data_mean_diff_4 = request.POST.get('data_mean_diff_4')
        use_first_row_as_header_mean_diff_4 = request.POST.get('use_first_row_as_header_mean_diff_4') == 'on'
        alpha_value_mean_diff_4 = request.POST.get('alpha_value_mean_diff_4')
    
        if alpha_value_mean_diff_4:
            try:
                alpha_value_mean_diff_4 = float(alpha_value_mean_diff_4)
            except ValueError:
                alpha_value_mean_diff_4 = None


        if not data_mean_diff_4.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_4'] = None
            context['graph_mean_diff_4'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_diff_4 = data_mean_diff_4.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_4.split('\n')]
        columns = []
        headers_mean_diff_4 = []

        if use_first_row_as_header_mean_diff_4 and rows:
            headers_mean_diff_4 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_4:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_4'] = None
            context['results_mean_diff_4'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_4:
            alpha_value_mean_diff_4 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and len(data_columns[0]) == len(data_columns[1]):
            data_1 = np.array([float(value) for value in data_columns[0]])
            data_2 = np.array([float(value) for value in data_columns[1]])

            if len(data_1) > 0:
                differences = data_1 - data_2
                n = len(differences)
                mean_difference = np.mean(differences)
                std_dev_diff = np.std(differences, ddof=1)
                std_error_diff = std_dev_diff / np.sqrt(n)
                t_critical = t.ppf(1 - (1 - float(alpha_value_mean_diff_4) / 100) / 2, df=n-1)
                margin_error_diff = t_critical * std_error_diff
                conf_interval = (mean_difference - margin_error_diff, mean_difference + margin_error_diff)
                variable_name = [headers_mean_diff_4[0],headers_mean_diff_4[1]] if headers_mean_diff_4 else ["var. 1", "var. 2"]

                results_mean_diff_4 = [{
                    'variable': f"{variable_name[0]}, {variable_name[1]}",
                    'n_elements': n,
                    'mean_diff': f"{mean_difference:.4f}",
                    'std_dev_diff': f"{std_dev_diff:.4f}",
                    'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})",
                    'alpha': alpha_value_mean_diff_4,
                }]
            else:
                context['error'] = "The data entered is incorrect."
        else:
            context['error'] = "The data entered is incorrect."

        
            
        context['results_mean_diff_4'] = results_mean_diff_4
        context['data_mean_diff_4'] = data_mean_diff_4
        context['alpha_value_mean_diff_4'] = alpha_value_mean_diff_4
        context['headers_mean_diff_4'] = headers_mean_diff_4 if use_first_row_as_header_mean_diff_4 else None
        context['use_first_row_as_header_mean_diff_4'] = 'checked' if use_first_row_as_header_mean_diff_4 else ''

        request.session['results_mean_diff_4'] = results_mean_diff_4
        request.session['data_mean_diff_4'] = data_mean_diff_4
        request.session['alpha_value_mean_diff_4'] = alpha_value_mean_diff_4
        request.session['headers_mean_diff_4'] = headers_mean_diff_4 if use_first_row_as_header_mean_diff_4 else None
        request.session['use_first_row_as_header_mean_diff_4'] = use_first_row_as_header_mean_diff_4

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, mean_difference, 'o', color='#9368E9', markersize=8, label='Mean Difference')
        error_lower = mean_difference - conf_interval[0]
        error_upper = conf_interval[1] - mean_difference
        ax.errorbar(index[0], mean_difference, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='y = 0')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Difference of Means with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Mean Difference'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_mean_diff_4 = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_mean_diff_4'] = f'data:image/png;base64,{graph_mean_diff_4}'
        request.session['graph_mean_diff_4'] = context['graph_mean_diff_4']

#######################################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "var_quo":
        data_var_quo = request.POST.get('data_var_quo')
        use_first_row_as_header_var_quo = request.POST.get('use_first_row_as_header_var_quo') == 'on'
        alpha_value_var_quo = request.POST.get('alpha_value_var_quo')
    
        if alpha_value_var_quo:
            try:
                alpha_value_var_quo = float(alpha_value_var_quo)
            except ValueError:
                alpha_value_var_quo = None


        if not data_var_quo.strip():
            context['error'] = "Please enter data before calculating."
            context['results_var_quo'] = None
            context['graph_var_quo'] = None
            return render(request, "inference/inference.html", context)
        
        data_var_quo = data_var_quo.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_var_quo.split('\n')]
        columns = []
        headers_var_quo = []

        if use_first_row_as_header_var_quo and rows:
            headers_var_quo = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_var_quo:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_var_quo'] = None
            context['results_var_quo'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_var_quo:
            alpha_value_var_quo = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
            n1 = int(data_columns[0][0])
            variance1 = float(data_columns[0][1])
            n2 = int(data_columns[1][0])
            variance2 = float(data_columns[1][1])
            variance_ratio = variance1 / variance2
            df1 = n1 - 1
            df2 = n2 - 1
            alpha = 1 - (float(alpha_value_var_quo) / 100)
            f_critical_low = f.ppf(alpha / 2, df1, df2)
            f_critical_high = f.ppf(1 - alpha / 2, df1, df2)
            conf_interval = (variance_ratio / f_critical_high, variance_ratio / f_critical_low)
            n_vector = [n1, n2]
            variances = [variance1, variance2]

        else:
            context['error'] = "The data entered is incorrect."

        variable_name = [headers_var_quo[0],headers_var_quo[1]] if headers_var_quo else ["var. 1", "var. 2"]
            
        results_var_quo = [{
            'variable': f"{variable_name[0]}, {variable_name[1]}",
            'n_elements': f"{n1}, {n2}",
            'variance': f"{variance1}, {variance2}",
            'alpha': alpha_value_var_quo,
            'var_ratio': f"{variance_ratio:.4f}",
            'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
        }]

        context['results_var_quo'] = results_var_quo
        context['data_var_quo'] = data_var_quo
        context['alpha_value_var_quo'] = alpha_value_var_quo
        context['headers_var_quo'] = headers_var_quo if use_first_row_as_header_var_quo else None
        context['use_first_row_as_header_var_quo'] = 'checked' if use_first_row_as_header_var_quo else ''

        request.session['results_var_quo'] = results_var_quo
        request.session['data_var_quo'] = data_var_quo
        request.session['alpha_value_var_quo'] = alpha_value_var_quo
        request.session['headers_var_quo'] = headers_var_quo if use_first_row_as_header_var_quo else None
        request.session['use_first_row_as_header_var_quo'] = use_first_row_as_header_var_quo

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, variance_ratio, 'o', color='#9368E9', markersize=8, label='Variance Ratio')
        error_lower = variance_ratio - conf_interval[0]
        error_upper = conf_interval[1] - variance_ratio
        ax.errorbar(index, variance_ratio, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(1, color='blue', linestyle='--', linewidth=1, label='y = 1')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Variances Ratio with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Variances Ratio'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_var_quo = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_var_quo'] = f'data:image/png;base64,{graph_var_quo}'
        request.session['graph_var_quo'] = context['graph_var_quo']

################################################################################################
    if request.method == "POST" and tab == "confidence" and subtab == "prop_diff":
        data_prop_diff = request.POST.get('data_prop_diff')
        use_first_row_as_header_prop_diff = request.POST.get('use_first_row_as_header_prop_diff') == 'on'
        alpha_value_prop_diff = request.POST.get('alpha_value_prop_diff')
    
        if alpha_value_prop_diff:
            try:
                alpha_value_prop_diff = float(alpha_value_prop_diff)
            except ValueError:
                alpha_value_prop_diff = None


        if not data_prop_diff.strip():
            context['error'] = "Please enter data before calculating."
            context['results_prop_diff'] = None
            context['graph_prop_diff'] = None
            return render(request, "inference/inference.html", context)
        
        data_prop_diff = data_prop_diff.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_prop_diff.split('\n')]
        columns = []
        headers_prop_diff = []

        if use_first_row_as_header_prop_diff and rows:
            headers_prop_diff = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_prop_diff:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_prop_diff'] = None
            context['results_prop_diff'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_prop_diff:
            alpha_value_prop_diff = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
            n1 = int(data_columns[0][0])
            p1 = float(data_columns[0][1])
            n2 = int(data_columns[1][0])
            p2 = float(data_columns[1][1])
            prop_diff = p1 - p2
            std_error = np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
            alpha = 1 - (float(alpha_value_prop_diff) / 100)
            z_critical = norm.ppf(1 - alpha / 2)
            margin_error = z_critical * std_error
            conf_interval = (prop_diff - margin_error, prop_diff + margin_error)
            n_vector = [n1, n2]
            proportions = [p1, p2]

        else:
            context['error'] = "The data entered is incorrect."

        variable_name = [headers_prop_diff[0],headers_prop_diff[1]] if headers_prop_diff else ["var. 1", "var. 2"]
            
        results_prop_diff = [{
            'variable': f"{variable_name[0]}, {variable_name[1]}",
            'n_elements': f"{n1}, {n2}",
            'proportion': f"{p1}, {p2}",
            'alpha': alpha_value_prop_diff,
            'prop_diff': f"{prop_diff:.4f}",
            'confidence_interval': f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
        }]

        context['results_prop_diff'] = results_prop_diff
        context['data_prop_diff'] = data_prop_diff
        context['alpha_value_prop_diff'] = alpha_value_prop_diff
        context['headers_prop_diff'] = headers_prop_diff if use_first_row_as_header_prop_diff else None
        context['use_first_row_as_header_prop_diff'] = 'checked' if use_first_row_as_header_prop_diff else ''

        request.session['results_prop_diff'] = results_prop_diff
        request.session['data_prop_diff'] = data_prop_diff
        request.session['alpha_value_prop_diff'] = alpha_value_prop_diff
        request.session['headers_prop_diff'] = headers_prop_diff if use_first_row_as_header_prop_diff else None
        request.session['use_first_row_as_header_prop_diff'] = use_first_row_as_header_prop_diff

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.array([0])

        ax.plot(index, prop_diff, 'o', color='#9368E9', markersize=8, label='Proportion Difference')
        error_lower = prop_diff - conf_interval[0]
        error_upper = conf_interval[1] - prop_diff
        ax.errorbar(index, prop_diff, yerr=[[error_lower], [error_upper]], 
                    fmt='o', color='black', capsize=5, label='Confidence Interval')
        
        ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='y = 0')
        ax.set_ylabel('Value', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Proportions Difference with Confidence Interval', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(['Proportions Difference'], fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_prop_diff = base64.b64encode(buf.read()).decode('utf-8')

        context['graph_prop_diff'] = f'data:image/png;base64,{graph_prop_diff}'
        request.session['graph_prop_diff'] = context['graph_prop_diff']

#############################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_test" and subsubtab == "mean_test_1":
        data_mean_test_1 = request.POST.get('data_mean_test_1')
        use_first_row_as_header_mean_test_1 = request.POST.get('use_first_row_as_header_mean_test_1') == 'on'
        test_type_mean_test_1 = request.POST.get('test_type_mean_test_1')

        if not data_mean_test_1.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_test_1'] = None
            context['graph_mean_test_1'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_test_1 = data_mean_test_1.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_test_1.split('\n')]
        columns = []
        headers_mean_test_1 = []

        if use_first_row_as_header_mean_test_1 and rows:
            headers_mean_test_1 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_test_1:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_test_1'] = None
            context['results_mean_test_1'] = None
            return render(request, "inference/inference.html", context)

        if test_type_mean_test_1 == "greater":
            results_mean_test_1 = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                    
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                
                n, mean, observed_mean, variance, alpha = valid_col[:5]
                std_error = np.sqrt(variance) / np.sqrt(n)
                z_score = (mean - observed_mean) / std_error
                z_critical = norm.ppf(alpha / 100)
                p_value = 1 - norm.cdf(z_score)
                reject_null = "Yes" if z_score > z_critical else "No"
                variable_name = headers_mean_test_1[i] if headers_mean_test_1 and i < len(headers_mean_test_1) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_mean_test_1.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(variance),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                
                valid_columns.append(i)
                variable_names.append(variable_name)
            
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, population variance, and significance level values."
                context['results_mean_test_1'] = None
                context['graph_mean_test_1'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, population variance, and significance level values."
            else:
                context['warning'] = None
            
            context['results_mean_test_1'] = results_mean_test_1
            context['data_mean_test_1'] = data_mean_test_1
            context['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            context['use_first_row_as_header_mean_test_1'] = 'checked' if use_first_row_as_header_mean_test_1 else ''
            context['test_type_mean_test_1'] = test_type_mean_test_1

            request.session['results_mean_test_1'] = results_mean_test_1
            request.session['data_mean_test_1'] = data_mean_test_1
            request.session['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            request.session['use_first_row_as_header_mean_test_1'] = use_first_row_as_header_mean_test_1
            request.session['test_type_mean_test_1'] = test_type_mean_test_1

            graph_mean_test_1 = create_unilateral_rigth_graph_mean_test_1(variable_names, z_scores, z_criticals)

            context['graph_mean_test_1'] = graph_mean_test_1
            request.session['graph_mean_test_1'] = graph_mean_test_1

            explanation_mean_test_1 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ > μ₀"
            context['explanation_mean_test_1'] = explanation_mean_test_1
            request.session['explanation_mean_test_1'] = explanation_mean_test_1

        elif test_type_mean_test_1 == "lesser":
            results_mean_test_1 = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                    
                n, mean, observed_mean, variance, alpha = valid_col[:5]
                std_error = np.sqrt(variance) / np.sqrt(n)
                z_score = (mean - observed_mean) / std_error
                z_critical = norm.ppf(1 - alpha / 100)
                p_value = norm.cdf(z_score)
                reject_null = "Yes" if z_score < z_critical else "No"
                variable_name = headers_mean_test_1[i] if headers_mean_test_1 and i < len(headers_mean_test_1) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_mean_test_1.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(variance),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, population variance, and significance level values."
                context['results_mean_test_1'] = None
                context['graph_mean_test_1'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, population variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_mean_test_1'] = results_mean_test_1
            context['data_mean_test_1'] = data_mean_test_1
            context['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            context['use_first_row_as_header_mean_test_1'] = 'checked' if use_first_row_as_header_mean_test_1 else ''
            context['test_type_mean_test_1'] = test_type_mean_test_1

            request.session['results_mean_test_1'] = results_mean_test_1
            request.session['data_mean_test_1'] = data_mean_test_1
            request.session['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            request.session['use_first_row_as_header_mean_test_1'] = use_first_row_as_header_mean_test_1
            request.session['test_type_mean_test_1'] = test_type_mean_test_1

            graph_mean_test_1 = create_unilateral_left_graph_mean_test_1(variable_names, z_scores, z_criticals)

            context['graph_mean_test_1'] = graph_mean_test_1
            request.session['graph_mean_test_1'] = graph_mean_test_1

            explanation_mean_test_1 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ < μ₀"
            context['explanation_mean_test_1'] = explanation_mean_test_1
            request.session['explanation_mean_test_1'] = explanation_mean_test_1
        
        elif test_type_mean_test_1 == "different":
            results_mean_test_1 = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                    
                n, mean, observed_mean, variance, alpha = valid_col[:5]
                std_error = np.sqrt(variance) / np.sqrt(n)
                z_score = (mean - observed_mean) / std_error
                z_critical = -1*(norm.ppf((100 - alpha) / 200))
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
                reject_null = "Yes" if abs(z_score) > z_critical else "No"
                variable_name = headers_mean_test_1[i] if headers_mean_test_1 and i < len(headers_mean_test_1) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_mean_test_1.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(variance),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, population variance, and significance level values."
                context['results_mean_test_1'] = None
                context['graph_mean_test_1'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, population variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_mean_test_1'] = results_mean_test_1
            context['data_mean_test_1'] = data_mean_test_1
            context['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            context['use_first_row_as_header_mean_test_1'] = 'checked' if use_first_row_as_header_mean_test_1 else ''
            context['test_type_mean_test_1'] = test_type_mean_test_1

            request.session['results_mean_test_1'] = results_mean_test_1
            request.session['data_mean_test_1'] = data_mean_test_1
            request.session['headers_mean_test_1'] = headers_mean_test_1 if use_first_row_as_header_mean_test_1 else None
            request.session['use_first_row_as_header_mean_test_1'] = use_first_row_as_header_mean_test_1
            request.session['test_type_mean_test_1'] = test_type_mean_test_1

            graph_mean_test_1 = create_bilateral_graph_mean_test_1(variable_names, z_scores, z_criticals)

            context['graph_mean_test_1'] = graph_mean_test_1
            request.session['graph_mean_test_1'] = graph_mean_test_1

            explanation_mean_test_1 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ ≠ μ₀"
            context['explanation_mean_test_1'] = explanation_mean_test_1
            request.session['explanation_mean_test_1'] = explanation_mean_test_1

##########################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_test" and subsubtab == "mean_test_2":
        data_mean_test_2 = request.POST.get('data_mean_test_2')
        use_first_row_as_header_mean_test_2 = request.POST.get('use_first_row_as_header_mean_test_2') == 'on'
        test_type_mean_test_2 = request.POST.get('test_type_mean_test_2')

        if not data_mean_test_2.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_test_2'] = None
            context['graph_mean_test_2'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_test_2 = data_mean_test_2.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_test_2.split('\n')]
        columns = []
        headers_mean_test_2 = []

        if use_first_row_as_header_mean_test_2 and rows:
            headers_mean_test_2 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_test_2:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_test_2'] = None
            context['results_mean_test_2'] = None
            return render(request, "inference/inference.html", context)

        if test_type_mean_test_2 == "greater":
            results_mean_test_2 = []
            t_scores = []
            t_criticals = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                    
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                
                n, mean, observed_mean, sample_variance, alpha = valid_col[:5]
                std_error = np.sqrt(sample_variance) / np.sqrt(n)
                t_score = (mean - observed_mean) / std_error
                t_critical = t.ppf(alpha / 100, df=n-1)
                p_value = 1 - t.cdf(t_score, df=n-1)
                reject_null = "Yes" if t_score > t_critical else "No"
                variable_name = headers_mean_test_2[i] if headers_mean_test_2 and i < len(headers_mean_test_2) else f"var. {i + 1}"
                t_scores.append(t_score)
                t_criticals.append(t_critical)
                n_vector.append(n)

                results_mean_test_2.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(sample_variance),
                    'alpha': float(alpha),
                    't_score': round(float(t_score), 4),
                    't_critical': round(float(t_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                
                valid_columns.append(i)
                variable_names.append(variable_name)
            
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_mean_test_2'] = None
                context['graph_mean_test_2'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
            
            context['results_mean_test_2'] = results_mean_test_2
            context['data_mean_test_2'] = data_mean_test_2
            context['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            context['use_first_row_as_header_mean_test_2'] = 'checked' if use_first_row_as_header_mean_test_2 else ''
            context['test_type_mean_test_2'] = test_type_mean_test_2

            request.session['results_mean_test_2'] = results_mean_test_2
            request.session['data_mean_test_2'] = data_mean_test_2
            request.session['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            request.session['use_first_row_as_header_mean_test_2'] = use_first_row_as_header_mean_test_2
            request.session['test_type_mean_test_2'] = test_type_mean_test_2

            graph_mean_test_2 = create_unilateral_rigth_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector)

            context['graph_mean_test_2'] = graph_mean_test_2
            request.session['graph_mean_test_2'] = graph_mean_test_2

            explanation_mean_test_2 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ > μ₀"
            context['explanation_mean_test_2'] = explanation_mean_test_2
            request.session['explanation_mean_test_2'] = explanation_mean_test_2

        elif test_type_mean_test_2 == "lesser":
            results_mean_test_2 = []
            t_scores = []
            t_criticals = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                    
                n, mean, observed_mean, sample_variance, alpha = valid_col[:5]
                std_error = np.sqrt(sample_variance) / np.sqrt(n)
                t_score = (mean - observed_mean) / std_error
                t_critical = t.ppf((1 - (alpha / 100)), df=n-1)
                p_value = t.cdf(t_score, df=n-1)
                reject_null = "Yes" if t_score < t_critical else "No"
                variable_name = headers_mean_test_2[i] if headers_mean_test_2 and i < len(headers_mean_test_2) else f"var. {i + 1}"
                t_scores.append(t_score)
                t_criticals.append(t_critical)
                n_vector.append(n)

                results_mean_test_2.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(sample_variance),
                    'alpha': float(alpha),
                    't_score': round(float(t_score), 4),
                    't_critical': round(float(t_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_mean_test_2'] = None
                context['graph_mean_test_2'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_mean_test_2'] = results_mean_test_2
            context['data_mean_test_2'] = data_mean_test_2
            context['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            context['use_first_row_as_header_mean_test_2'] = 'checked' if use_first_row_as_header_mean_test_2 else ''
            context['test_type_mean_test_2'] = test_type_mean_test_2

            request.session['results_mean_test_2'] = results_mean_test_2
            request.session['data_mean_test_2'] = data_mean_test_2
            request.session['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            request.session['use_first_row_as_header_mean_test_2'] = use_first_row_as_header_mean_test_2
            request.session['test_type_mean_test_2'] = test_type_mean_test_2

            graph_mean_test_2 = create_unilateral_left_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector)

            context['graph_mean_test_2'] = graph_mean_test_2
            request.session['graph_mean_test_2'] = graph_mean_test_2

            explanation_mean_test_2 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ < μ₀"
            context['explanation_mean_test_2'] = explanation_mean_test_2
            request.session['explanation_mean_test_2'] = explanation_mean_test_2
        
        elif test_type_mean_test_2 == "different":
            results_mean_test_2 = []
            t_scores = []
            t_criticals = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 5:
                    invalid_columns.append(i)
                    continue
                    
                n, mean, observed_mean, sample_variance, alpha = valid_col[:5]
                std_error = np.sqrt(sample_variance) / np.sqrt(n)
                t_score = (mean - observed_mean) / std_error
                t_critical = -1*(t.ppf(((100 - alpha) / 200), df=n-1))
                p_value = 2 * (1 - t.cdf(abs(t_score), df=n-1))
                reject_null = "Yes" if abs(t_score) > t_critical else "No"
                variable_name = headers_mean_test_2[i] if headers_mean_test_2 and i < len(headers_mean_test_2) else f"var. {i + 1}"
                t_scores.append(t_score)
                t_criticals.append(t_critical)
                n_vector.append(n)

                results_mean_test_2.append({
                    'variable': variable_name,
                    'mean': float(mean),
                    'observed_mean': float(observed_mean),
                    'n_elements': int(n),
                    'variance': float(sample_variance),
                    'alpha': float(alpha),
                    't_score': round(float(t_score), 4),
                    't_critical': round(float(t_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_mean_test_2'] = None
                context['graph_mean_test_2'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_mean_test_2'] = results_mean_test_2
            context['data_mean_test_2'] = data_mean_test_2
            context['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            context['use_first_row_as_header_mean_test_2'] = 'checked' if use_first_row_as_header_mean_test_2 else ''
            context['test_type_mean_test_2'] = test_type_mean_test_2

            request.session['results_mean_test_2'] = results_mean_test_2
            request.session['data_mean_test_2'] = data_mean_test_2
            request.session['headers_mean_test_2'] = headers_mean_test_2 if use_first_row_as_header_mean_test_2 else None
            request.session['use_first_row_as_header_mean_test_2'] = use_first_row_as_header_mean_test_2
            request.session['test_type_mean_test_2'] = test_type_mean_test_2

            graph_mean_test_2 = create_bilateral_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector)

            context['graph_mean_test_2'] = graph_mean_test_2
            request.session['graph_mean_test_2'] = graph_mean_test_2

            explanation_mean_test_2 = "One-Tailed Right Test - Null Hypothesis (H₀): μ = μ₀ - Alternative Hypothesis (H₁):  μ ≠ μ₀"
            context['explanation_mean_test_2'] = explanation_mean_test_2
            request.session['explanation_mean_test_2'] = explanation_mean_test_2

###################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "var_test":
        data_var_test = request.POST.get('data_var_test')
        use_first_row_as_header_var_test = request.POST.get('use_first_row_as_header_var_test') == 'on'
        test_type_var_test = request.POST.get('test_type_var_test')

        if not data_var_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_var_test'] = None
            context['graph_var_test'] = None
            return render(request, "inference/inference.html", context)
        
        data_var_test = data_var_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_var_test.split('\n')]
        columns = []
        headers_var_test = []

        if use_first_row_as_header_var_test and rows:
            headers_var_test = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_var_test:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_var_test'] = None
            context['results_var_test'] = None
            return render(request, "inference/inference.html", context)

        if test_type_var_test == "greater":
            results_var_test = []
            chi_scores = []
            chi_criticals = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                    
                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue
                
                n, sample_variance, population_variance, alpha = valid_col[:4]
                chi_score = (n - 1) * sample_variance / population_variance
                chi_critical = chi2.ppf(alpha / 100, df=n - 1)
                p_value = 1 - chi2.cdf(chi_score, df=n - 1)
                reject_null = "Yes" if chi_score > chi_critical else "No"
                variable_name = headers_var_test[i] if headers_var_test and i < len(headers_var_test) else f"var. {i + 1}"
                chi_scores.append(chi_score)
                chi_criticals.append(chi_critical)
                n_vector.append(n)

                results_var_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'sample_variance': float(sample_variance),
                    'population_variance': float(population_variance),
                    'alpha': float(alpha),
                    'chi_score': round(float(chi_score), 4),
                    'chi_critical': round(float(chi_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                
                valid_columns.append(i)
                variable_names.append(variable_name)
            
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_var_test'] = None
                context['graph_var_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
            
            context['results_var_test'] = results_var_test
            context['data_var_test'] = data_var_test
            context['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            context['use_first_row_as_header_var_test'] = 'checked' if use_first_row_as_header_var_test else ''
            context['test_type_var_test'] = test_type_var_test

            request.session['results_var_test'] = results_var_test
            request.session['data_var_test'] = data_var_test
            request.session['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            request.session['use_first_row_as_header_var_test'] = use_first_row_as_header_var_test
            request.session['test_type_var_test'] = test_type_var_test

            graph_var_test = create_unilateral_rigth_graph_var_test(variable_names, chi_scores, chi_criticals, n_vector)

            context['graph_var_test'] = graph_var_test
            request.session['graph_var_test'] = graph_var_test

            explanation_var_test = "One-Tailed Right Test - Null Hypothesis (H₀): σ² = σ₀² - Alternative Hypothesis (H₁): σ² > σ₀²"
            context['explanation_var_test'] = explanation_var_test
            request.session['explanation_var_test'] = explanation_var_test

        elif test_type_var_test == "lesser":
            results_var_test = []
            chi_scores = []
            chi_criticals = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue
                
                n, sample_variance, population_variance, alpha = valid_col[:4]
                chi_score = (n - 1) * sample_variance / population_variance
                chi_critical = chi2.ppf((100 - alpha) / 100, df=n - 1)
                p_value = chi2.cdf(chi_score, df=n - 1)
                reject_null = "Yes" if chi_score < chi_critical else "No"
                variable_name = headers_var_test[i] if headers_var_test and i < len(headers_var_test) else f"var. {i + 1}"
                chi_scores.append(chi_score)
                chi_criticals.append(chi_critical)
                n_vector.append(n)

                results_var_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'sample_variance': float(sample_variance),
                    'population_variance': float(population_variance),
                    'alpha': float(alpha),
                    'chi_score': round(float(chi_score), 4),
                    'chi_critical': round(float(chi_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_var_test'] = None
                context['graph_var_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_var_test'] = results_var_test
            context['data_var_test'] = data_var_test
            context['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            context['use_first_row_as_header_var_test'] = 'checked' if use_first_row_as_header_var_test else ''
            context['test_type_var_test'] = test_type_var_test

            request.session['results_var_test'] = results_var_test
            request.session['data_var_test'] = data_var_test
            request.session['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            request.session['use_first_row_as_header_var_test'] = use_first_row_as_header_var_test
            request.session['test_type_var_test'] = test_type_var_test

            graph_var_test = create_unilateral_left_graph_var_test(variable_names, chi_scores, chi_criticals, n_vector)

            context['graph_var_test'] = graph_var_test
            request.session['graph_var_test'] = graph_var_test

            explanation_var_test = "One-Tailed Right Test - Null Hypothesis (H₀): σ² = σ₀² - Alternative Hypothesis (H₁):  σ² < σ₀²"
            context['explanation_var_test'] = explanation_var_test
            request.session['explanation_var_test'] = explanation_var_test
        
        elif test_type_var_test == "different":
            results_var_test = []
            chi_scores = []
            chi_criticals_lower = []
            chi_criticals_upper = []
            variable_names = []
            n_vector = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]
                        
                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue
                    
                n, sample_variance, population_variance, alpha = valid_col[:4]
                chi_score = (n - 1) * sample_variance / population_variance
                chi_critical_lower = chi2.ppf((((100 - alpha)/100))/2, df=n - 1)
                chi_critical_upper = chi2.ppf(1-((((100 - alpha)/100)))/2, df=n - 1)
                p_value = 2 * min(chi2.cdf(chi_score, df=n - 1), 1 - chi2.cdf(chi_score, df=n - 1))
                reject_null = "Yes" if (chi_score < chi_critical_lower or chi_score > chi_critical_upper) else "No"
                variable_name = headers_var_test[i] if headers_var_test and i < len(headers_var_test) else f"var. {i + 1}"
                chi_scores.append(chi_score)
                chi_criticals_lower.append(chi_critical_lower)
                chi_criticals_upper.append(chi_critical_upper)
                n_vector.append(n)

                results_var_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'sample_variance': float(sample_variance),
                    'population_variance': float(population_variance),
                    'alpha': float(alpha),
                    'chi_score': round(float(chi_score), 4),
                    'chi_critical': [round(float(chi_critical_lower), 4), round(float(chi_critical_upper), 4)],
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })
                    
                valid_columns.append(i)
                variable_names.append(variable_name)
                
            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, mean, sample variance, and significance level values."
                context['results_var_test'] = None
                context['graph_var_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, mean, sample variance, and significance level values."
            else:
                context['warning'] = None
                
            context['results_var_test'] = results_var_test
            context['data_var_test'] = data_var_test
            context['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            context['use_first_row_as_header_var_test'] = 'checked' if use_first_row_as_header_var_test else ''
            context['test_type_var_test'] = test_type_var_test

            request.session['results_var_test'] = results_var_test
            request.session['data_var_test'] = data_var_test
            request.session['headers_var_test'] = headers_var_test if use_first_row_as_header_var_test else None
            request.session['use_first_row_as_header_var_test'] = use_first_row_as_header_var_test
            request.session['test_type_var_test'] = test_type_var_test
            graph_var_test = create_bilateral_graph_var_test(variable_names, chi_scores, chi_criticals_lower, chi_criticals_upper, n_vector)

            context['graph_var_test'] = graph_var_test
            request.session['graph_var_test'] = graph_var_test

            explanation_var_test = "One-Tailed Right Test - Null Hypothesis (H₀): σ² = σ₀² - Alternative Hypothesis (H₁):  σ² ≠ σ₀²"
            context['explanation_var_test'] = explanation_var_test
            request.session['explanation_var_test'] = explanation_var_test

#####################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_diff_test" and subsubtab == "mean_diff_test_1":
        data_mean_diff_test_1 = request.POST.get('data_mean_diff_test_1')
        use_first_row_as_header_mean_diff_test_1 = request.POST.get('use_first_row_as_header_mean_diff_test_1') == 'on'
        test_type_mean_diff_test_1 = request.POST.get('test_type_mean_diff_test_1')
        alpha_value_mean_diff_test_1 = request.POST.get('alpha_value_mean_diff_test_1')

        if alpha_value_mean_diff_test_1:
            try:
                alpha_value_mean_diff_test_1 = float(alpha_value_mean_diff_test_1)
            except ValueError:
                alpha_value_mean_diff_test_1 = None

        if not data_mean_diff_test_1.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_test_1'] = None
            context['graph_mean_diff_test_1'] = None
            return render(request, "inference/inference.html", context)

        data_mean_diff_test_1 = data_mean_diff_test_1.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_test_1.split('\n')]
        columns = []
        headers_mean_diff_test_1 = []

        if use_first_row_as_header_mean_diff_test_1 and rows:
            headers_mean_diff_test_1 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_test_1:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_test_1'] = None
            context['results_mean_diff_test_1'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_test_1:
            alpha_value_mean_diff_test_1 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if test_type_mean_diff_test_1 == "greater":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_1) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt(variance1 / n1 + variance2 / n2)
                z_stat = mean_difference / std_error_diff
                z_critical = norm.ppf(alpha)
                p_value = 1 - norm.cdf(z_stat)
                reject_null = 'Yes' if z_stat > z_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."

            variable_names = [headers_mean_diff_test_1[0],headers_mean_diff_test_1[1]] if headers_mean_diff_test_1 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_1 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_1,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                'z_stat': f"{z_stat:.4f}",
                'z_critical': f"{z_critical:.4f}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_1'] = results_mean_diff_test_1
            context['data_mean_diff_test_1'] = data_mean_diff_test_1
            context['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            context['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            context['use_first_row_as_header_mean_diff_test_1'] = 'checked' if use_first_row_as_header_mean_diff_test_1 else ''
            context['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            request.session['results_mean_diff_test_1'] = results_mean_diff_test_1
            request.session['data_mean_diff_test_1'] = data_mean_diff_test_1
            request.session['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            request.session['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            request.session['use_first_row_as_header_mean_diff_test_1'] = use_first_row_as_header_mean_diff_test_1
            request.session['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            graph_mean_diff_test_1 = create_right_graph_mean_diff_test_1(variable_names, z_stat, z_critical)
            context['graph_mean_diff_test_1'] = f'data:image/png;base64,{graph_mean_diff_test_1}'
            request.session['graph_mean_diff_test_1'] = context['graph_mean_diff_test_1']
            explanation_mean_diff_test_1 = "One-Tailed Right Test - Null Hypothesis (H₀): μ₁ ≤ μ₂ - Alternative Hypothesis (H₁): μ₁ > μ₂"
            context['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1
            request.session['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1
        
        elif test_type_mean_diff_test_1 == "lesser":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_1) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt(variance1 / n1 + variance2 / n2)
                z_stat = mean_difference / std_error_diff
                z_critical = norm.ppf(1-alpha)
                p_value = norm.cdf(z_stat)
                reject_null = 'Yes' if z_stat < z_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_1[0],headers_mean_diff_test_1[1]] if headers_mean_diff_test_1 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_1 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_1,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                'z_stat': f"{z_stat:.4f}",
                'z_critical': f"{z_critical:.4f}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_1'] = results_mean_diff_test_1
            context['data_mean_diff_test_1'] = data_mean_diff_test_1
            context['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            context['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            context['use_first_row_as_header_mean_diff_test_1'] = 'checked' if use_first_row_as_header_mean_diff_test_1 else ''
            context['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            request.session['results_mean_diff_test_1'] = results_mean_diff_test_1
            request.session['data_mean_diff_test_1'] = data_mean_diff_test_1
            request.session['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            request.session['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            request.session['use_first_row_as_header_mean_diff_test_1'] = use_first_row_as_header_mean_diff_test_1
            request.session['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            graph_mean_diff_test_1 = create_left_graph_mean_diff_test_1(variable_names, z_stat, z_critical)
            context['graph_mean_diff_test_1'] = f'data:image/png;base64,{graph_mean_diff_test_1}'
            request.session['graph_mean_diff_test_1'] = context['graph_mean_diff_test_1']
            explanation_mean_diff_test_1 = "One-Tailed Left Test - Null Hypothesis (H₀): μ₁ ≥ μ₂ - Alternative Hypothesis (H₁): μ₁ < μ₂"
            context['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1
            request.session['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1

        elif test_type_mean_diff_test_1 == "different":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_1) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt(variance1 / n1 + variance2 / n2)
                z_stat = mean_difference / std_error_diff
                z_critical = -1*norm.ppf((1 - alpha) / 2)
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))
                reject_null = 'Yes' if abs(z_stat) > z_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_1[0],headers_mean_diff_test_1[1]] if headers_mean_diff_test_1 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_1 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_1,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                'z_stat': f"{z_stat:.4f}",
                'z_critical': f"{z_critical:.4f}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_1'] = results_mean_diff_test_1
            context['data_mean_diff_test_1'] = data_mean_diff_test_1
            context['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            context['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            context['use_first_row_as_header_mean_diff_test_1'] = 'checked' if use_first_row_as_header_mean_diff_test_1 else ''
            context['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            request.session['results_mean_diff_test_1'] = results_mean_diff_test_1
            request.session['data_mean_diff_test_1'] = data_mean_diff_test_1
            request.session['alpha_value_mean_diff_test_1'] = alpha_value_mean_diff_test_1
            request.session['headers_mean_diff_test_1'] = headers_mean_diff_test_1 if use_first_row_as_header_mean_diff_test_1 else None
            request.session['use_first_row_as_header_mean_diff_test_1'] = use_first_row_as_header_mean_diff_test_1
            request.session['test_type_mean_diff_test_1'] = test_type_mean_diff_test_1
            graph_mean_diff_test_1 = create_bilateral_graph_mean_diff_test_1(variable_names, z_stat, z_critical)
            context['graph_mean_diff_test_1'] = f'data:image/png;base64,{graph_mean_diff_test_1}'
            request.session['graph_mean_diff_test_1'] = context['graph_mean_diff_test_1']
            explanation_mean_diff_test_1 = "Two-Tailed Test - Null Hypothesis (H₀): μ₁ = μ₂ - Alternative Hypothesis (H₁): μ₁ ≠ μ₂"
            context['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1
            request.session['explanation_mean_diff_test_1'] = explanation_mean_diff_test_1

#####################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_diff_test" and subsubtab == "mean_diff_test_2":
        data_mean_diff_test_2 = request.POST.get('data_mean_diff_test_2')
        use_first_row_as_header_mean_diff_test_2 = request.POST.get('use_first_row_as_header_mean_diff_test_2') == 'on'
        test_type_mean_diff_test_2 = request.POST.get('test_type_mean_diff_test_2')
        alpha_value_mean_diff_test_2 = request.POST.get('alpha_value_mean_diff_test_2')

        if alpha_value_mean_diff_test_2:
            try:
                alpha_value_mean_diff_test_2 = float(alpha_value_mean_diff_test_2)
            except ValueError:
                alpha_value_mean_diff_test_2 = None

        if not data_mean_diff_test_2.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_test_2'] = None
            context['graph_mean_diff_test_2'] = None
            return render(request, "inference/inference.html", context)

        data_mean_diff_test_2 = data_mean_diff_test_2.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_test_2.split('\n')]
        columns = []
        headers_mean_diff_test_2 = []

        if use_first_row_as_header_mean_diff_test_2 and rows:
            headers_mean_diff_test_2 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_test_2:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_test_2'] = None
            context['results_mean_diff_test_2'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_test_2:
            alpha_value_mean_diff_test_2 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if test_type_mean_diff_test_2 == "greater":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_2) / 100)
                pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
                df = n1 + n2 - 2
                t_stat = mean_difference / std_error_diff
                t_critical = t.ppf(alpha, df)
                p_value = 1 - t.cdf(t_stat, df)
                reject_null = 'Yes' if t_stat > t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."

            variable_names = [headers_mean_diff_test_2[0],headers_mean_diff_test_2[1]] if headers_mean_diff_test_2 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_2 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_2,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_2'] = results_mean_diff_test_2
            context['data_mean_diff_test_2'] = data_mean_diff_test_2
            context['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            context['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            context['use_first_row_as_header_mean_diff_test_2'] = 'checked' if use_first_row_as_header_mean_diff_test_2 else ''
            context['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            request.session['results_mean_diff_test_2'] = results_mean_diff_test_2
            request.session['data_mean_diff_test_2'] = data_mean_diff_test_2
            request.session['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            request.session['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            request.session['use_first_row_as_header_mean_diff_test_2'] = use_first_row_as_header_mean_diff_test_2
            request.session['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            graph_mean_diff_test_2 = create_right_graph_mean_diff_test_2(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_2'] = f'data:image/png;base64,{graph_mean_diff_test_2}'
            request.session['graph_mean_diff_test_2'] = context['graph_mean_diff_test_2']
            explanation_mean_diff_test_2 = "One-Tailed Right Test - Null Hypothesis (H₀): μ₁ ≤ μ₂ - Alternative Hypothesis (H₁): μ₁ > μ₂"
            context['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2
            request.session['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2
        
        elif test_type_mean_diff_test_2 == "lesser":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_2) / 100)
                pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
                std_error_diff = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
                mean_difference = mean1 - mean2
                t_stat = mean_difference / std_error_diff
                df = n1 + n2 - 2
                t_critical = t.ppf(1 - alpha, df)
                p_value = t.cdf(t_stat, df)
                reject_null = 'Yes' if t_stat < t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_2[0],headers_mean_diff_test_2[1]] if headers_mean_diff_test_2 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_2 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_2,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_2'] = results_mean_diff_test_2
            context['data_mean_diff_test_2'] = data_mean_diff_test_2
            context['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            context['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            context['use_first_row_as_header_mean_diff_test_2'] = 'checked' if use_first_row_as_header_mean_diff_test_2 else ''
            context['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            request.session['results_mean_diff_test_2'] = results_mean_diff_test_2
            request.session['data_mean_diff_test_2'] = data_mean_diff_test_2
            request.session['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            request.session['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            request.session['use_first_row_as_header_mean_diff_test_2'] = use_first_row_as_header_mean_diff_test_2
            request.session['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            graph_mean_diff_test_2 = create_left_graph_mean_diff_test_2(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_2'] = f'data:image/png;base64,{graph_mean_diff_test_2}'
            request.session['graph_mean_diff_test_2'] = context['graph_mean_diff_test_2']
            explanation_mean_diff_test_2 = "One-Tailed Left Test - Null Hypothesis (H₀): μ₁ ≥ μ₂ - Alternative Hypothesis (H₁): μ₁ < μ₂"
            context['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2
            request.session['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2

        elif test_type_mean_diff_test_2 == "different":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_2) / 100)
                pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
                std_error_diff = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
                mean_difference = mean1 - mean2
                t_stat = mean_difference / std_error_diff
                df = n1 + n2 - 2
                t_critical = -1*(t.ppf((1 - alpha) / 2, df))
                t_critical_negative = t.ppf((1 - alpha) / 2, df)
                p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                reject_null = 'Yes' if t_stat < t_critical_negative or t_stat > t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_2[0],headers_mean_diff_test_2[1]] if headers_mean_diff_test_2 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_2 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_2,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical_negative:.4f}, {t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_2'] = results_mean_diff_test_2
            context['data_mean_diff_test_2'] = data_mean_diff_test_2
            context['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            context['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            context['use_first_row_as_header_mean_diff_test_2'] = 'checked' if use_first_row_as_header_mean_diff_test_2 else ''
            context['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            request.session['results_mean_diff_test_2'] = results_mean_diff_test_2
            request.session['data_mean_diff_test_2'] = data_mean_diff_test_2
            request.session['alpha_value_mean_diff_test_2'] = alpha_value_mean_diff_test_2
            request.session['headers_mean_diff_test_2'] = headers_mean_diff_test_2 if use_first_row_as_header_mean_diff_test_2 else None
            request.session['use_first_row_as_header_mean_diff_test_2'] = use_first_row_as_header_mean_diff_test_2
            request.session['test_type_mean_diff_test_2'] = test_type_mean_diff_test_2
            graph_mean_diff_test_2 = create_bilateral_graph_mean_diff_test_2(variable_names, t_stat, t_critical, t_critical_negative, df)
            context['graph_mean_diff_test_2'] = f'data:image/png;base64,{graph_mean_diff_test_2}'
            request.session['graph_mean_diff_test_2'] = context['graph_mean_diff_test_2']
            explanation_mean_diff_test_2 = "Two-Tailed Test - Null Hypothesis (H₀): μ₁ = μ₂ - Alternative Hypothesis (H₁): μ₁ ≠ μ₂"
            context['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2
            request.session['explanation_mean_diff_test_2'] = explanation_mean_diff_test_2

#####################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_diff_test" and subsubtab == "mean_diff_test_3":
        data_mean_diff_test_3 = request.POST.get('data_mean_diff_test_3')
        use_first_row_as_header_mean_diff_test_3 = request.POST.get('use_first_row_as_header_mean_diff_test_3') == 'on'
        test_type_mean_diff_test_3 = request.POST.get('test_type_mean_diff_test_3')
        alpha_value_mean_diff_test_3 = request.POST.get('alpha_value_mean_diff_test_3')

        if alpha_value_mean_diff_test_3:
            try:
                alpha_value_mean_diff_test_3 = float(alpha_value_mean_diff_test_3)
            except ValueError:
                alpha_value_mean_diff_test_3 = None

        if not data_mean_diff_test_3.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_test_3'] = None
            context['graph_mean_diff_test_3'] = None
            return render(request, "inference/inference.html", context)

        data_mean_diff_test_3 = data_mean_diff_test_3.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_test_3.split('\n')]
        columns = []
        headers_mean_diff_test_3 = []

        if use_first_row_as_header_mean_diff_test_3 and rows:
            headers_mean_diff_test_3 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_test_3:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_test_3'] = None
            context['results_mean_diff_test_3'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_test_3:
            alpha_value_mean_diff_test_3 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."
        
        if test_type_mean_diff_test_3 == "greater":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_3) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt((variance1 / n1) + (variance2 / n2))
                df = ((variance1 / n1 + variance2 / n2) ** 2) / (((variance1 / n1) ** 2) / (n1 - 1) + ((variance2 / n2) ** 2) / (n2 - 1))
                t_stat = mean_difference / std_error_diff
                t_critical = t.ppf(alpha, df)
                p_value = 1 - t.cdf(t_stat, df)
                reject_null = 'Yes' if t_stat > t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."

            variable_names = [headers_mean_diff_test_3[0],headers_mean_diff_test_3[1]] if headers_mean_diff_test_3 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_3 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_3,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_3'] = results_mean_diff_test_3
            context['data_mean_diff_test_3'] = data_mean_diff_test_3
            context['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            context['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            context['use_first_row_as_header_mean_diff_test_3'] = 'checked' if use_first_row_as_header_mean_diff_test_3 else ''
            context['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            request.session['results_mean_diff_test_3'] = results_mean_diff_test_3
            request.session['data_mean_diff_test_3'] = data_mean_diff_test_3
            request.session['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            request.session['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            request.session['use_first_row_as_header_mean_diff_test_3'] = use_first_row_as_header_mean_diff_test_3
            request.session['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            graph_mean_diff_test_3 = create_right_graph_mean_diff_test_3(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_3'] = f'data:image/png;base64,{graph_mean_diff_test_3}'
            request.session['graph_mean_diff_test_3'] = context['graph_mean_diff_test_3']
            explanation_mean_diff_test_3 = "One-Tailed Right Test - Null Hypothesis (H₀): μ₁ ≤ μ₂ - Alternative Hypothesis (H₁): μ₁ > μ₂"
            context['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3
            request.session['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3
        
        elif test_type_mean_diff_test_3 == "lesser":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_3) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt((variance1 / n1) + (variance2 / n2))
                df = ((variance1 / n1 + variance2 / n2) ** 2) / (((variance1 / n1) ** 2) / (n1 - 1) + ((variance2 / n2) ** 2) / (n2 - 1))
                t_stat = mean_difference / std_error_diff
                t_critical = t.ppf(1 - alpha, df)
                p_value = t.cdf(t_stat, df)
                reject_null = 'Yes' if t_stat < t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_3[0],headers_mean_diff_test_3[1]] if headers_mean_diff_test_3 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_3 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_3,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_3'] = results_mean_diff_test_3
            context['data_mean_diff_test_3'] = data_mean_diff_test_3
            context['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            context['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            context['use_first_row_as_header_mean_diff_test_3'] = 'checked' if use_first_row_as_header_mean_diff_test_3 else ''
            context['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            request.session['results_mean_diff_test_3'] = results_mean_diff_test_3
            request.session['data_mean_diff_test_3'] = data_mean_diff_test_3
            request.session['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            request.session['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            request.session['use_first_row_as_header_mean_diff_test_3'] = use_first_row_as_header_mean_diff_test_3
            request.session['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            graph_mean_diff_test_3 = create_left_graph_mean_diff_test_3(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_3'] = f'data:image/png;base64,{graph_mean_diff_test_3}'
            request.session['graph_mean_diff_test_3'] = context['graph_mean_diff_test_3']
            explanation_mean_diff_test_3 = "One-Tailed Left Test - Null Hypothesis (H₀): μ₁ ≥ μ₂ - Alternative Hypothesis (H₁): μ₁ < μ₂"
            context['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3
            request.session['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3

        elif test_type_mean_diff_test_3 == "different":
            if len(data_columns) == 2 and all(len(row) == 3 for row in data_columns):
                n1 = int(data_columns[0][0])
                mean1 = float(data_columns[0][1])
                variance1 = float(data_columns[0][2])
                n2 = int(data_columns[1][0])
                mean2 = float(data_columns[1][1])
                variance2 = float(data_columns[1][2])
                alpha = (float(alpha_value_mean_diff_test_3) / 100)
                mean_difference = mean1 - mean2
                std_error_diff = np.sqrt((variance1 / n1) + (variance2 / n2))
                df = ((variance1 / n1 + variance2 / n2) ** 2) / (((variance1 / n1) ** 2) / (n1 - 1) + ((variance2 / n2) ** 2) / (n2 - 1))
                t_stat = mean_difference / std_error_diff
                t_critical = -1*(t.ppf((1 - alpha) / 2, df))
                t_critical_negative = t.ppf((1 - alpha) / 2, df)
                p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                reject_null = 'Yes' if t_stat < t_critical_negative or t_stat > t_critical else 'No'
            
            else:
                context['error'] = "The data entered is incorrect."
            
            variable_names = [headers_mean_diff_test_3[0],headers_mean_diff_test_3[1]] if headers_mean_diff_test_3 else ["var. 1", "var. 2"]
            
            results_mean_diff_test_3 = [{
                'variable': f"{variable_names[0]}, {variable_names[1]}",
                'n_elements': f"{n1}, {n2}",
                'variance': f"{variance1}, {variance2}",
                'alpha': alpha_value_mean_diff_test_3,
                'mean': f"{mean1}, {mean2}",
                'mean_diff': f"{mean_difference:.4f}",
                't_stat': f"{t_stat:.4f}",
                't_critical': f"{t_critical_negative:.4f}, {t_critical:.4f}",
                'df': f"{df}",
                'p_value': f"{p_value:.4f}",
                'reject_null': reject_null,
            }]
            
            context['results_mean_diff_test_3'] = results_mean_diff_test_3
            context['data_mean_diff_test_3'] = data_mean_diff_test_3
            context['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            context['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            context['use_first_row_as_header_mean_diff_test_3'] = 'checked' if use_first_row_as_header_mean_diff_test_3 else ''
            context['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            request.session['results_mean_diff_test_3'] = results_mean_diff_test_3
            request.session['data_mean_diff_test_3'] = data_mean_diff_test_3
            request.session['alpha_value_mean_diff_test_3'] = alpha_value_mean_diff_test_3
            request.session['headers_mean_diff_test_3'] = headers_mean_diff_test_3 if use_first_row_as_header_mean_diff_test_3 else None
            request.session['use_first_row_as_header_mean_diff_test_3'] = use_first_row_as_header_mean_diff_test_3
            request.session['test_type_mean_diff_test_3'] = test_type_mean_diff_test_3
            graph_mean_diff_test_3 = create_bilateral_graph_mean_diff_test_3(variable_names, t_stat, t_critical, t_critical_negative, df)
            context['graph_mean_diff_test_3'] = f'data:image/png;base64,{graph_mean_diff_test_3}'
            request.session['graph_mean_diff_test_3'] = context['graph_mean_diff_test_3']
            explanation_mean_diff_test_3 = "Two-Tailed Test - Null Hypothesis (H₀): μ₁ = μ₂ - Alternative Hypothesis (H₁): μ₁ ≠ μ₂"
            context['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3
            request.session['explanation_mean_diff_test_3'] = explanation_mean_diff_test_3

#####################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "mean_diff_test" and subsubtab == "mean_diff_test_4":
        data_mean_diff_test_4 = request.POST.get('data_mean_diff_test_4')
        use_first_row_as_header_mean_diff_test_4 = request.POST.get('use_first_row_as_header_mean_diff_test_4') == 'on'
        test_type_mean_diff_test_4 = request.POST.get('test_type_mean_diff_test_4')
        alpha_value_mean_diff_test_4 = request.POST.get('alpha_value_mean_diff_test_4')

        if alpha_value_mean_diff_test_4:
            try:
                alpha_value_mean_diff_test_4 = float(alpha_value_mean_diff_test_4)
            except ValueError:
                alpha_value_mean_diff_test_4 = None

        if not data_mean_diff_test_4.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mean_diff_test_4'] = None
            context['graph_mean_diff_test_4'] = None
            return render(request, "inference/inference.html", context)
        
        data_mean_diff_test_4 = data_mean_diff_test_4.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_mean_diff_test_4.split('\n')]
        columns = []
        headers_mean_diff_test_4 = []

        if use_first_row_as_header_mean_diff_test_4 and rows:
            headers_mean_diff_test_4 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_mean_diff_test_4:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_mean_diff_test_4'] = None
            context['results_mean_diff_test_4'] = None
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_mean_diff_test_4:
            alpha_value_mean_diff_test_4 = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."

        if test_type_mean_diff_test_4 == "greater":
            if len(data_columns) == 2 and len(data_columns[0]) == len(data_columns[1]):
                data_1 = np.array([float(value) for value in data_columns[0]])
                data_2 = np.array([float(value) for value in data_columns[1]])

                if len(data_1) > 0:
                    differences = data_1 - data_2
                    n = len(differences)
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences, ddof=1)
                    std_error_diff = std_diff / np.sqrt(n)
                    df = n - 1
                    alpha = (float(alpha_value_mean_diff_test_4) / 100)
                    t_stat = mean_diff / std_error_diff
                    t_critical = t.ppf(alpha, df)
                    p_value = 1 - t.cdf(t_stat, df)
                    reject_null = 'Yes' if t_stat > t_critical else 'No'

                    variable_names = [headers_mean_diff_test_4[0],headers_mean_diff_test_4[1]] if headers_mean_diff_test_4 else ["var. 1", "var. 2"]
            
                    results_mean_diff_test_4 = [{
                        'variable': f"{variable_names[0]}, {variable_names[1]}",
                        'n_elements': f"{n}",
                        'alpha': alpha_value_mean_diff_test_4,
                        'mean_diff': f"{mean_diff:.4f}",
                        't_stat': f"{t_stat:.4f}",
                        't_critical': f"{t_critical:.4f}",
                        'df': f"{df}",
                        'p_value': f"{p_value:.4f}",
                        'reject_null': reject_null,
                    }]

                else:
                    context['error'] = "The data entered is incorrect."
            else:
                context['error'] = "The data entered is incorrect."
            
            context['results_mean_diff_test_4'] = results_mean_diff_test_4
            context['data_mean_diff_test_4'] = data_mean_diff_test_4
            context['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            context['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            context['use_first_row_as_header_mean_diff_test_4'] = 'checked' if use_first_row_as_header_mean_diff_test_4 else ''
            context['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            request.session['results_mean_diff_test_4'] = results_mean_diff_test_4
            request.session['data_mean_diff_test_4'] = data_mean_diff_test_4
            request.session['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            request.session['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            request.session['use_first_row_as_header_mean_diff_test_4'] = use_first_row_as_header_mean_diff_test_4
            request.session['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            graph_mean_diff_test_4 = create_right_graph_mean_diff_test_4(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_4'] = f'data:image/png;base64,{graph_mean_diff_test_4}'
            request.session['graph_mean_diff_test_4'] = context['graph_mean_diff_test_4']
            explanation_mean_diff_test_4 = "One-Tailed Right Test - Null Hypothesis (H₀): μd ≤ 0 - Alternative Hypothesis (H₁): μd > 0"
            context['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4
            request.session['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4

        elif test_type_mean_diff_test_4 == "lesser":
            if len(data_columns) == 2 and len(data_columns[0]) == len(data_columns[1]):
                data_1 = np.array([float(value) for value in data_columns[0]])
                data_2 = np.array([float(value) for value in data_columns[1]])

                if len(data_1) > 0:
                    differences = data_1 - data_2
                    n = len(differences)
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences, ddof=1)
                    std_error_diff = std_diff / np.sqrt(n)
                    df = n - 1
                    alpha = (float(alpha_value_mean_diff_test_4) / 100)
                    t_stat = mean_diff / std_error_diff
                    t_critical = t.ppf(1 - alpha, df)
                    p_value = t.cdf(t_stat, df)
                    reject_null = 'Yes' if t_stat < t_critical else 'No'
                    variable_names = [headers_mean_diff_test_4[0],headers_mean_diff_test_4[1]] if headers_mean_diff_test_4 else ["var. 1", "var. 2"]
            
                    results_mean_diff_test_4 = [{
                        'variable': f"{variable_names[0]}, {variable_names[1]}",
                        'n_elements': f"{n}",
                        'alpha': alpha_value_mean_diff_test_4,
                        'mean_diff': f"{mean_diff:.4f}",
                        't_stat': f"{t_stat:.4f}",
                        't_critical': f"{t_critical:.4f}",
                        'df': f"{df}",
                        'p_value': f"{p_value:.4f}",
                        'reject_null': reject_null,
                    }]

                else:
                    context['error'] = "The data entered is incorrect."
            else:
                context['error'] = "The data entered is incorrect."
            
            context['results_mean_diff_test_4'] = results_mean_diff_test_4
            context['data_mean_diff_test_4'] = data_mean_diff_test_4
            context['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            context['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            context['use_first_row_as_header_mean_diff_test_4'] = 'checked' if use_first_row_as_header_mean_diff_test_4 else ''
            context['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            request.session['results_mean_diff_test_4'] = results_mean_diff_test_4
            request.session['data_mean_diff_test_4'] = data_mean_diff_test_4
            request.session['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            request.session['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            request.session['use_first_row_as_header_mean_diff_test_4'] = use_first_row_as_header_mean_diff_test_4
            request.session['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            graph_mean_diff_test_4 = create_left_graph_mean_diff_test_4(variable_names, t_stat, t_critical, df)
            context['graph_mean_diff_test_4'] = f'data:image/png;base64,{graph_mean_diff_test_4}'
            request.session['graph_mean_diff_test_4'] = context['graph_mean_diff_test_4']
            explanation_mean_diff_test_4 = "One-Tailed Left Test - Null Hypothesis (H₀): μd ≥ 0 - Alternative Hypothesis (H₁): μd < 0"
            context['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4
            request.session['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4

        elif test_type_mean_diff_test_4 == "different":
            if len(data_columns) == 2 and len(data_columns[0]) == len(data_columns[1]):
                data_1 = np.array([float(value) for value in data_columns[0]])
                data_2 = np.array([float(value) for value in data_columns[1]])

                if len(data_1) > 0:
                    differences = data_1 - data_2
                    n = len(differences)
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences, ddof=1)
                    std_error_diff = std_diff / np.sqrt(n)
                    df = n - 1
                    alpha = (float(alpha_value_mean_diff_test_4) / 100)
                    t_stat = mean_diff / std_error_diff
                    t_critical = -1*(t.ppf((1 - alpha) / 2, df))
                    t_critical_negative = t.ppf((1 - alpha) / 2, df)
                    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                    reject_null = 'Yes' if t_stat < t_critical_negative or t_stat > t_critical else 'No'
                    variable_names = [headers_mean_diff_test_4[0],headers_mean_diff_test_4[1]] if headers_mean_diff_test_4 else ["var. 1", "var. 2"]
            
                    results_mean_diff_test_4 = [{
                        'variable': f"{variable_names[0]}, {variable_names[1]}",
                        'n_elements': f"{n}",
                        'alpha': alpha_value_mean_diff_test_4,
                        'mean_diff': f"{mean_diff:.4f}",
                        't_stat': f"{t_stat:.4f}",
                        't_critical': f"{t_critical:.4f}",
                        'df': f"{df}",
                        'p_value': f"{p_value:.4f}",
                        'reject_null': reject_null,
                    }]

                else:
                    context['error'] = "The data entered is incorrect."
            else:
                context['error'] = "The data entered is incorrect."
            
            context['results_mean_diff_test_4'] = results_mean_diff_test_4
            context['data_mean_diff_test_4'] = data_mean_diff_test_4
            context['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            context['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            context['use_first_row_as_header_mean_diff_test_4'] = 'checked' if use_first_row_as_header_mean_diff_test_4 else ''
            context['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            request.session['results_mean_diff_test_4'] = results_mean_diff_test_4
            request.session['data_mean_diff_test_4'] = data_mean_diff_test_4
            request.session['alpha_value_mean_diff_test_4'] = alpha_value_mean_diff_test_4
            request.session['headers_mean_diff_test_4'] = headers_mean_diff_test_4 if use_first_row_as_header_mean_diff_test_4 else None
            request.session['use_first_row_as_header_mean_diff_test_4'] = use_first_row_as_header_mean_diff_test_4
            request.session['test_type_mean_diff_test_4'] = test_type_mean_diff_test_4
            graph_mean_diff_test_4 = create_bilateral_graph_mean_diff_test_4(variable_names, t_stat, t_critical, t_critical_negative, df)
            context['graph_mean_diff_test_4'] = f'data:image/png;base64,{graph_mean_diff_test_4}'
            request.session['graph_mean_diff_test_4'] = context['graph_mean_diff_test_4']
            explanation_mean_diff_test_4 = "Two-Tailed Test - Null Hypothesis (H₀): μd = 0 - Alternative Hypothesis (H₁): μd ≠ 0"
            context['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4
            request.session['explanation_mean_diff_test_4'] = explanation_mean_diff_test_4

################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "prop_test":
        data_prop_test = request.POST.get('data_prop_test')
        use_first_row_as_header_prop_test = request.POST.get('use_first_row_as_header_prop_test') == 'on'
        test_type_prop_test = request.POST.get('test_type_prop_test')

        if not data_prop_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_prop_test'] = None
            context['graph_prop_test'] = None
            return render(request, "inference/inference.html", context)

        data_prop_test = data_prop_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_prop_test.split('\n')]
        columns = []
        headers_prop_test = []

        if use_first_row_as_header_prop_test and rows:
            headers_prop_test = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_prop_test:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_prop_test'] = None
            context['results_prop_test'] = None
            return render(request, "inference/inference.html", context)
        
        if test_type_prop_test == "greater":
            results_prop_test = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]

                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue

                n, p_0, p_observed, alpha = valid_col[:4]
                std_error = np.sqrt(p_0 * (1 - p_0) / n)
                z_score = (p_observed - p_0) / std_error
                z_critical = norm.ppf(alpha / 100)
                p_value = 1 - norm.cdf(z_score)
                reject_null = "Yes" if z_score > z_critical else "No"
                variable_name = headers_prop_test[i] if headers_prop_test and i < len(headers_prop_test) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_prop_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'p_0': float(p_0),
                    'p_observed': float(p_observed),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })

                valid_columns.append(i)
                variable_names.append(variable_name)

            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, p_0, observed p, and significance level values."
                context['results_prop_test'] = None
                context['graph_prop_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, p_0, observed p, and significance level values."
            else:
                context['warning'] = None

            context['results_prop_test'] = results_prop_test
            context['data_prop_test'] = data_prop_test
            context['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            context['use_first_row_as_header_prop_test'] = 'checked' if use_first_row_as_header_prop_test else ''
            context['test_type_prop_test'] = test_type_prop_test
            request.session['results_prop_test'] = results_prop_test
            request.session['data_prop_test'] = data_prop_test
            request.session['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            request.session['use_first_row_as_header_prop_test'] = use_first_row_as_header_prop_test
            request.session['test_type_prop_test'] = test_type_prop_test
            graph_prop_test = create_unilateral_rigth_graph_prop_test(variable_names, z_scores, z_criticals)
            context['graph_prop_test'] = graph_prop_test
            request.session['graph_prop_test'] = graph_prop_test
            explanation_prop_test = "One-Tailed Right Test - Null Hypothesis (H₀): p = p₀ - Alternative Hypothesis (H₁): p > p₀"
            context['explanation_prop_test'] = explanation_prop_test
            request.session['explanation_prop_test'] = explanation_prop_test

        elif test_type_prop_test == "lesser":
            results_prop_test = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]

                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue

                n, p_0, p_observed, alpha = valid_col[:4]
                std_error = np.sqrt(p_0 * (1 - p_0) / n)
                z_score = (p_observed - p_0) / std_error
                z_critical = norm.ppf(1 - alpha / 100)
                p_value = norm.cdf(z_score)
                reject_null = "Yes" if z_score < z_critical else "No"
                variable_name = headers_prop_test[i] if headers_prop_test and i < len(headers_prop_test) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_prop_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'p_0': float(p_0),
                    'p_observed': float(p_observed),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })

                valid_columns.append(i)
                variable_names.append(variable_name)

            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, p_0, observed p, and significance level values."
                context['results_prop_test'] = None
                context['graph_prop_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, p_0, observed p, and significance level values."
            else:
                context['warning'] = None

            context['results_prop_test'] = results_prop_test
            context['data_prop_test'] = data_prop_test
            context['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            context['use_first_row_as_header_prop_test'] = 'checked' if use_first_row_as_header_prop_test else ''
            context['test_type_prop_test'] = test_type_prop_test
            request.session['results_prop_test'] = results_prop_test
            request.session['data_prop_test'] = data_prop_test
            request.session['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            request.session['use_first_row_as_header_prop_test'] = use_first_row_as_header_prop_test
            request.session['test_type_prop_test'] = test_type_prop_test
            graph_prop_test = create_unilateral_left_graph_prop_test(variable_names, z_scores, z_criticals)
            context['graph_prop_test'] = graph_prop_test
            request.session['graph_prop_test'] = graph_prop_test
            explanation_prop_test = "One-Tailed Left Test - Null Hypothesis (H₀): p = p₀ - Alternative Hypothesis (H₁): p < p₀"
            context['explanation_prop_test'] = explanation_prop_test
            request.session['explanation_prop_test'] = explanation_prop_test

        elif test_type_prop_test == "different":
            results_prop_test = []
            z_scores = []
            z_criticals = []
            variable_names = []
            valid_columns = []
            invalid_columns = []
            for i, col in enumerate(data_columns):
                valid_col = col[~np.isnan(col)]

                if len(valid_col) < 4:
                    invalid_columns.append(i)
                    continue

                n, p_0, p_observed, alpha = valid_col[:4]
                std_error = np.sqrt(p_0 * (1 - p_0) / n)
                z_score = (p_observed - p_0) / std_error
                z_critical = -1*(norm.ppf((100 - alpha) / 200))
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
                reject_null = "Yes" if abs(z_score) > z_critical else "No"
                variable_name = headers_prop_test[i] if headers_prop_test and i < len(headers_prop_test) else f"var. {i + 1}"
                z_scores.append(z_score)
                z_criticals.append(z_critical)

                results_prop_test.append({
                    'variable': variable_name,
                    'n_elements': int(n),
                    'p_0': float(p_0),
                    'p_observed': float(p_observed),
                    'alpha': float(alpha),
                    'z_score': round(float(z_score), 4),
                    'z_critical': round(float(z_critical), 4),
                    'p_value': round(p_value, 4),
                    'reject_null': reject_null,
                })

                valid_columns.append(i)
                variable_names.append(variable_name)

            if not valid_columns:
                context['error'] = "Error: Incomplete data. Please ensure each column has n, p_0, observed p, and significance level values."
                context['results_prop_test'] = None
                context['graph_prop_test'] = None
                return render(request, "inference/inference.html", context)

            if invalid_columns:
                context['warning'] = f"Some variables were excluded due to missing data: Please ensure each column has n, p_0, observed p, and significance level values."
            else:
                context['warning'] = None

            context['results_prop_test'] = results_prop_test
            context['data_prop_test'] = data_prop_test
            context['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            context['use_first_row_as_header_prop_test'] = 'checked' if use_first_row_as_header_prop_test else ''
            context['test_type_prop_test'] = test_type_prop_test
            request.session['results_prop_test'] = results_prop_test
            request.session['data_prop_test'] = data_prop_test
            request.session['headers_prop_test'] = headers_prop_test if use_first_row_as_header_prop_test else None
            request.session['use_first_row_as_header_prop_test'] = use_first_row_as_header_prop_test
            request.session['test_type_prop_test'] = test_type_prop_test
            graph_prop_test = create_bilateral_graph_prop_test(variable_names, z_scores, z_criticals)
            context['graph_prop_test'] = graph_prop_test
            request.session['graph_prop_test'] = graph_prop_test
            explanation_prop_test = "Two-Tailed Test - Null Hypothesis (H₀): p = p₀ - Alternative Hypothesis (H₁): p ≠ p₀"
            context['explanation_prop_test'] = explanation_prop_test
            request.session['explanation_prop_test'] = explanation_prop_test

###########################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "prop_diff_test":
        data_prop_diff_test = request.POST.get('data_prop_diff_test')
        use_first_row_as_header_prop_diff_test = request.POST.get('use_first_row_as_header_prop_diff_test') == 'on'
        alpha_value_prop_diff_test = request.POST.get('alpha_value_prop_diff_test')
        test_type_prop_diff_test = request.POST.get('test_type_prop_diff_test')

        if alpha_value_prop_diff_test:
            try:
                alpha_value_prop_diff_test = float(alpha_value_prop_diff_test)
            except ValueError:
                alpha_value_prop_diff_test = None

        if not data_prop_diff_test.strip():
            context['error'] = "Please enter data before calculating."
            return render(request, "inference/inference.html", context)

        data_prop_diff_test = data_prop_diff_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_prop_diff_test.split('\n')]
        columns = []
        headers_prop_diff_test = []

        if use_first_row_as_header_prop_diff_test and rows:
            headers_prop_diff_test_3 = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            raise ValueError("Non-numeric value found. Ensure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            return render(request, "inference/inference.html", context)
        
        if not alpha_value_prop_diff_test:
            alpha_value_prop_diff_test = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."

        if test_type_prop_diff_test == "greater":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                prop1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                prop2 = float(data_columns[1][1])
                x1 = prop1 * n1
                x2 = prop2 * n2
                combined_prop = (x1 + x2) / (n1 + n2)
                std_error_diff = np.sqrt(combined_prop * (1 - combined_prop) * (1 / n1 + 1 / n2))
                z_stat = (prop1 - prop2) / std_error_diff
                alpha = (float(alpha_value_prop_diff_test) / 100)
                z_critical = norm.ppf(alpha)
                p_value = 1 - norm.cdf(z_stat)
                reject_null = 'Yes' if z_stat > z_critical else 'No'
                variable_names = [headers_prop_diff_test[0], headers_prop_diff_test[1]] if headers_prop_diff_test else ["var. 1", "var. 2"]

                results_prop_diff_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'proportion': f"{prop1:.4f}, {prop2:.4f}",
                    'alpha': alpha_value_prop_diff_test,
                    'z_stat': f"{z_stat:.4f}",
                    'z_critical': f"{z_critical:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_prop_diff_test'] = results_prop_diff_test
                context['data_prop_diff_test'] = data_prop_diff_test
                context['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                context['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                context['use_first_row_as_header_prop_diff_test'] = 'checked' if use_first_row_as_header_prop_diff_test else ''
                context['test_type_prop_diff_test'] = test_type_prop_diff_test
                request.session['results_prop_diff_test'] = results_prop_diff_test
                request.session['data_prop_diff_test'] = data_prop_diff_test
                request.session['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                request.session['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                request.session['use_first_row_as_header_prop_diff_test'] = use_first_row_as_header_prop_diff_test
                request.session['test_type_prop_diff_test'] = test_type_prop_diff_test
                graph_prop_diff_test = create_right_graph_prop_diff_test(variable_names, z_stat, z_critical)
                context['graph_prop_diff_test'] = f'data:image/png;base64,{graph_prop_diff_test}'
                request.session['graph_prop_diff_test'] = context['graph_prop_diff_test']
                explanation_prop_diff_test = "One-Tailed Right Test - Null Hypothesis (H₀): p₁ ≤ p₂ - Alternative Hypothesis (H₁): p₁ > p₂"
                context['explanation_prop_diff_test'] = explanation_prop_diff_test
                request.session['explanation_prop_diff_test'] = explanation_prop_diff_test

        elif test_type_prop_diff_test == "lesser":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                prop1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                prop2 = float(data_columns[1][1])
                x1 = prop1 * n1
                x2 = prop2 * n2
                combined_prop = (x1 + x2) / (n1 + n2)
                std_error_diff = np.sqrt(combined_prop * (1 - combined_prop) * (1 / n1 + 1 / n2))
                z_stat = (prop1 - prop2) / std_error_diff
                alpha = (float(alpha_value_prop_diff_test) / 100)
                z_critical = norm.ppf(1 - alpha)
                p_value = norm.cdf(z_stat)
                reject_null = 'Yes' if z_stat < z_critical else 'No'
                variable_names = [headers_prop_diff_test[0], headers_prop_diff_test[1]] if headers_prop_diff_test else ["var. 1", "var. 2"]

                results_prop_diff_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'proportion': f"{prop1:.4f}, {prop2:.4f}",
                    'alpha': alpha_value_prop_diff_test,
                    'z_stat': f"{z_stat:.4f}",
                    'z_critical': f"{z_critical:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_prop_diff_test'] = results_prop_diff_test
                context['data_prop_diff_test'] = data_prop_diff_test
                context['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                context['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                context['use_first_row_as_header_prop_diff_test'] = 'checked' if use_first_row_as_header_prop_diff_test else ''
                context['test_type_prop_diff_test'] = test_type_prop_diff_test
                request.session['results_prop_diff_test'] = results_prop_diff_test
                request.session['data_prop_diff_test'] = data_prop_diff_test
                request.session['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                request.session['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                request.session['use_first_row_as_header_prop_diff_test'] = use_first_row_as_header_prop_diff_test
                request.session['test_type_prop_diff_test'] = test_type_prop_diff_test
                graph_prop_diff_test = create_left_graph_prop_diff_test(variable_names, z_stat, z_critical)
                context['graph_prop_diff_test'] = f'data:image/png;base64,{graph_prop_diff_test}'
                request.session['graph_prop_diff_test'] = context['graph_prop_diff_test']
                explanation_prop_diff_test = "One-Tailed Left Test - Null Hypothesis (H₀): p₁ ≥ p₂ - Alternative Hypothesis (H₁): p₁ < p₂"
                context['explanation_prop_diff_test'] = explanation_prop_diff_test
                request.session['explanation_prop_diff_test'] = explanation_prop_diff_test

        elif test_type_prop_diff_test == "different":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                prop1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                prop2 = float(data_columns[1][1])
                x1 = prop1 * n1
                x2 = prop2 * n2
                combined_prop = (x1 + x2) / (n1 + n2)
                std_error_diff = np.sqrt(combined_prop * (1 - combined_prop) * (1 / n1 + 1 / n2))
                z_stat = (prop1 - prop2) / std_error_diff
                alpha = (float(alpha_value_prop_diff_test) / 100)
                z_critical = -1 * norm.ppf((1 - alpha) / 2)
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))
                reject_null = 'Yes' if abs(z_stat) > z_critical else 'No'
                variable_names = [headers_prop_diff_test[0], headers_prop_diff_test[1]] if headers_prop_diff_test else ["var. 1", "var. 2"]

                results_prop_diff_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'proportion': f"{prop1:.4f}, {prop2:.4f}",
                    'alpha': alpha_value_prop_diff_test,
                    'z_stat': f"{z_stat:.4f}",
                    'z_critical': f"{z_critical:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_prop_diff_test'] = results_prop_diff_test
                context['data_prop_diff_test'] = data_prop_diff_test
                context['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                context['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                context['use_first_row_as_header_prop_diff_test'] = 'checked' if use_first_row_as_header_prop_diff_test else ''
                context['test_type_prop_diff_test'] = test_type_prop_diff_test
                request.session['results_prop_diff_test'] = results_prop_diff_test
                request.session['data_prop_diff_test'] = data_prop_diff_test
                request.session['alpha_value_prop_diff_test'] = alpha_value_prop_diff_test
                request.session['headers_prop_diff_test'] = headers_prop_diff_test if use_first_row_as_header_prop_diff_test else None
                request.session['use_first_row_as_header_prop_diff_test'] = use_first_row_as_header_prop_diff_test
                request.session['test_type_prop_diff_test'] = test_type_prop_diff_test
                graph_prop_diff_test = create_bilateral_graph_prop_diff_test(variable_names, z_stat, z_critical)
                context['graph_prop_diff_test'] = f'data:image/png;base64,{graph_prop_diff_test}'
                request.session['graph_prop_diff_test'] = context['graph_prop_diff_test']
                explanation_prop_diff_test = "Two-Tailed Test - Null Hypothesis (H₀): p₁ = p₂ - Alternative Hypothesis (H₁): p₁ ≠ p₂"
                context['explanation_prop_diff_test'] = explanation_prop_diff_test
                request.session['explanation_prop_diff_test'] = explanation_prop_diff_test

########################################################################################################################################################
    if request.method == "POST" and tab == "hypothesis" and subtab == "var_quo_test":
        data_var_quo_test = request.POST.get('data_var_quo_test')
        use_first_row_as_header_var_quo_test = request.POST.get('use_first_row_as_header_var_quo_test') == 'on'
        test_type_var_quo_test = request.POST.get('test_type_var_quo_test')
        alpha_value_var_quo_test = request.POST.get('alpha_value_var_quo_test')

        if alpha_value_var_quo_test:
            try:
                alpha_value_var_quo_test = float(alpha_value_var_quo_test)
            except ValueError:
                alpha_value_var_quo_test = None

        if not data_var_quo_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_var_quo_test'] = None
            context['graph_var_quo_test'] = None
            return render(request, "inference/inference.html", context)

        data_var_quo_test = data_var_quo_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_var_quo_test.split('\n')]
        columns = []
        headers_var_quo_test = []

        if use_first_row_as_header_var_quo_test and rows:
            headers_var_quo_test = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_var_quo_test:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph_var_quo_test'] = None
            context['results_var_quo_test'] = None
            return render(request, "inference/inference.html", context)

        if not alpha_value_var_quo_test:
            alpha_value_var_quo_test = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."

        if test_type_var_quo_test == "greater":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                var1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                var2 = float(data_columns[1][1])
                alpha = (float(alpha_value_var_quo_test) / 100)
                f_stat = var1 / var2
                df1 = n1 - 1
                df2 = n2 - 1
                f_critical = f.ppf(alpha, df1, df2)
                p_value = f.cdf(f_stat, df1, df2)
                reject_null = 'Yes' if f_stat > f_critical else 'No'
                variable_names = [headers_var_quo_test[0], headers_var_quo_test[1]] if headers_var_quo_test else ["var. 1", "var. 2"]

                results_var_quo_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'variance': f"{var1:.4f}, {var2:.4f}",
                    'df': f"{df1}, {df2}",
                    'alpha': alpha_value_var_quo_test,
                    'f_stat': f"{f_stat:.4f}",
                    'f_critical': f"{f_critical:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_var_quo_test'] = results_var_quo_test
                context['data_var_quo_test'] = data_var_quo_test
                context['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                context['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                context['use_first_row_as_header_var_quo_test'] = 'checked' if use_first_row_as_header_var_quo_test else ''
                context['test_type_var_quo_test'] = test_type_var_quo_test
                request.session['results_var_quo_test'] = results_var_quo_test
                request.session['data_var_quo_test'] = data_var_quo_test
                request.session['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                request.session['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                request.session['use_first_row_as_header_var_quo_test'] = use_first_row_as_header_var_quo_test
                request.session['test_type_var_quo_test'] = test_type_var_quo_test
                graph_var_quo_test = create_right_graph_var_quo_test(variable_names, f_stat, f_critical, df1, df2)
                context['graph_var_quo_test'] = f'data:image/png;base64,{graph_var_quo_test}'
                request.session['graph_var_quo_test'] = context['graph_var_quo_test']
                explanation_var_quo_test = "One-Tailed Right Test - Null Hypothesis (H₀): σ₁² = σ₂² - Alternative Hypothesis (H₁): σ₁² > σ₂²"
                context['explanation_var_quo_test'] = explanation_var_quo_test
                request.session['explanation_var_quo_test'] = explanation_var_quo_test

        elif test_type_var_quo_test == "lesser":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                var1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                var2 = float(data_columns[1][1])
                alpha = (float(alpha_value_var_quo_test) / 100)
                f_stat = var1 / var2
                df1 = n1 - 1
                df2 = n2 - 1
                f_critical = f.ppf(1 - alpha, df1, df2)
                p_value = f.cdf(f_stat, df1, df2)
                reject_null = 'Yes' if f_stat < f_critical else 'No'
                variable_names = [headers_var_quo_test[0], headers_var_quo_test[1]] if headers_var_quo_test else ["var. 1", "var. 2"]

                results_var_quo_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'variance': f"{var1:.4f}, {var2:.4f}",
                    'df': f"{df1}, {df2}",
                    'alpha': alpha_value_var_quo_test,
                    'f_stat': f"{f_stat:.4f}",
                    'f_critical': f"{f_critical:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_var_quo_test'] = results_var_quo_test
                context['data_var_quo_test'] = data_var_quo_test
                context['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                context['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                context['use_first_row_as_header_var_quo_test'] = 'checked' if use_first_row_as_header_var_quo_test else ''
                context['test_type_var_quo_test'] = test_type_var_quo_test
                request.session['results_var_quo_test'] = results_var_quo_test
                request.session['data_var_quo_test'] = data_var_quo_test
                request.session['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                request.session['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                request.session['use_first_row_as_header_var_quo_test'] = use_first_row_as_header_var_quo_test
                request.session['test_type_var_quo_test'] = test_type_var_quo_test
                graph_var_quo_test = create_left_graph_var_quo_test(variable_names, f_stat, f_critical, df1, df2)
                context['graph_var_quo_test'] = f'data:image/png;base64,{graph_var_quo_test}'
                request.session['graph_var_quo_test'] = context['graph_var_quo_test']
                explanation_var_quo_test = "One-Tailed Left Test - Null Hypothesis (H₀): σ₁² = σ₂² - Alternative Hypothesis (H₁): σ₁² < σ₂²"
                context['explanation_var_quo_test'] = explanation_var_quo_test
                request.session['explanation_var_quo_test'] = explanation_var_quo_test

        elif test_type_var_quo_test == "different":
            if len(data_columns) == 2 and all(len(row) == 2 for row in data_columns):
                n1 = int(data_columns[0][0])
                var1 = float(data_columns[0][1])
                n2 = int(data_columns[1][0])
                var2 = float(data_columns[1][1])
                alpha = (float(alpha_value_var_quo_test))
                f_stat = var1 / var2
                df1 = n1 - 1
                df2 = n2 - 1
                f_critical_low = f.ppf((((100 - alpha)/100))/2, df1, df2)
                f_critical_high = f.ppf(1-((((100 - alpha)/100)))/2, df1, df2)
                if f_stat > 1:
                    p_value = 2 * (1 - f.cdf(f_stat, df1, df2))
                else:
                    p_value = 2 * f.cdf(f_stat, df1, df2)
                reject_null = 'Yes' if f_stat < f_critical_low or f_stat > f_critical_high else 'No'
                variable_names = [headers_var_quo_test[0], headers_var_quo_test[1]] if headers_var_quo_test else ["var. 1", "var. 2"]

                results_var_quo_test = [{
                    'variable': f"{variable_names[0]}, {variable_names[1]}",
                    'n_elements': f"{n1}, {n2}",
                    'variance': f"{var1:.4f}, {var2:.4f}",
                    'df': f"{df1}, {df2}",
                    'alpha': alpha_value_var_quo_test,
                    'f_stat': f"{f_stat:.4f}",
                    'f_critical': f"{f_critical_low:.4f}, {f_critical_high:.4f}",
                    'p_value': f"{p_value:.4f}",
                    'reject_null': reject_null,
                }]

                context['results_var_quo_test'] = results_var_quo_test
                context['data_var_quo_test'] = data_var_quo_test
                context['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                context['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                context['use_first_row_as_header_var_quo_test'] = 'checked' if use_first_row_as_header_var_quo_test else ''
                context['test_type_var_quo_test'] = test_type_var_quo_test
                request.session['results_var_quo_test'] = results_var_quo_test
                request.session['data_var_quo_test'] = data_var_quo_test
                request.session['alpha_value_var_quo_test'] = alpha_value_var_quo_test
                request.session['headers_var_quo_test'] = headers_var_quo_test if use_first_row_as_header_var_quo_test else None
                request.session['use_first_row_as_header_var_quo_test'] = use_first_row_as_header_var_quo_test
                request.session['test_type_var_quo_test'] = test_type_var_quo_test
                graph_var_quo_test = create_two_tail_graph_var_quo_test(variable_names, f_stat, f_critical_low, f_critical_high, df1, df2)
                context['graph_var_quo_test'] = f'data:image/png;base64,{graph_var_quo_test}'
                request.session['graph_var_quo_test'] = context['graph_var_quo_test']
                explanation_var_quo_test = "Two-Tailed Test - Null Hypothesis (H₀): σ₁² = σ₂² - Alternative Hypothesis (H₁): σ₁² ≠ σ₂²"
                context['explanation_var_quo_test'] = explanation_var_quo_test
                request.session['explanation_var_quo_test'] = explanation_var_quo_test

###########################################################################################################
    if request.method == "POST" and tab == "non_param" and subtab == "U_test":
        data_U_test = request.POST.get('data_U_test')
        use_first_row_as_header_U_test = request.POST.get('use_first_row_as_header_U_test') == 'on'
        test_type_U_test = request.POST.get('test_type_U_test')
        alpha_value_U_test = request.POST.get('alpha_value_U_test')

        if alpha_value_U_test:
            try:
                alpha_value_U_test = float(alpha_value_U_test)
            except ValueError:
                alpha_value_U_test = None

        if not data_U_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_U_test'] = None
            return render(request, "inference/inference.html", context)

        data_U_test = data_U_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_U_test.split('\n')]
        headers_U_test = []

        if use_first_row_as_header_U_test and rows:
            headers_U_test = rows[0]
            rows = rows[1:]

        try:
            sample_1 = []
            sample_2 = []
            for row_idx, row in enumerate(rows):
                if len(row) < 1:
                    continue
                try:
                    if len(row) >= 1 and row[0].strip():  # Si hay un valor en la primera columna
                        sample_1.append(float(row[0].strip()))
                    if len(row) >= 2 and row[1].strip():  # Si hay un valor en la segunda columna
                        sample_2.append(float(row[1].strip()))
                except ValueError:
                    raise ValueError(f"Invalid numeric value found in row {row_idx + 1}.")
            
            if not sample_1 or not sample_2:
                raise ValueError("Both samples must contain data.")

        except ValueError as e:
            context['error'] = str(e)
            context['results_U_test'] = None
            return render(request, "inference/inference.html", context)

        if not alpha_value_U_test:
            alpha_value_U_test = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."

        alpha = (float(alpha_value_U_test) / 100)

        alternative = {'greater': 'greater', 'lesser': 'less', 'different': 'two-sided'}[test_type_U_test]
        u_stat, p_value = mannwhitneyu(sample_1, sample_2, alternative=alternative)
        mean_sample1 = np.mean(sample_1)
        mean_sample2 = np.mean(sample_2)
        reject_null = 'Yes' if p_value < 1 - alpha else 'No'

        variable_names = headers_U_test if headers_U_test else ["var. 1", "var. 2"]
        results_U_test = [{
            'variable': f"{variable_names[0]}, {variable_names[1]}",
            'n_elements': f"{len(sample_1)}, {len(sample_2)}",
            'means': f"{mean_sample1:.4f}, {mean_sample2:.4f}",
            'alpha': alpha_value_U_test,
            'U_stat': f"{u_stat:.4f}",
            'p_value': f"{p_value:.4f}",
            'reject_null': reject_null,
        }]

        context['results_U_test'] = results_U_test
        context['data_U_test'] = data_U_test
        context['alpha_value_U_test'] = alpha_value_U_test
        context['headers_U_test'] = headers_U_test if use_first_row_as_header_U_test else None
        context['use_first_row_as_header_U_test'] = 'checked' if use_first_row_as_header_U_test else ''
        context['test_type_U_test'] = test_type_U_test
        request.session['results_U_test'] = results_U_test
        request.session['data_U_test'] = data_U_test
        request.session['alpha_value_U_test'] = alpha_value_U_test
        request.session['headers_U_test'] = headers_U_test if use_first_row_as_header_U_test else None
        request.session['use_first_row_as_header_U_test'] = use_first_row_as_header_U_test
        request.session['test_type_U_test'] = test_type_U_test
        explanation_U_test = {
            'greater': "One-Tailed Right Test - Null Hypothesis (H₀): Sample 1 ≤ Sample 2 - Alternative Hypothesis (H₁): Sample 1 > Sample 2",
            'lesser': "One-Tailed Left Test - Null Hypothesis (H₀): Sample 1 ≥ Sample 2 - Alternative Hypothesis (H₁): Sample 1 < Sample 2",
            'different': "Two-Tailed Test - Null Hypothesis (H₀): Sample 1 = Sample 2 - Alternative Hypothesis (H₁): Sample 1 ≠ Sample 2",
        }
        context['explanation_U_test'] = explanation_U_test[test_type_U_test]
        request.session['explanation_U_test'] = context['explanation_U_test']

#############################################################################################################
    if request.method == "POST" and tab == "non_param" and subtab == "U_signed_test":
        data_U_signed_test = request.POST.get('data_U_signed_test')
        use_first_row_as_header_U_signed_test = request.POST.get('use_first_row_as_header_U_signed_test') == 'on'
        test_type_U_signed_test = request.POST.get('test_type_U_signed_test')
        alpha_value_U_signed_test = request.POST.get('alpha_value_U_signed_test')

        if alpha_value_U_signed_test:
            try:
                alpha_value_U_signed_test = float(alpha_value_U_signed_test)
            except ValueError:
                alpha_value_U_signed_test = None

        if not data_U_signed_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_U_signed_test'] = None
            return render(request, "inference/inference.html", context)

        data_U_signed_test = data_U_signed_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_U_signed_test.split('\n')]
        headers_U_signed_test = []

        if use_first_row_as_header_U_signed_test and rows:
            headers_U_signed_test = rows[0]
            rows = rows[1:]

        try:
            sample_1 = []
            sample_2 = []
            for row_idx, row in enumerate(rows):
                if len(row) < 2:
                    raise ValueError("Each row must have at least two columns for the paired samples.")
                try:
                    sample_1.append(float(row[0].strip()))
                    sample_2.append(float(row[1].strip()))
                except ValueError:
                    raise ValueError(f"Invalid numeric value found in row {row_idx + 1}.")
            
            if not sample_1 or not sample_2:
                raise ValueError("Both samples must contain data.")

            if len(sample_1) != len(sample_2):
                raise ValueError("Paired samples must have the same number of elements.")

        except ValueError as e:
            context['error'] = str(e)
            context['results_U_signed_test'] = None
            return render(request, "inference/inference.html", context)

        if not alpha_value_U_signed_test:
            alpha_value_U_signed_test = 95
            context['warning'] = "Confidence Level not defined, defaulting to 95%."

        alpha = (float(alpha_value_U_signed_test) / 100)

        alternative = {'greater': 'greater', 'lesser': 'less', 'different': 'two-sided'}[test_type_U_signed_test]
        
        try:
            w_stat, p_value = wilcoxon(sample_1, sample_2, alternative=alternative)
            mean_diff = np.mean(np.array(sample_1) - np.array(sample_2))
            reject_null = 'Yes' if p_value < 1 - alpha else 'No'
        except ValueError as e:
            context['error'] = f"Test failed: {str(e)}"
            context['results_U_signed_test'] = None
            return render(request, "inference/inference.html", context)

        variable_names = headers_U_signed_test if headers_U_signed_test else ["var. 1", "var. 2"]
        results_U_signed_test = [{
            'variable': f"{variable_names[0]}, {variable_names[1]}",
            'n_elements': f"{len(sample_1)}, {len(sample_2)}",
            'mean_difference': f"{mean_diff:.4f}",
            'alpha': alpha_value_U_signed_test,
            'W_stat': f"{w_stat:.4f}",
            'p_value': f"{p_value:.4f}",
            'reject_null': reject_null,
        }]

        context['results_U_signed_test'] = results_U_signed_test
        context['data_U_signed_test'] = data_U_signed_test
        context['alpha_value_U_signed_test'] = alpha_value_U_signed_test
        context['headers_U_signed_test'] = headers_U_signed_test if use_first_row_as_header_U_signed_test else None
        context['use_first_row_as_header_U_signed_test'] = 'checked' if use_first_row_as_header_U_signed_test else ''
        context['test_type_U_signed_test'] = test_type_U_signed_test
        request.session['results_U_signed_test'] = results_U_signed_test
        request.session['data_U_signed_test'] = data_U_signed_test
        request.session['alpha_value_U_signed_test'] = alpha_value_U_signed_test
        request.session['headers_U_signed_test'] = headers_U_signed_test if use_first_row_as_header_U_signed_test else None
        request.session['use_first_row_as_header_U_signed_test'] = use_first_row_as_header_U_signed_test
        request.session['test_type_U_signed_test'] = test_type_U_signed_test
        explanation_U_signed_test = {
            'greater': "One-Tailed Right Test - Null Hypothesis (H₀): Median difference ≤ 0 - Alternative Hypothesis (H₁): Median difference > 0",
            'lesser': "One-Tailed Left Test - Null Hypothesis (H₀): Median difference ≥ 0 - Alternative Hypothesis (H₁): Median difference < 0",
            'different': "Two-Tailed Test - Null Hypothesis (H₀): Median difference = 0 - Alternative Hypothesis (H₁): Median difference ≠ 0",
        }
        context['explanation_U_signed_test'] = explanation_U_signed_test[test_type_U_signed_test]
        request.session['explanation_U_signed_test'] = context['explanation_U_signed_test']

############################################################################################################################
    if request.method == "POST" and tab == "dist_test" and subtab == "shap_test":
        data_shap_test = request.POST.get('data_shap_test')
        use_first_row_as_header_shap_test = request.POST.get('use_first_row_as_header_shap_test') == 'on'
        alpha_value_shap_test = request.POST.get('alpha_value_shap_test')

        if alpha_value_shap_test:
            try:
                alpha_value_shap_test = float(alpha_value_shap_test)
            except ValueError:
                alpha_shap_signed_test = None

        if not data_shap_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_shap_test'] = None
            return render(request, "inference/inference.html", context)

        data_shap_test = data_shap_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_shap_test.split('\n')]
        columns = []
        headers_shap_test = []

        if use_first_row_as_header_shap_test and rows:
            headers_shap_test = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_shap_test:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['results_shap_test'] = None
            return render(request, "inference/inference.html", context)

        results_shap_test = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                n_elements = len(valid_col)
                shapiro_w, shapiro_p = shapiro(valid_col)
                variable_name = headers_shap_test[i] if use_first_row_as_header_shap_test and i < len(headers_shap_test) else f"var. {i + 1}"
                
                if not alpha_value_shap_test:
                    alpha_value_shap_test = 95
                    context['warning'] = "Confidence Level not defined, defaulting to 95%."

                alpha = 1 - (alpha_value_shap_test / 100)
                reject_null = 'Yes' if shapiro_p < alpha else 'No'

                results_shap_test.append({
                    'variable': variable_name,
                    'n_elements': n_elements,
                    'alpha': alpha_value_shap_test,
                    'shapiro_w': "{:.4g}".format(shapiro_w),
                    'shapiro_p': "{:.4g}".format(shapiro_p),
                    'reject_null': reject_null,
                })

        context['results_shap_test'] = results_shap_test
        context['data_shap_test'] = data_shap_test
        context['headers_shap_test'] = headers_shap_test if use_first_row_as_header_shap_test else None
        context['use_first_row_as_header_shap_test'] = 'checked' if use_first_row_as_header_shap_test else ''
        context['alpha_value_shap_test'] = alpha_value_shap_test
        request.session['results_shap_test'] = results_shap_test
        request.session['data_shap_test'] = data_shap_test
        request.session['headers_shap_test'] = headers_shap_test if use_first_row_as_header_shap_test else None
        request.session['use_first_row_as_header_shap_test'] = use_first_row_as_header_shap_test
        request.session['alpha_value_shap_test'] = alpha_value_shap_test
        explanation_shap_test = "Shapiro-Wilk Test - Null Hypothesis (H₀): The data follows a normal distribution - Alternative Hypothesis (H₁): The data does not follow a normal distribution."
        context['explanation_shap_test'] = explanation_shap_test
        request.session['explanation_shap_test'] = explanation_shap_test

############################################################################################################################
    if request.method == "POST" and tab == "dist_test" and subtab == "kol_test":
        data_kol_test = request.POST.get('data_kol_test')
        use_first_row_as_header_kol_test = request.POST.get('use_first_row_as_header_kol_test') == 'on'
        alpha_value_kol_test = request.POST.get('alpha_value_kol_test')
        distribution_function = request.POST.get('distribution_function')

        if alpha_value_kol_test:
            try:
                alpha_value_kol_test = float(alpha_value_kol_test)
            except ValueError:
                alpha_kol_signed_test = None

        if not data_kol_test.strip():
            context['error'] = "Please enter data before calculating."
            context['results_kol_test'] = None
            return render(request, "inference/inference.html", context)

        data_kol_test = data_kol_test.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_kol_test.split('\n')]
        columns = []
        headers_kol_test = []

        if use_first_row_as_header_kol_test and rows:
            headers_kol_test = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_kol_test:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['results_kol_test'] = None
            return render(request, "inference/inference.html", context)

        results_kol_test = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                n_elements = len(valid_col)

                if not distribution_function:
                    context['error'] = "Please select a distribution function for the Kolmogorov-Smirnov Test."
                    context['results_kol_test'] = None
                    return render(request, "inference/inference.html", context)

                if distribution_function == "normal":
                    ks_stat, ks_p = kstest(valid_col, 'norm', args=(np.mean(valid_col), np.std(valid_col)))
                elif distribution_function == "uniform":
                    ks_stat, ks_p = kstest(valid_col, 'uniform', args=(np.min(valid_col), np.max(valid_col) - np.min(valid_col)))

                variable_name = headers_kol_test[i] if use_first_row_as_header_kol_test and i < len(headers_kol_test) else f"var. {i + 1}"

                if not alpha_value_kol_test:
                    alpha_value_kol_test = 95
                    context['warning'] = "Confidence Level not defined, defaulting to 95%."

                alpha = 1 - (alpha_value_kol_test / 100)
                reject_null = 'Yes' if ks_p < alpha else 'No'

                results_kol_test.append({
                    'variable': variable_name,
                    'n_elements': n_elements,
                    'alpha': alpha_value_kol_test,
                    'ks_stat': "{:.4g}".format(ks_stat),
                    'ks_p': "{:.4g}".format(ks_p),
                    'reject_null': reject_null,
                })

        context['results_kol_test'] = results_kol_test
        context['data_kol_test'] = data_kol_test
        context['headers_kol_test'] = headers_kol_test if use_first_row_as_header_kol_test else None
        context['use_first_row_as_header_kol_test'] = 'checked' if use_first_row_as_header_kol_test else ''
        context['alpha_value_kol_test'] = alpha_value_kol_test
        request.session['results_kol_test'] = results_kol_test
        request.session['data_kol_test'] = data_kol_test
        request.session['headers_kol_test'] = headers_kol_test if use_first_row_as_header_kol_test else None
        request.session['use_first_row_as_header_kol_test'] = use_first_row_as_header_kol_test
        request.session['alpha_value_kol_test'] = alpha_value_kol_test
        explanation_kol_test = "Kolmogorov-Smirnov Test - Null Hypothesis (H₀): The data follows the specified distribution - Alternative Hypothesis (H₁): The data does not follow the specified distribution."
        context['explanation_kol_test'] = explanation_kol_test
        request.session['explanation_kol_test'] = explanation_kol_test

####################################################################################################################################

    return render(request, "inference/inference.html", context)


def count_significant_figures(num_str):
    if isinstance(num_str, (float, np.float64)):
        num_str = str(num_str)
    else:
        num_str = num_str
    
    num_str = num_str.strip()
    num_str = re.sub(r'[eE][+-]?\d+', '', num_str)
    num_str = num_str.replace('.', '')
    return len(num_str.lstrip('0'))


def create_unilateral_rigth_graph_mean_test_1(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs

def create_unilateral_left_graph_mean_test_1(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_bilateral_graph_mean_test_1(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < -z_critical), color='red', alpha=0.3)
        ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_unilateral_rigth_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector):
    graphs = []
    for i in range(len(t_scores)):
        variable_name = variable_names[i]
        t_score = t_scores[i]
        t_critical = t_criticals[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x = np.linspace(-6, 6, 100)
        y = t.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
        ax.axvline(t_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('t')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_unilateral_left_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector):
    graphs = []
    for i in range(len(t_scores)):
        variable_name = variable_names[i]
        t_score = t_scores[i]
        t_critical = t_criticals[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x = np.linspace(-6, 6, 100)
        y = t.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < t_critical), color='red', alpha=0.3)
        ax.axvline(t_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('t')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_bilateral_graph_mean_test_2(variable_names, t_scores, t_criticals, n_vector):
    graphs = []
    for i in range(len(t_scores)):
        variable_name = variable_names[i]
        t_score = t_scores[i]
        t_critical = t_criticals[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x = np.linspace(-6, 6, 100)
        y = t.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < -t_critical), color='red', alpha=0.3)
        ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
        ax.axvline(t_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('t')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_unilateral_rigth_graph_var_test(variable_names, chi_scores, chi_criticals, n_vector):
    graphs = []
    for i in range(len(chi_scores)):
        variable_name = variable_names[i]
        chi_score = chi_scores[i]
        chi_critical = chi_criticals[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x_min = 0
        x_max = max(chi_critical + 3 * np.sqrt(2 * df), 2 * df)
        x = np.linspace(x_min, x_max, 100)
        y = chi2.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x > chi_critical), color='red', alpha=0.3)
        ax.axvline(chi_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('χ²')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_unilateral_left_graph_var_test(variable_names, chi_scores, chi_criticals, n_vector):
    graphs = []
    for i in range(len(chi_scores)):
        variable_name = variable_names[i]
        chi_score = chi_scores[i]
        chi_critical = chi_criticals[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x_min = 0
        x_max = max(chi_critical + 3 * np.sqrt(2 * df), 2 * df)
        x = np.linspace(x_min, x_max, 100)
        y = chi2.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < chi_critical), color='red', alpha=0.3)
        ax.axvline(chi_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('χ²')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_bilateral_graph_var_test(variable_names, chi_scores, chi_criticals_lower, chi_criticals_upper, n_vector):
    graphs = []
    for i in range(len(chi_scores)):
        variable_name = variable_names[i]
        chi_score = chi_scores[i]
        chi_critical_lower = chi_criticals_lower[i]
        chi_critical_upper = chi_criticals_upper[i]
        df = n_vector[i]
        fig, ax = plt.subplots()
        x_min = 0
        x_max = max(chi_critical_upper + 3 * np.sqrt(2 * df), 2 * df)
        x = np.linspace(x_min, x_max, 100)
        y = chi2.pdf(x, df)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < chi_critical_lower), color='red', alpha=0.3)
        ax.fill_between(x, y, where=(x > chi_critical_upper), color='red', alpha=0.3)
        ax.axvline(chi_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('χ²')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_right_graph_mean_diff_test_1(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3, label="Rejection Region")
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_left_graph_mean_diff_test_1(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < z_critical), color='red', alpha=0.3, label="Rejection Region")
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_bilateral_graph_mean_diff_test_1(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < -z_critical), color='red', alpha=0.3)
    ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_right_graph_mean_diff_test_2(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_left_graph_mean_diff_test_2(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_bilateral_graph_mean_diff_test_2(variable_names, t_stat, t_critical, t_critical_negative, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical_negative), color='red', alpha=0.3)
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Two-Tailed Test Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_right_graph_mean_diff_test_3(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_left_graph_mean_diff_test_3(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_bilateral_graph_mean_diff_test_3(variable_names, t_stat, t_critical, t_critical_negative, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical_negative), color='red', alpha=0.3)
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Two-Tailed Test Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_right_graph_mean_diff_test_4(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_left_graph_mean_diff_test_4(variable_names, t_stat, t_critical, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_bilateral_graph_mean_diff_test_4(variable_names, t_stat, t_critical, t_critical_negative, df):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)

    fig, ax = plt.subplots()
    x = np.linspace(-6, 6, 100)
    y = t.pdf(x, df)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < t_critical_negative), color='red', alpha=0.3)
    ax.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3)
    ax.axvline(t_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('t')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Two-Tailed Test Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_unilateral_rigth_graph_prop_test(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_unilateral_left_graph_prop_test(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_bilateral_graph_prop_test(variable_names, z_scores, z_criticals):
    graphs = []
    for i in range(len(z_scores)):
        variable_name = variable_names[i]
        z_score = z_scores[i]
        z_critical = z_criticals[i]
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x < -z_critical), color='red', alpha=0.3)
        ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
        ax.axvline(z_score, color='black', linestyle='dashed')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        ax.set_title(f"Rejection Region Graph for {variable_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph_base64)
        plt.close(fig)

    return graphs


def create_right_graph_prop_diff_test(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3, label="Rejection Region")
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_left_graph_prop_diff_test(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < z_critical), color='red', alpha=0.3, label="Rejection Region")
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_bilateral_graph_prop_diff_test(variable_names, z_stat, z_critical):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < -z_critical), color='red', alpha=0.3)
    ax.fill_between(x, y, where=(x > z_critical), color='red', alpha=0.3)
    ax.axvline(z_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlabel('z')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return graph


def create_right_graph_var_quo_test(variable_names, f_stat, f_critical, df1, df2):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    x = np.linspace(0, f.ppf(0.99, df1, df2), 100)
    y = f.pdf(x, df1, df2)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x > f_critical), color='red', alpha=0.3)
    ax.axvline(f_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlim(0, x.max())
    ax.set_xlabel('F')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph (Left-tailed) for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_left_graph_var_quo_test(variable_names, f_stat, f_critical, df1, df2):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    x = np.linspace(0, f.ppf(0.99, df1, df2), 100)
    y = f.pdf(x, df1, df2)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < f_critical), color='red', alpha=0.3)
    ax.axvline(f_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlim(0, x.max())
    ax.set_xlabel('F')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph (Left-tailed) for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph


def create_two_tail_graph_var_quo_test(variable_names, f_stat, f_critical_low, f_critical_high, df1, df2):
    if isinstance(variable_names, list):
        variable_names = " vs ".join(variable_names)
    x = np.linspace(0, f.ppf(0.99, df1, df2), 100)
    y = f.pdf(x, df1, df2)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, where=(x < f_critical_low) | (x > f_critical_high), color='red', alpha=0.3)
    ax.axvline(f_stat, color='black', linestyle='dashed')
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlim(0, x.max())
    ax.set_xlabel('F')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"Rejection Region Graph (Two-tailed) for {variable_names}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph