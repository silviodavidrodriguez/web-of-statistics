from django.shortcuts import render
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from scipy.stats import norm, t, chi2, f

def probability(request):
    tab = request.GET.get('tab', 'density')
    
    context = {
        'segment': 'probability',
        'active_tab': tab,
    }

    selected_distribution = []
    x_or_p = False
    tail = False
    show_second_form = False
    second_form_type = None
    result = None
    graph_image = None
    
    if request.method == "POST" and tab == "density":
        selected_distribution = request.POST.get('distribution')
        x_or_p = request.POST.get('x_or_p') == 'on'
        tail = request.POST.get('tail') == 'on'
########################################################################################################################################################
        if selected_distribution == "normal":
            show_second_form = "normal"
            if not x_or_p and not tail:
                second_form_type = "single_x"
            elif not x_or_p and tail:
                second_form_type = "double_x"
            elif x_or_p and not tail:
                second_form_type = "single_p"
            elif x_or_p and tail:
                second_form_type = "double_p"
            
            context['second_form_type'] = second_form_type
           
        elif 'x_value' in request.POST or 'p_value' in request.POST or 'x_value_1' in request.POST or 'p_value_1' in request.POST:
            second_form_type = request.POST.get('second_form_type')

            if second_form_type == "single_x":
                x_value = request.POST.get('x_value')
                x_value = float(x_value)
                prob_left = norm.cdf(x_value)
                prob_right = 1 - prob_left
                result = f"For a z value of: {x_value}\nProbability to the left (blue): {prob_left:.5f}\nProbability to the right (red): {prob_right:.5f}"
                result = result.replace("\n", "<br>")
                graph_image = plot_single_x(x_value)
                
            elif second_form_type == "double_x":
                x_value_1 = request.POST.get('x_value_1')
                x_value_2 = request.POST.get('x_value_2')
                x_value_1 = float(x_value_1)
                x_value_2 = float(x_value_2)
                x_value_1, x_value_2 = min(x_value_1, x_value_2), max(x_value_1, x_value_2)
                prob_left = norm.cdf(x_value_1)
                prob_right = 1 - norm.cdf(x_value_2)
                prob_between = norm.cdf(x_value_2) - norm.cdf(x_value_1)
                result = (
                    f"For the values of z: {x_value_1} and {x_value_2}\n"
                    f"Probability to the left of {x_value_1} (red): {prob_left:.5f}\n"
                    f"Probability between {x_value_1} and {x_value_2} (blue): {prob_between:.5f}\n"
                    f"Probability to the right of {x_value_2} (red): {prob_right:.5f}"
                )
                result = result.replace("\n", "<br>")
                graph_image = plot_double_x(x_value_1, x_value_2)
                
            elif second_form_type == "single_p":
                p_value = request.POST.get('p_value')
                p_value = float(p_value)
                if 0 <= p_value <= 1:
                    x_value = norm.ppf(p_value)
                    result = (
                            f"For a probability value of: {p_value} (blue)\n"
                            f"Value of z: {x_value:.5f}\n"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_single_x(x_value)
                else:
                    result =  "Error: The entered probability must be between 0 and 1."

            elif second_form_type == "double_p":
                p_value_1 = request.POST.get('p_value_1')
                p_value_2 = request.POST.get('p_value_2')
                p_value_1 = float(p_value_1)
                p_value_2 = float(p_value_2)
                if 0 <= p_value_1 <= 1 and 0 <= p_value_2 <= 1:
                    p_value_1, p_value_2 = min(p_value_1, p_value_2), max(p_value_1, p_value_2)
                    p_between = p_value_2 - p_value_1
                    x_value_1 = norm.ppf(p_value_1)
                    x_value_2 = norm.ppf(p_value_2)
                    result = (
                        f"For probabilities: {p_value_1} and {p_value_2}\n"
                        f"Value of z at left probability: {x_value_1:.5f}\n"
                        f"Value of z at right probability: {x_value_2:.5f}\n"
                        f"Probability between (blue): {p_between:.5f}"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_double_x(x_value_1, x_value_2)
                else:
                    result =  "Error: The entered probabilities must be between 0 and 1."


#################################################################################################################################################################
        if selected_distribution == "t_student":
            show_second_form = "t_student"
            if not x_or_p and not tail:
                second_form_type = "single_xt"
            elif not x_or_p and tail:
                second_form_type = "double_xt"
            elif x_or_p and not tail:
                second_form_type = "single_pt"
            elif x_or_p and tail:
                second_form_type = "double_pt"
            
            context['second_form_type'] = second_form_type

        elif 'xt_value' in request.POST or 'pt_value' in request.POST or 'xt_value_1' in request.POST or 'pt_value_1' in request.POST:
            second_form_type = request.POST.get('second_form_type')

            if second_form_type == "single_xt":
                x_value = request.POST.get('xt_value')
                df_value = request.POST.get('df_value')
                x_value = float(x_value)
                df = int(df_value)
                prob_left = t.cdf(x_value, df)
                prob_right = 1 - prob_left
                result = f"For a t value of: {x_value} with {df} degrees of freedom\nProbability to the left (blue): {prob_left:.4f}\nProbability to the right (red): {prob_right:.4f}"
                result = result.replace("\n", "<br>")
                graph_image = plot_single_xt(x_value, df)
                
            elif second_form_type == "double_xt":
                x_value_1 = request.POST.get('xt_value_1')
                x_value_2 = request.POST.get('xt_value_2')
                df_value = request.POST.get('df_value')
                x_value_1 = float(x_value_1)
                x_value_2 = float(x_value_2)
                x_value_1, x_value_2 = min(x_value_1, x_value_2), max(x_value_1, x_value_2)
                df = int(df_value)
                prob_left = t.cdf(x_value_1, df)
                prob_right = 1 - t.cdf(x_value_2, df)
                prob_between = t.cdf(x_value_2, df) - t.cdf(x_value_1, df)
                result = (
                    f"For the values of t: {x_value_1} and {x_value_2} with {df} degrees fo freedom\n"
                    f"Probability to the left of {x_value_1} (red): {prob_left:.4f}\n"
                    f"Probability between {x_value_1} and {x_value_2} (blue): {prob_between:.4f}\n"
                    f"Probability to the right of {x_value_2} (red): {prob_right:.4f}"
                )
                result = result.replace("\n", "<br>")
                graph_image = plot_double_xt(x_value_1, x_value_2, df)
                
            elif second_form_type == "single_pt":
                p_value = request.POST.get('pt_value')
                df_value = request.POST.get('df_value')
                p_value = float(p_value)
                df = int(df_value)
                if 0 <= p_value <= 1:
                    x_value = t.ppf(p_value, df)
                    result = (
                            f"For a probability value of: {p_value} (blue) with {df} degrees of freedom\n"
                            f"Value of t: {x_value:.4f}\n"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_single_xt(x_value, df)
                else:
                    result =  "Error: The entered probability must be between 0 and 1."

            elif second_form_type == "double_pt":
                p_value_1 = request.POST.get('pt_value_1')
                p_value_2 = request.POST.get('pt_value_2')
                df_value = request.POST.get('df_value')
                p_value_1 = float(p_value_1)
                p_value_2 = float(p_value_2)
                df = int(df_value)
                if 0 <= p_value_1 <= 1 and 0 <= p_value_2 <= 1:
                    p_value_1, p_value_2 = min(p_value_1, p_value_2), max(p_value_1, p_value_2)
                    p_between = p_value_2 - p_value_1
                    x_value_1 = t.ppf(p_value_1, df)
                    x_value_2 = t.ppf(p_value_2, df)
                    result = (
                        f"For probabilities: {p_value_1} and {p_value_2} with {df} degrees of freedom\n"
                        f"Value of t at left probability: {x_value_1:.4f}\n"
                        f"Value of t at right probability: {x_value_2:.4f}\n"
                        f"Probability between (blue): {p_between:.4f}"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_double_xt(x_value_1, x_value_2, df)
                else:
                    result =  "Error: The entered probabilities must be between 0 and 1."

##################################################################################################################################################
        if selected_distribution == "chi_squared":
            show_second_form = "chi_squared"
            if not x_or_p and not tail:
                second_form_type = "single_xc"
            elif not x_or_p and tail:
                second_form_type = "double_xc"
            elif x_or_p and not tail:
                second_form_type = "single_pc"
            elif x_or_p and tail:
                second_form_type = "double_pc"
            
            context['second_form_type'] = second_form_type

        elif 'xc_value' in request.POST or 'pc_value' in request.POST or 'xc_value_1' in request.POST or 'pc_value_1' in request.POST:
            second_form_type = request.POST.get('second_form_type')

            if second_form_type == "single_xc":
                x_value = request.POST.get('xc_value')
                dfc_value = request.POST.get('dfc_value')
                x_value = float(x_value)
                dfc = int(dfc_value)
                prob_left = chi2.cdf(x_value, dfc)
                prob_right = 1 - prob_left
                result = f"For a χ² value of: {x_value} with {dfc} degrees of freedom\nProbability to the left (blue): {prob_left:.4f}\nProbability to the right (red): {prob_right:.4f}"
                result = result.replace("\n", "<br>")
                graph_image = plot_single_xc(x_value, dfc)
                
            elif second_form_type == "double_xc":
                x_value_1 = request.POST.get('xc_value_1')
                x_value_2 = request.POST.get('xc_value_2')
                dfc_value = request.POST.get('dfc_value')
                x_value_1 = float(x_value_1)
                x_value_2 = float(x_value_2)
                x_value_1, x_value_2 = min(x_value_1, x_value_2), max(x_value_1, x_value_2)
                dfc = int(dfc_value)
                prob_left = chi2.cdf(x_value_1, dfc)
                prob_right = 1 - chi2.cdf(x_value_2, dfc)
                prob_between = chi2.cdf(x_value_2, dfc) - chi2.cdf(x_value_1, dfc)
                result = (
                    f"For the values of χ²: {x_value_1} and {x_value_2} with {dfc} degrees fo freedom\n"
                    f"Probability to the left of {x_value_1} (red): {prob_left:.4f}\n"
                    f"Probability between {x_value_1} and {x_value_2} (blue): {prob_between:.4f}\n"
                    f"Probability to the right of {x_value_2} (red): {prob_right:.4f}"
                )
                result = result.replace("\n", "<br>")
                graph_image = plot_double_xc(x_value_1, x_value_2, dfc)
                
            elif second_form_type == "single_pc":
                p_value = request.POST.get('pc_value')
                dfc_value = request.POST.get('dfc_value')
                p_value = float(p_value)
                dfc = int(dfc_value)
                if 0 <= p_value <= 1:
                    x_value = chi2.ppf(p_value, dfc)
                    result = (
                            f"For a probability value of: {p_value} (blue) with {dfc} degrees of freedom\n"
                            f"Value of χ²: {x_value:.4f}\n"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_single_xc(x_value, dfc)
                else:
                    result =  "Error: The entered probability must be between 0 and 1."

            elif second_form_type == "double_pc":
                p_value_1 = request.POST.get('pc_value_1')
                p_value_2 = request.POST.get('pc_value_2')
                dfc_value = request.POST.get('dfc_value')
                p_value_1 = float(p_value_1)
                p_value_2 = float(p_value_2)
                dfc = int(dfc_value)
                if 0 <= p_value_1 <= 1 and 0 <= p_value_2 <= 1:
                    p_value_1, p_value_2 = min(p_value_1, p_value_2), max(p_value_1, p_value_2)
                    p_between = p_value_2 - p_value_1
                    x_value_1 = chi2.ppf(p_value_1, dfc)
                    x_value_2 = chi2.ppf(p_value_2, dfc)
                    result = (
                        f"For probabilities: {p_value_1} and {p_value_2} with {dfc} degrees of freedom\n"
                        f"Value of χ² at left probability: {x_value_1:.4f}\n"
                        f"Value of χ² at right probability: {x_value_2:.4f}\n"
                        f"Probability between (blue): {p_between:.4f}"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_double_xc(x_value_1, x_value_2, dfc)
                else:
                    result =  "Error: The entered probabilities must be between 0 and 1."
#################################################################################################################################################################
        if selected_distribution == "fisher":
            show_second_form = "fisher"
            if not x_or_p and not tail:
                second_form_type = "single_xf"
            elif not x_or_p and tail:
                second_form_type = "double_xf"
            elif x_or_p and not tail:
                second_form_type = "single_pf"
            elif x_or_p and tail:
                second_form_type = "double_pf"
            
            context['second_form_type'] = second_form_type

        elif 'xf_value' in request.POST or 'pf_value' in request.POST or 'xf_value_1' in request.POST or 'pf_value_1' in request.POST:
            second_form_type = request.POST.get('second_form_type')

            if second_form_type == "single_xf":
                x_value = request.POST.get('xf_value')
                dff1_value = request.POST.get('dff1_value')
                dff2_value = request.POST.get('dff2_value')
                x_value = float(x_value)
                dff1 = int(dff1_value)
                dff2 = int(dff2_value)
                prob_left = f.cdf(x_value, dff1, dff2)
                prob_right = 1 - prob_left
                result = f"For a F value of: {x_value} with {dff1} and {dff2} degrees of freedom\nProbability to the left (blue): {prob_left:.4f}\nProbability to the right (red): {prob_right:.4f}"
                result = result.replace("\n", "<br>")
                graph_image = plot_single_xf(x_value, dff1, dff2)
                
            elif second_form_type == "double_xf":
                x_value_1 = request.POST.get('xf_value_1')
                x_value_2 = request.POST.get('xf_value_2')
                dff1_value = request.POST.get('dff1_value')
                dff2_value = request.POST.get('dff2_value')
                x_value_1 = float(x_value_1)
                x_value_2 = float(x_value_2)
                x_value_1, x_value_2 = min(x_value_1, x_value_2), max(x_value_1, x_value_2)
                dff1 = int(dff1_value)
                dff2 = int(dff2_value)

                prob_left = f.cdf(x_value_1, dff1, dff2)
                prob_right = 1 - f.cdf(x_value_2, dff1, dff2)
                prob_between = f.cdf(x_value_2, dff1, dff2) - f.cdf(x_value_1, dff1, dff2)
                result = (
                    f"For the values of F: {x_value_1} and {x_value_2} with {dff1} and {dff2} degrees fo freedom\n"
                    f"Probability to the left of {x_value_1} (red): {prob_left:.4f}\n"
                    f"Probability between {x_value_1} and {x_value_2} (blue): {prob_between:.4f}\n"
                    f"Probability to the right of {x_value_2} (red): {prob_right:.4f}"
                )
                result = result.replace("\n", "<br>")
                graph_image = plot_double_xf(x_value_1, x_value_2, dff1, dff2)
                
            elif second_form_type == "single_pf":
                p_value = request.POST.get('pf_value')
                dff1_value = request.POST.get('dff1_value')
                dff2_value = request.POST.get('dff2_value')
                p_value = float(p_value)
                dff1 = int(dff1_value)
                dff2 = int(dff2_value)
                if 0 <= p_value <= 1:
                    x_value = f.ppf(p_value, dff1, dff2)
                    result = (
                            f"For a probability value of: {p_value} (blue) with {dff1} and {dff2} degrees of freedom\n"
                            f"Value of F: {x_value:.4f}\n"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_single_xf(x_value, dff1, dff2)
                else:
                    result =  "Error: The entered probability must be between 0 and 1."

            elif second_form_type == "double_pf":
                p_value_1 = request.POST.get('pf_value_1')
                p_value_2 = request.POST.get('pf_value_2')
                dff1_value = request.POST.get('dff1_value')
                dff2_value = request.POST.get('dff2_value')
                p_value_1 = float(p_value_1)
                p_value_2 = float(p_value_2)
                dff1 = int(dff1_value)
                dff2 = int(dff2_value)
                if 0 <= p_value_1 <= 1 and 0 <= p_value_2 <= 1:
                    p_value_1, p_value_2 = min(p_value_1, p_value_2), max(p_value_1, p_value_2)
                    p_between = p_value_2 - p_value_1
                    x_value_1 = f.ppf(p_value_1, dff1, dff2)
                    x_value_2 = f.ppf(p_value_2, dff1, dff2)
                    result = (
                        f"For probabilities: {p_value_1} and {p_value_2} with {dff1} and {dff2} degrees of freedom\n"
                        f"Value of F at left probability: {x_value_1:.4f}\n"
                        f"Value of F at right probability: {x_value_2:.4f}\n"
                        f"Probability between (blue): {p_between:.4f}"
                    )
                    result = result.replace("\n", "<br>")
                    graph_image = plot_double_xf(x_value_1, x_value_2, dff1, dff2)
                else:
                    result =  "Error: The entered probabilities must be between 0 and 1."
                    
#####################################################################################################################################
    elif tab == "normal_table":
        z_values = np.arange(0, 3.6, 0.1)
        table_data = []
        for z in z_values:
            row_data = [z]
            for i in np.arange(0, 0.1, 0.01):
                prob = norm.cdf(z + i)
                row_data.append(f"{prob:.4f}")
            table_data.append(row_data)
        column_names = ['z'] + [f"{i:.2f}" for i in np.arange(0, 0.1, 0.01)]
        n_table = pd.DataFrame(table_data, columns=column_names)
        n_table_html = n_table.to_html(classes='table table-responsive', index=False, escape=False)
        n_table_html = n_table_html.replace('<th>', '<th style="font-size: 1em; padding: 10px;">')
        n_table_html = n_table_html.replace('<td>', '<td style="font-size: 1em; padding: 10px;">')
        context['n_table'] = n_table_html
    
    elif tab == "t_table":
        degrees_of_freedom = range(1, 121)
        probabilities = [0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995]
        table_data = []
        for df in degrees_of_freedom:
            row_data = [df]
            for prob in probabilities:
                t_value = t.ppf(1 - (1 - prob) / 2, df)
                row_data.append(f"{t_value:.4f}")
            table_data.append(row_data)
        column_names = ['df'] + [f"{prob}" for prob in probabilities]
        t_table = pd.DataFrame(table_data, columns=column_names)
        t_table_html = t_table.to_html(classes='table table-responsive', index=False, escape=False)
        t_table_html = t_table_html.replace('<th>', '<th style="font-size: 1em; padding: 10px;">')
        t_table_html = t_table_html.replace('<td>', '<td style="font-size: 1em; padding: 10px;">')
        context['t_table'] = t_table_html

    elif tab == "chi_table":
        degrees_of_freedom = range(1, 101)
        probabilities = [0.995, 0.99, 0.975, 0.95, 0.90, 0.5, 0.10, 0.05, 0.025, 0.01, 0.005]
        table_data = []
        for df in degrees_of_freedom:
            row_data = [df]
            for prob in probabilities:
                chi_value = chi2.ppf(1 - prob, df)
                row_data.append(f"{chi_value:.4f}")
            table_data.append(row_data)
        column_names = ['df'] + [f"{(1 - prob):.3f}" for prob in probabilities]
        chi_table = pd.DataFrame(table_data, columns=column_names)
        chi_table_html = chi_table.to_html(classes='table table-responsive', index=False, escape=False)
        chi_table_html = chi_table_html.replace('<th>', '<th style="font-size: 1em; padding: 10px;">')
        chi_table_html = chi_table_html.replace('<td>', '<td style="font-size: 1em; padding: 10px;">')
        context['chi_table'] = chi_table_html

    elif tab == "f_table":
        df_numerators = list(range(1, 11)) + [12, 15, 20, 24, 30, 40, 60, 120]
        df_denominators = range(1, 121)
        probabilities = [0.10, 0.05, 0.01]
        tables_html = []
        for prob in probabilities:
            table_data = []
            for df_denom in df_denominators:
                row_data = [df_denom]
                for df_num in df_numerators:
                    f_value = f.ppf(1 - prob, df_num, df_denom)
                    row_data.append(f"{f_value:.4f}")
                table_data.append(row_data)
            column_names = ['Denominator df'] + [f"Num df {df}" for df in df_numerators]
            f_table = pd.DataFrame(table_data, columns=column_names)
            f_table_html = f_table.to_html(classes='table table-responsive', index=False, escape=False)
            f_table_html = f"<h3>F Distribution (α = {prob})</h3>" + f_table_html
            f_table_html = f_table_html.replace('<th>', '<th style="font-size: 1em; padding: 10px;">')
            f_table_html = f_table_html.replace('<td>', '<td style="font-size: 1em; padding: 10px;">')
            tables_html.append(f_table_html)
        context['f_tables'] = ''.join(tables_html)

#################################################################################################################################################################
    context['result'] = result
    context['graph_image'] = graph_image
    context['selected_distribution'] = selected_distribution
    context['x_or_p'] = 'checked' if x_or_p else ''
    context['tail'] = 'checked' if tail else ''
    context['show_second_form'] = show_second_form

    return render(request, "probability/probability.html", context)


def plot_single_x(x1):
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x1), color="lightcoral", alpha=0.6)
    plt.xlabel("z")
    plt.ylabel("Probability Density")
    plt.title("Standard Normal Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_double_x(x1, x2):
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightcoral", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x2), color="lightcoral", alpha=0.6)
    plt.xlabel("z")
    plt.ylabel("Probability Density")
    plt.title("Standard Normal Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_single_xt(x1, df):
    x = np.linspace(-6, 6, 1000)
    y = t.pdf(x, df)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x1), color="lightcoral", alpha=0.6)
    plt.xlabel("t")
    plt.ylabel("Probability Density")
    plt.title("t-Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_double_xt(x1, x2, df):
    x = np.linspace(-6, 6, 1000)
    y = t.pdf(x, df)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightcoral", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x2), color="lightcoral", alpha=0.6)
    plt.xlabel("t")
    plt.ylabel("Probability Density")
    plt.title("t-Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_single_xc(x1, df):
    x = np.linspace(0, 30, 1000)
    y = chi2.pdf(x, df)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x1), color="lightcoral", alpha=0.6)
    plt.xlabel("χ²")
    plt.ylabel("Probability Density")
    plt.title("χ² Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_double_xc(x1, x2, df):
    x = np.linspace(0, 30, 1000)
    y = chi2.pdf(x, df)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightcoral", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x2), color="lightcoral", alpha=0.6)
    plt.xlabel("χ²")
    plt.ylabel("Probability Density")
    plt.title("χ² Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_single_xf(x1, df1, df2):
    x = np.linspace(0, 7, 1000)
    y = f.pdf(x, df1, df2)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x1), color="lightcoral", alpha=0.6)
    plt.xlabel("F")
    plt.ylabel("Probability Density")
    plt.title("F Distribution")
    plt.grid(True)
    return get_graph_as_image()


def plot_double_xf(x1, x2, df1, df2):
    x = np.linspace(0, 7, 1000)
    y = f.pdf(x, df1, df2)
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color="lightblue", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x <= x1), color="lightcoral", alpha=0.6)
    plt.fill_between(x, 0, y, where=(x >= x2), color="lightcoral", alpha=0.6)
    plt.xlabel("F")
    plt.ylabel("Probability Density")
    plt.title("F Distribution")
    plt.grid(True)
    return get_graph_as_image()


def get_graph_as_image():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph_image = base64.b64encode(image_png)
    graph_image = graph_image.decode('utf-8')
    buffer.close()
    return graph_image