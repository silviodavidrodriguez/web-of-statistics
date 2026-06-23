from django.shortcuts import render
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def control(request):
    tab = request.GET.get('tab', 'shew_1')
    subtab = request.GET.get('subtab', None)

    context = {
        'segment': 'control',
        'active_tab': tab,
        'active_subtab': subtab,
        'data_xr_chart': request.session.get('data_xr_chart', ""),
        'results_xr_chart': request.session.get('results_xr_chart', None),
        'graph_xr_chart': request.session.get('graph_xr_chart', None),
        'data_xs_chart': request.session.get('data_xs_chart', ""),
        'results_xs_chart': request.session.get('results_xs_chart', None),
        'graph_xs_chart': request.session.get('graph_xs_chart', None),
        'data_mr_chart': request.session.get('data_mr_chart', ""),
        'results_mr_chart': request.session.get('results_mr_chart', None),
        'graph_mr_chart': request.session.get('graph_mr_chart', None),
        'data_individual_chart': request.session.get('data_individual_chart', ""),
        'results_individual_chart': request.session.get('results_individual_chart', None),
        'graph_individual_chart': request.session.get('graph_individual_chart', None),
        'data_p_chart': request.session.get('data_p_chart', ""),
        'results_p_chart': request.session.get('results_p_chart', None),
        'graph_p_chart': request.session.get('graph_p_chart', None),
        'data_np_chart': request.session.get('data_np_chart', ""),
        'results_np_chart': request.session.get('results_np_chart', None),
        'graph_np_chart': request.session.get('graph_np_chart', None),
        'data_c_chart': request.session.get('data_c_chart', ""),
        'results_c_chart': request.session.get('results_c_chart', None),
        'graph_c_chart': request.session.get('graph_c_chart', None),
        'data_u_chart': request.session.get('data_u_chart', ""),
        'results_u_chart': request.session.get('results_u_chart', None),
        'graph_u_chart': request.session.get('graph_u_chart', None),
        'data_cusum_chart': request.session.get('data_cusum_chart', ""),
        'results_cusum_chart': request.session.get('results_cusum_chart', None),
        'graph_cusum_chart': request.session.get('graph_cusum_chart', None),
        'mu_0': request.session.get('mu_0', None),
        'k_value': request.session.get('k_value', None),
        'h_value': request.session.get('h_value', None),
        'data_ewma_chart': request.session.get('data_ewma_chart', ""),
        'results_ewma_chart': request.session.get('results_ewma_chart', None),
        'graph_ewma_chart': request.session.get('graph_ewma_chart', None),
        'mu_0_ewma': request.session.get('mu_0_ewma', None),
        'lambda_value': request.session.get('lambda_value', None),
        'data_precontrol_chart': request.session.get('data_precontrol_chart', ""),
        'results_precontrol_chart': request.session.get('results_precontrol_chart', None),
        'graph_precontrol_chart': request.session.get('graph_precontrol_chart', None),
        'nominal_value': request.session.get('nominal_value', None),
        'tolerance_value': request.session.get('tolerance_value', None),
        }
    
    if request.method == "POST" and request.POST.get("clear_xr_chart") == "true":
        if 'data_xr_chart' in request.session:
            del request.session['data_xr_chart']
        if 'results_xr_chart' in request.session:
            del request.session['results_xr_chart']
        if 'graph_xr_chart' in request.session:
            del request.session['graph_xr_chart']
        context['data_xr_chart'] = ""
        context['results_xr_chart'] = None
        context['graph_xr_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_xs_chart") == "true":
        if 'data_xs_chart' in request.session:
            del request.session['data_xs_chart']
        if 'results_xs_chart' in request.session:
            del request.session['results_xs_chart']
        if 'graph_xs_chart' in request.session:
            del request.session['graph_xs_chart']
        context['data_xs_chart'] = ""
        context['results_xs_chart'] = None
        context['graph_xs_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_mr_chart") == "true":
        if 'data_mr_chart' in request.session:
            del request.session['data_mr_chart']
        if 'results_mr_chart' in request.session:
            del request.session['results_mr_chart']
        if 'graph_mr_chart' in request.session:
            del request.session['graph_mr_chart']
        context['data_mr_chart'] = ""
        context['results_mr_chart'] = None
        context['graph_mr_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_individual_chart") == "true":
        if 'data_individual_chart' in request.session:
            del request.session['data_individual_chart']
        if 'results_individual_chart' in request.session:
            del request.session['results_individual_chart']
        if 'graph_individual_chart' in request.session:
            del request.session['graph_individual_chart']
        context['data_individual_chart'] = ""
        context['results_individual_chart'] = None
        context['graph_individual_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_p_chart") == "true":
        if 'data_p_chart' in request.session:
            del request.session['data_p_chart']
        if 'results_p_chart' in request.session:
            del request.session['results_p_chart']
        if 'graph_p_chart' in request.session:
            del request.session['graph_p_chart']
        context['data_p_chart'] = ""
        context['results_p_chart'] = None
        context['graph_p_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_np_chart") == "true":
        if 'data_np_chart' in request.session:
            del request.session['data_np_chart']
        if 'results_np_chart' in request.session:
            del request.session['results_np_chart']
        if 'graph_np_chart' in request.session:
            del request.session['graph_np_chart']
        context['data_np_chart'] = ""
        context['results_np_chart'] = None
        context['graph_np_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_c_chart") == "true":
        if 'data_c_chart' in request.session:
            del request.session['data_c_chart']
        if 'results_c_chart' in request.session:
            del request.session['results_c_chart']
        if 'graph_c_chart' in request.session:
            del request.session['graph_c_chart']
        context['data_c_chart'] = ""
        context['results_c_chart'] = None
        context['graph_c_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_u_chart") == "true":
        if 'data_u_chart' in request.session:
            del request.session['data_u_chart']
        if 'results_u_chart' in request.session:
            del request.session['results_u_chart']
        if 'graph_u_chart' in request.session:
            del request.session['graph_u_chart']
        context['data_u_chart'] = ""
        context['results_u_chart'] = None
        context['graph_u_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_cusum_chart") == "true":
        if 'data_cusum_chart' in request.session:
            del request.session['data_cusum_chart']
        if 'results_cusum_chart' in request.session:
            del request.session['results_cusum_chart']
        if 'mu_0' in request.session:
            del request.session['mu_0']
        if 'k_value' in request.session:
            del request.session['k_value']
        if 'h_value' in request.session:
            del request.session['h_value']
        if 'graph_cusum_chart' in request.session:
            del request.session['graph_cusum_chart']
        context['data_cusum_chart'] = ""
        context['results_cusum_chart'] = None
        context['mu_0'] = None
        context['k_value'] = None
        context['h_value'] = None
        context['graph_cusum_chart'] = None
        return render(request, "control/control.html", context)
    
    if request.method == "POST" and request.POST.get("clear_ewma_chart") == "true":
        if 'data_ewma_chart' in request.session:
            del request.session['data_ewma_chart']
        if 'results_ewma_chart' in request.session:
            del request.session['results_ewma_chart']
        if 'mu_0_ewma' in request.session:
            del request.session['mu_0_ewma']
        if 'lambda_value' in request.session:
            del request.session['lambda_value']
        if 'graph_ewma_chart' in request.session:
            del request.session['graph_ewma_chart']
        context['data_ewma_chart'] = ""
        context['results_ewma_chart'] = None
        context['mu_0_ewma'] = None
        context['lambda_value'] = None
        context['graph_ewma_chart'] = None
        return render(request, "control/control.html", context)

    if request.method == "POST" and request.POST.get("clear_precontrol_chart") == "true":
        if 'data_precontrol_chart' in request.session:
            del request.session['data_precontrol_chart']
        if 'results_precontrol_chart' in request.session:
            del request.session['results_precontrol_chart']
        if 'nominal_value' in request.session:
            del request.session['nominal_value']
        if 'tolerance_value' in request.session:
            del request.session['tolerance_value']
        if 'graph_precontrol_chart' in request.session:
            del request.session['graph_precontrol_chart']
        context['data_precontrol_chart'] = ""
        context['results_precontrol_chart'] = None
        context['nominal_value'] = None
        context['tolerance_value'] = None
        context['graph_precontrol_chart'] = None
        return render(request, "control/control.html", context)

###################################################################################################################################    
    if request.method == "POST" and tab == "shew_1" and subtab == "xr_chart":
        data_xr_chart = request.POST.get('data_xr_chart')

        if not data_xr_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_xr_chart'] = None
            context['graph_xr_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_xr_chart = data_xr_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_xr_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            subgroup_sizes = [len(row) for row in rows]

            if len(set(subgroup_sizes)) > 1:
                raise ValueError("All subgroups must have the same size.")

            n = subgroup_sizes[0]

            valid_n_values = {2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25}

            if n not in valid_n_values:
                raise ValueError(f"Invalid subgroup size 'n' = {n}. Valid values are: {', '.join(map(str, valid_n_values))}.")

            A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 15: 0.223, 25: 0.153}[n]
            D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223, 15: 0.347, 25: 0.459}[n]
            D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.964, 9: 1.916, 10: 1.777, 15: 1.653, 25: 1.541}[n]

            subgroup_means = [np.mean(row) for row in rows]
            subgroup_ranges = [np.ptp(row) for row in rows]
            overall_mean = np.mean(subgroup_means)
            mean_range = np.mean(subgroup_ranges)

            UCL_X = overall_mean + A2 * mean_range
            LCL_X = overall_mean - A2 * mean_range
            UCL_R = D4 * mean_range
            LCL_R = D3 * mean_range

            results_xr_chart = {
                'subgroup_means': [round_to_significant(val) for val in subgroup_means],
                'subgroup_ranges': [round_to_significant(val) for val in subgroup_ranges],
                'overall_mean': round_to_significant(overall_mean),
                'mean_range': round_to_significant(mean_range),
                'UCL_X': round_to_significant(UCL_X),
                'LCL_X': round_to_significant(LCL_X),
                'UCL_R': round_to_significant(UCL_R),
                'LCL_R': round_to_significant(LCL_R),
            }

            fig, axs = plt.subplots(2, 1)
            subgroup_numbers = range(1, len(subgroup_means) + 1)
            axs[0].plot(subgroup_numbers, subgroup_means, marker='o')
            axs[0].axhline(overall_mean, color='green', linestyle='--')
            axs[0].axhline(UCL_X, color='red', linestyle='--')
            axs[0].axhline(LCL_X, color='red', linestyle='--')
            axs[0].set_title('X-bar Chart')
            axs[0].set_xlabel('Subgroup')
            axs[0].set_ylabel('Mean')

            axs[1].plot(subgroup_numbers, subgroup_ranges, marker='o')
            axs[1].axhline(mean_range, color='green', linestyle='--')
            axs[1].axhline(UCL_R, color='red', linestyle='--')
            axs[1].axhline(LCL_R, color='red', linestyle='--')
            axs[1].set_title('R Chart')
            axs[1].set_xlabel('Subgroup')
            axs[1].set_ylabel('Range')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_xr_chart'] = results_xr_chart
            context['data_xr_chart'] = data_xr_chart
            context['graph_xr_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_xr_chart'] = results_xr_chart
            request.session['data_xr_chart'] = data_xr_chart
            request.session['graph_xr_chart'] = context['graph_xr_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_xr_chart'] = None
            context['graph_xr_chart'] = None
            return render(request, "control/control.html", context)

#################################################################################################################################
    if request.method == "POST" and tab == "shew_1" and subtab == "xs_chart":
        data_xs_chart = request.POST.get('data_xs_chart')

        if not data_xs_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_xs_chart'] = None
            context['graph_xs_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_xs_chart = data_xs_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_xs_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            subgroup_sizes = [len(row) for row in rows]

            if len(set(subgroup_sizes)) > 1:
                raise ValueError("All subgroups must have the same size.")

            n = subgroup_sizes[0]

            valid_n_values = {2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25}

            if n not in valid_n_values:
                raise ValueError(f"Invalid subgroup size 'n' = {n}. Valid values are: {', '.join(map(str, valid_n_values))}.")

            A3 = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975, 15: 0.789, 25: 0.606}[n]
            B3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284, 15: 0.428, 25: 0.565}[n]
            B4 = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716, 15: 1.572, 25: 1.435}[n]

            subgroup_means = [np.mean(row) for row in rows]
            subgroup_stdevs = [np.std(row, ddof=1) for row in rows]
            overall_mean = np.mean(subgroup_means)
            mean_stdev = np.mean(subgroup_stdevs)

            UCL_X = overall_mean + (A3 * mean_stdev)
            LCL_X = overall_mean - (A3 * mean_stdev)
            UCL_S = B4 * mean_stdev
            LCL_S = B3 * mean_stdev

            results_xs_chart = {
                'subgroup_means': [round_to_significant(val) for val in subgroup_means],
                'subgroup_stdevs': [round_to_significant(val) for val in subgroup_stdevs],
                'overall_mean': round_to_significant(overall_mean),
                'mean_stdev': round_to_significant(mean_stdev),
                'UCL_X': round_to_significant(UCL_X),
                'LCL_X': round_to_significant(LCL_X),
                'UCL_S': round_to_significant(UCL_S),
                'LCL_S': round_to_significant(LCL_S),
            }

            fig, axs = plt.subplots(2, 1)
            subgroup_numbers = range(1, len(subgroup_means) + 1)
            
            axs[0].plot(subgroup_numbers, subgroup_means, marker='o')
            axs[0].axhline(overall_mean, color='green', linestyle='--')
            axs[0].axhline(UCL_X, color='red', linestyle='--')
            axs[0].axhline(LCL_X, color='red', linestyle='--')
            axs[0].set_title('X-bar Chart')
            axs[0].set_xlabel('Subgroup')
            axs[0].set_ylabel('Mean')

            axs[1].plot(subgroup_numbers, subgroup_stdevs, marker='o')
            axs[1].axhline(mean_stdev, color='green', linestyle='--')
            axs[1].axhline(UCL_S, color='red', linestyle='--')
            axs[1].axhline(LCL_S, color='red', linestyle='--')
            axs[1].set_title('S Chart')
            axs[1].set_xlabel('Subgroup')
            axs[1].set_ylabel('Standard Deviation')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_xs_chart'] = results_xs_chart
            context['data_xs_chart'] = data_xs_chart
            context['graph_xs_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_xs_chart'] = results_xs_chart
            request.session['data_xs_chart'] = data_xs_chart
            request.session['graph_xs_chart'] = context['graph_xs_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_xs_chart'] = None
            context['graph_xs_chart'] = None
            return render(request, "control/control.html", context)

######################################################################################################################
    if request.method == "POST" and tab == "shew_1" and subtab == "mr_chart":
        data_mr_chart = request.POST.get('data_mr_chart')

        if not data_mr_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_mr_chart'] = None
            context['graph_mr_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_mr_chart = data_mr_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_mr_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            subgroup_sizes = [len(row) for row in rows]

            if len(set(subgroup_sizes)) > 1:
                raise ValueError("All subgroups must have the same size.")

            n = subgroup_sizes[0]

            valid_n_values = {2, 3, 4, 5, 6, 7, 8, 9, 10}

            if n not in valid_n_values:
                raise ValueError(f"Invalid subgroup size 'n' = {n}. Valid values are: {', '.join(map(str, valid_n_values))}.")

            A2 = {2: 1.880, 3: 1.187, 4: 0.796, 5: 0.691, 6: 0.548, 7: 0.508, 8: 0.433, 9: 0.412, 10: 362}[n]
            D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}[n]
            D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}[n]

            subgroup_medians = [np.median(row) for row in rows]
            subgroup_ranges = [np.ptp(row) for row in rows]
            overall_median = np.mean(subgroup_medians)
            mean_range = np.mean(subgroup_ranges)

            UCL_M = overall_median + (A2 * mean_range)
            LCL_M = overall_median - (A2 * mean_range)
            UCL_R = D4 * mean_range
            LCL_R = D3 * mean_range

            results_mr_chart = {
                'subgroup_medians': [round_to_significant(val) for val in subgroup_medians],
                'subgroup_ranges': [round_to_significant(val) for val in subgroup_ranges],
                'overall_median': round_to_significant(overall_median),
                'mean_range': round_to_significant(mean_range),
                'UCL_M': round_to_significant(UCL_M),
                'LCL_M': round_to_significant(LCL_M),
                'UCL_R': round_to_significant(UCL_R),
                'LCL_R': round_to_significant(LCL_R),
            }

            fig, axs = plt.subplots(2, 1)
            subgroup_numbers = range(1, len(subgroup_medians) + 1)

            axs[0].plot(subgroup_numbers, subgroup_medians, marker='o')
            axs[0].axhline(overall_median, color='green', linestyle='--')
            axs[0].axhline(UCL_M, color='red', linestyle='--')
            axs[0].axhline(LCL_M, color='red', linestyle='--')
            axs[0].set_title('Median Chart')
            axs[0].set_xlabel('Subgroup')
            axs[0].set_ylabel('Median')

            axs[1].plot(subgroup_numbers, subgroup_ranges, marker='o')
            axs[1].axhline(mean_range, color='green', linestyle='--')
            axs[1].axhline(UCL_R, color='red', linestyle='--')
            axs[1].axhline(LCL_R, color='red', linestyle='--')
            axs[1].set_title('R Chart')
            axs[1].set_xlabel('Subgroup')
            axs[1].set_ylabel('Range')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_mr_chart'] = results_mr_chart
            context['data_mr_chart'] = data_mr_chart
            context['graph_mr_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_mr_chart'] = results_mr_chart
            request.session['data_mr_chart'] = data_mr_chart
            request.session['graph_mr_chart'] = context['graph_mr_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_mr_chart'] = None
            context['graph_mr_chart'] = None
            return render(request, "control/control.html", context)

########################################################################################################################
    if request.method == "POST" and tab == "shew_1" and subtab == "individual_chart":
        data_individual_chart = request.POST.get('data_individual_chart')

        if not data_individual_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_individual_chart'] = None
            context['graph_individual_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_individual_chart = data_individual_chart.replace('\r', '').strip()
            rows = [line.strip() for line in data_individual_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            if any(not is_valid_number(row) for row in rows):
                raise ValueError("Non-numeric values detected in the dataset.")

            data_values = [float(row) for row in rows]

            if len(data_values) < 2:
                raise ValueError("At least two data points are required for Individuals chart.")

            overall_mean = np.mean(data_values)

            moving_ranges = [abs(data_values[i] - data_values[i - 1]) for i in range(1, len(data_values))]
            mean_moving_range = np.mean(moving_ranges)

            E2 = 2.660

            UCL_I = overall_mean + (E2 * mean_moving_range)
            LCL_I = overall_mean - (E2 * mean_moving_range)

            UCL_MR = 3.267 * mean_moving_range
            LCL_MR = 0

            results_individual_chart = {
                'overall_mean': round_to_significant(overall_mean),
                'mean_moving_range': round_to_significant(mean_moving_range),
                'UCL_I': round_to_significant(UCL_I),
                'LCL_I': round_to_significant(LCL_I),
                'UCL_MR': round_to_significant(UCL_MR),
                'LCL_MR': round_to_significant(LCL_MR),
            }

            fig, axs = plt.subplots(2, 1)
            data_points = range(1, len(data_values) + 1)
            mr_points = range(2, len(data_values) + 1)

            axs[0].plot(data_points, data_values, marker='o')
            axs[0].axhline(overall_mean, color='green', linestyle='--')
            axs[0].axhline(UCL_I, color='red', linestyle='--')
            axs[0].axhline(LCL_I, color='red', linestyle='--')
            axs[0].set_title('Individuals Chart')
            axs[0].set_xlabel('Observation')
            axs[0].set_ylabel('Value')

            axs[1].plot(mr_points, moving_ranges, marker='o')
            axs[1].axhline(mean_moving_range, color='green', linestyle='--')
            axs[1].axhline(UCL_MR, color='red', linestyle='--')
            axs[1].axhline(LCL_MR, color='red', linestyle='--')
            axs[1].set_title('Moving Range Chart')
            axs[1].set_xlabel('Observation')
            axs[1].set_ylabel('Range')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_individual_chart'] = results_individual_chart
            context['data_individual_chart'] = data_individual_chart
            context['graph_individual_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_individual_chart'] = results_individual_chart
            request.session['data_individual_chart'] = data_individual_chart
            request.session['graph_individual_chart'] = context['graph_individual_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_individual_chart'] = None
            context['graph_individual_chart'] = None
            return render(request, "control/control.html", context)

########################################################################################################################
    if request.method == "POST" and tab == "shew_2" and subtab == "p_chart":
        data_p_chart = request.POST.get('data_p_chart')

        if not data_p_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_p_chart'] = None
            context['graph_p_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_p_chart = data_p_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_p_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if len(row) != 2:
                    raise ValueError("Each row must have exactly two values: sample size and number of defects.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            sample_sizes = [float(row[0]) for row in rows]
            defects = [float(row[1]) for row in rows]

            if any(size <= 0 for size in sample_sizes):
                raise ValueError("Sample sizes must be greater than zero.")
            if any(defect < 0 for defect in defects):
                raise ValueError("Number of defects cannot be negative.")

            p_values = [defects[i] / sample_sizes[i] for i in range(len(sample_sizes))]

            total_defects = sum(defects)
            total_samples = sum(sample_sizes)
            mean_p = total_defects / total_samples

            sigma_p = [np.sqrt(mean_p * (1 - mean_p) / size) for size in sample_sizes]
            UCL_p = [mean_p + 3 * s for s in sigma_p]
            LCL_p = [max(mean_p - 3 * s, 0) for s in sigma_p]

            results_p_chart = {
                'mean_p': round_to_significant(mean_p),
            }

            fig, ax = plt.subplots()
            sample_numbers = range(1, len(sample_sizes) + 1)

            ax.plot(sample_numbers, p_values, marker='o')
            ax.plot(sample_numbers, [mean_p] * len(sample_sizes), linestyle='--', color='green')
            ax.plot(sample_numbers, UCL_p, linestyle='--', color='red')
            ax.plot(sample_numbers, LCL_p, linestyle='--', color='red')
            ax.set_title('p Chart')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Proportion of Defective Items (p)')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_p_chart'] = results_p_chart
            context['data_p_chart'] = data_p_chart
            context['graph_p_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_p_chart'] = results_p_chart
            request.session['data_p_chart'] = data_p_chart
            request.session['graph_p_chart'] = context['graph_p_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_p_chart'] = None
            context['graph_p_chart'] = None
            return render(request, "control/control.html", context)

############################################################################
    if request.method == "POST" and tab == "shew_2" and subtab == "np_chart":
        data_np_chart = request.POST.get('data_np_chart')

        if not data_np_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_np_chart'] = None
            context['graph_np_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_np_chart = data_np_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_np_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if len(row) != 2:
                    raise ValueError("Each row must have exactly two values: sample size and number of defects.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            sample_sizes = [float(row[0]) for row in rows]
            defects = [float(row[1]) for row in rows]

            if len(set(sample_sizes)) == 1:
                n = sample_sizes[0]
            else:
                n = None

            if any(defect < 0 for defect in defects):
                raise ValueError("Number of defects cannot be negative.")

            mean_d = np.mean(defects)
            
            if n is not None:
                
                sigma_np = np.sqrt(n * mean_d * (1 - (mean_d / n)))
                
                UCL_np = mean_d + 3 * sigma_np
                LCL_np = max(mean_d - 3 * sigma_np, 0)

                results_np_chart = {
                    'mean_d': round_to_significant(mean_d),
                    'UCL_np': round_to_significant(UCL_np),
                    'LCL_np': round_to_significant(LCL_np),
                }

                fig, ax = plt.subplots()
                sample_numbers = range(1, len(defects) + 1)

                ax.plot(sample_numbers, defects, marker='o')
                ax.axhline(mean_d, color='green', linestyle='--')
                ax.axhline(UCL_np, color='red', linestyle='--')
                ax.axhline(LCL_np, color='red', linestyle='--')
                ax.set_title('np Chart')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Number of Defects')

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()

                context['results_np_chart'] = results_np_chart
                context['data_np_chart'] = data_np_chart
                context['graph_np_chart'] = f"data:image/png;base64,{graph_url}"
                request.session['results_np_chart'] = results_np_chart
                request.session['data_np_chart'] = data_np_chart
                request.session['graph_np_chart'] = context['graph_np_chart']

            else:
                raise ValueError("Inconsistent sample sizes detected. For np chart, the sample size must be constant.")

        except ValueError as e:
            context['error'] = str(e)
            context['results_np_chart'] = None
            context['graph_np_chart'] = None
            return render(request, "control/control.html", context)

#####################################################################################
    if request.method == "POST" and tab == "shew_2" and subtab == "c_chart":
        data_c_chart = request.POST.get('data_c_chart')

        if not data_c_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_c_chart'] = None
            context['graph_c_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_c_chart = data_c_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_c_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if len(row) != 1:
                    raise ValueError("Each row must have exactly one value: number of defects.")
                if not is_valid_number(row[0]):
                    raise ValueError("Non-numeric values detected in the dataset.")

            defects = [float(row[0]) for row in rows]

            mean_c = np.mean(defects)
            
            sigma_c = np.sqrt(mean_c)

            UCL_c = mean_c + 3 * sigma_c
            LCL_c = max(mean_c - 3 * sigma_c, 0)

            results_c_chart = {
                'mean_c': round_to_significant(mean_c),
                'UCL_c': round_to_significant(UCL_c),
                'LCL_c': round_to_significant(LCL_c),
            }

            fig, ax = plt.subplots()
            sample_numbers = range(1, len(defects) + 1)

            ax.plot(sample_numbers, defects, marker='o')
            ax.axhline(mean_c, color='green', linestyle='--')
            ax.axhline(UCL_c, color='red', linestyle='--')
            ax.axhline(LCL_c, color='red', linestyle='--')
            ax.set_title('c Chart')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Number of Defects')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_c_chart'] = results_c_chart
            context['data_c_chart'] = data_c_chart
            context['graph_c_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_c_chart'] = results_c_chart
            request.session['data_c_chart'] = data_c_chart
            request.session['graph_c_chart'] = context['graph_c_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_c_chart'] = None
            context['graph_c_chart'] = None
            return render(request, "control/control.html", context)

######################################################################################
    if request.method == "POST" and tab == "shew_2" and subtab == "u_chart":
        data_u_chart = request.POST.get('data_u_chart')

        if not data_u_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_u_chart'] = None
            context['graph_u_chart'] = None
            return render(request, "control/control.html", context)

        try:
            data_u_chart = data_u_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_u_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if len(row) != 2:
                    raise ValueError("Each row must have exactly two values: sample size and number of defectives.")
                if not (is_valid_number(row[0]) and is_valid_number(row[1])):
                    raise ValueError("Non-numeric values detected in the dataset.")

            sample_sizes = [float(row[0]) for row in rows]
            defectives = [float(row[1]) for row in rows]

            u_values = [defectives[i] / sample_sizes[i] for i in range(len(sample_sizes))]
            mean_u = np.mean(u_values)

            UCL_u = [mean_u + 3 * np.sqrt(mean_u / size) for size in sample_sizes]
            LCL_u = [max(mean_u - 3 * np.sqrt(mean_u / size), 0) for size in sample_sizes]

            results_u_chart = {
                'mean_u': round_to_significant(mean_u),
            }

            fig, ax = plt.subplots()
            sample_numbers = range(1, len(u_values) + 1)

            ax.plot(sample_numbers, u_values, marker='o', label='u values')
            ax.plot(sample_numbers, [mean_u] * len(sample_sizes), linestyle='--', color='green')
            ax.plot(sample_numbers, UCL_u, linestyle='--', color='red')
            ax.plot(sample_numbers, LCL_u, linestyle='--', color='red')
            ax.set_title('u Chart')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Defects per Unit (u)')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_u_chart'] = results_u_chart
            context['data_u_chart'] = data_u_chart
            context['graph_u_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_u_chart'] = results_u_chart
            request.session['data_u_chart'] = data_u_chart
            request.session['graph_u_chart'] = context['graph_u_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_u_chart'] = None
            context['graph_u_chart'] = None
            return render(request, "control/control.html", context)

######################################################################################
    if request.method == "POST" and tab == "ewma_cusum" and subtab == "cusum_chart":
        data_cusum_chart = request.POST.get('data_cusum_chart')
        mu_0 = request.POST.get('mu_0')
        k_value = request.POST.get('k_value')
        h_value = request.POST.get('h_value')

        if not data_cusum_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_cusum_chart'] = None
            context['graph_cusum_chart'] = None
            return render(request, "control/control.html", context)

        try:
            k_value = float(k_value)
            h_value = float(h_value)

            data_cusum_chart = data_cusum_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_cusum_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            if not mu_0 or mu_0.strip() == '':
                global_mean = np.mean([value for row in rows for value in row])
                mu_0 = global_mean
                context['warning'] = "No target mean (mu_0) was provided. Using global mean of the data as default."
            else:
                mu_0 = float(mu_0)

            subgroup_means = [np.mean(row) for row in rows]

            cusum_plus = [0]
            cusum_minus = [0]

            for i in range(1, len(subgroup_means)):
                cusum_plus_value = max(0, cusum_plus[-1] + (subgroup_means[i] - mu_0 - k_value))
                cusum_minus_value = -1*(max(0, cusum_minus[-1] + (mu_0 - subgroup_means[i] - k_value)))
                cusum_plus.append(cusum_plus_value)
                cusum_minus.append(cusum_minus_value)

            results_cusum_chart = {
                'mu_0': round_to_significant(mu_0),
                'k_value': k_value,
                'h_value': h_value,
            }

            fig, ax = plt.subplots()
            subgroup_numbers = range(1, len(subgroup_means) + 1)

            ax.plot(subgroup_numbers, cusum_plus, marker='o', color='green')
            ax.plot(subgroup_numbers, cusum_minus, marker='o', color='blue')

            ax.axhline(h_value, color='red', linestyle='--')
            ax.axhline(-h_value, color='red', linestyle='--')
            ax.axhline(0, color='black', linestyle='-')
            ax.set_title('CUSUM Chart')
            ax.set_xlabel('Subgroup')
            ax.set_ylabel('CUSUM Value')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_cusum_chart'] = results_cusum_chart
            context['data_cusum_chart'] = data_cusum_chart
            context['mu_0'] = mu_0
            context['k_value'] = k_value
            context['h_value'] = h_value
            context['graph_cusum_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_cusum_chart'] = results_cusum_chart
            request.session['data_cusum_chart'] = data_cusum_chart
            request.session['mu_0'] = mu_0
            request.session['k_value'] = k_value
            request.session['h_value'] = h_value
            request.session['graph_cusum_chart'] = context['graph_cusum_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_cusum_chart'] = None
            context['graph_cusum_chart'] = None
            return render(request, "control/control.html", context)

######################################################################################
    if request.method == "POST" and tab == "ewma_cusum" and subtab == "ewma_chart":
        data_ewma_chart = request.POST.get('data_ewma_chart')
        mu_0_ewma = request.POST.get('mu_0_ewma')
        lambda_value = request.POST.get('lambda_value')

        if not data_ewma_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_ewma_chart'] = None
            context['graph_ewma_chart'] = None
            return render(request, "control/control.html", context)

        try:
            lambda_value = float(lambda_value)
            if not (0 < lambda_value <= 1):
                raise ValueError("Lambda must be between 0 and 1.")

            data_ewma_chart = data_ewma_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_ewma_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            if not mu_0_ewma or mu_0_ewma.strip() == '':
                global_mean = np.mean([value for row in rows for value in row])
                mu_0_ewma = global_mean
                context['warning'] = "No target mean (mu_0) was provided. Using global mean of the data as default."
            else:
                mu_0_ewma = float(mu_0_ewma)

            subgroup_means = [np.mean(row) for row in rows]
            ewma_values = [mu_0_ewma]

            for i in range(1, len(subgroup_means)):
                ewma_value = lambda_value * subgroup_means[i] + (1 - lambda_value) * ewma_values[-1]
                ewma_values.append(ewma_value)

            overall_mean = np.mean(subgroup_means)
            subgroup_std = np.std(subgroup_means, ddof=1)
            n = len(rows[0])

            ucl_dynamic = []
            lcl_dynamic = []
            for i in range(1, len(subgroup_means) + 1):
                scaling_factor = np.sqrt((lambda_value / (n * (2 - lambda_value))) * (1 - (1 - lambda_value) ** (2 * i)))
                ucl_i = mu_0_ewma + 3 * subgroup_std * scaling_factor
                lcl_i = mu_0_ewma - 3 * subgroup_std * scaling_factor
                ucl_dynamic.append(ucl_i)
                lcl_dynamic.append(lcl_i)

            results_ewma_chart = {
                'overall_mean': round_to_significant(overall_mean),
                'lambda_value': lambda_value,
                'mu_0_ewma': round_to_significant(mu_0_ewma),
            }

            fig, ax = plt.subplots()
            subgroup_numbers = range(1, len(subgroup_means) + 1)

            ax.plot(subgroup_numbers, ewma_values, marker='o', color='blue')
            ax.plot(subgroup_numbers, ucl_dynamic, linestyle='--', color='red')
            ax.plot(subgroup_numbers, lcl_dynamic, linestyle='--', color='red')
            ax.axhline(mu_0_ewma, color='black', linestyle='-')

            ax.set_title('EWMA Chart with Dynamic Control Limits')
            ax.set_xlabel('Subgroup')
            ax.set_ylabel('EWMA Value')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_ewma_chart'] = results_ewma_chart
            context['data_ewma_chart'] = data_ewma_chart
            context['mu_0_ewma'] = mu_0_ewma
            context['lambda_value'] = lambda_value
            context['graph_ewma_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_ewma_chart'] = results_ewma_chart
            request.session['data_ewma_chart'] = data_ewma_chart
            request.session['mu_0_ewma'] = mu_0_ewma
            request.session['lambda_value'] = lambda_value
            request.session['graph_ewma_chart'] = context['graph_ewma_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_ewma_chart'] = None
            context['graph_ewma_chart'] = None
            return render(request, "control/control.html", context)

######################################################################################
    if request.method == "POST" and tab == "precontrol_chart":
        data_precontrol_chart = request.POST.get('data_precontrol_chart')
        nominal_value = request.POST.get('nominal_value')
        tolerance_value = request.POST.get('tolerance_value')

        if not data_precontrol_chart.strip():
            context['error'] = "Please enter data before calculating."
            context['results_precontrol_chart'] = None
            context['graph_precontrol_chart'] = None
            return render(request, "control/control.html", context)

        try:
            nominal_value = float(nominal_value)
            tolerance_value = float(tolerance_value)

            if tolerance_value <= 0:
                raise ValueError("Tolerance must be a positive value.")

            data_precontrol_chart = data_precontrol_chart.replace('\r', '').strip()
            rows = [row.split('\t') for row in data_precontrol_chart.split('\n')]

            def is_valid_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            for row in rows:
                if any(cell.strip() == '' for cell in row):
                    raise ValueError("Missing values detected in the dataset.")
                if not all(is_valid_number(cell) for cell in row):
                    raise ValueError("Non-numeric values detected in the dataset.")

            rows = [[float(cell) for cell in row] for row in rows]

            ust = nominal_value + tolerance_value
            lst = nominal_value - tolerance_value
            green_upper = nominal_value + tolerance_value / 2
            green_lower = nominal_value - tolerance_value / 2

            subgroup_means = [np.mean(row) for row in rows]

            results_precontrol_chart = {
                'nominal_value': round_to_significant(nominal_value),
                'tolerance_value': round_to_significant(tolerance_value),
                'ust': round_to_significant(ust),
                'lst': round_to_significant(lst),
            }

            fig, ax = plt.subplots()
            subgroup_numbers = range(1, len(subgroup_means) + 1)

            ax.plot(subgroup_numbers, subgroup_means, marker='o', color='blue')
            ax.axhline(nominal_value, color='black', linestyle='-')
            ax.axhline(ust, color='red', linestyle='--')
            ax.axhline(lst, color='red', linestyle='--')
            ax.axhline(green_upper, color='green', linestyle='--')
            ax.axhline(green_lower, color='green', linestyle='--')

            ax.set_title('Precontrol Chart')
            ax.set_xlabel('Subgroup')
            ax.set_ylabel('Measurement Value')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            context['results_precontrol_chart'] = results_precontrol_chart
            context['data_precontrol_chart'] = data_precontrol_chart
            context['nominal_value'] = nominal_value
            context['tolerance_value'] = tolerance_value
            context['graph_precontrol_chart'] = f"data:image/png;base64,{graph_url}"
            request.session['results_precontrol_chart'] = results_precontrol_chart
            request.session['data_precontrol_chart'] = data_precontrol_chart
            request.session['nominal_value'] = nominal_value
            request.session['tolerance_value'] = tolerance_value
            request.session['graph_precontrol_chart'] = context['graph_precontrol_chart']

        except ValueError as e:
            context['error'] = str(e)
            context['results_precontrol_chart'] = None
            context['graph_precontrol_chart'] = None
            return render(request, "control/control.html", context)

#######################################################################################

    return render(request, "control/control.html", context)


def round_to_significant(value, significant_digits=4):
    if value == 0:
        return 0
    else:
        return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)