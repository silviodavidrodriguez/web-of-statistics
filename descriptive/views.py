from django.shortcuts import render
import numpy as np
from scipy import stats
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re

def descriptive(request):
    tab = request.GET.get('tab', 'bulk')

    context = {
        'segment': 'descriptive',
        'active_tab': tab,
        'data': request.session.get('data', ""),
        'results': request.session.get('results', None),
        'graph': request.session.get('graph', None),
        'graph_h': request.session.get('graph_h', None),
        'boxplot': request.session.get('boxplot', None),
        'headers': request.session.get('headers', None),
        'use_first_row_as_header': 'checked' if request.session.get('use_first_row_as_header', False) else '',
    }

    if request.method == "POST" and request.POST.get("clear") == "true":
        if 'data' in request.session:
            del request.session['data']
        if 'results' in request.session:
            del request.session['results']
        if 'graph' in request.session:
            del request.session['graph']
        if 'graph_h' in request.session:
            del request.session['graph_h']
        if 'boxplot' in request.session:
            del request.session['boxplot']
        if 'headers' in request.session:
            del request.session['headers']
        if 'use_first_row_as_header' in request.session:
            request.session.pop('use_first_row_as_header', False)
        context['data'] = ""
        context['results'] = None
        context['graph'] = None
        context['graph_h'] = None
        context['boxplot'] = None
        context['use_first_row_as_header'] = False
        return render(request, "descriptive/descriptive.html", context)

####################################################################################################################
    if request.method == "POST" and tab == "bulk":
        data = request.POST.get('data')
        use_first_row_as_header = request.POST.get('use_first_row_as_header') == 'on'

        if not data.strip():
            context['error'] = "Please enter data before calculating."
            context['results'] = None
            context['graph'] = None
            return render(request, "descriptive/descriptive.html", context)

        data = data.replace('\r', '').strip()
        rows = [row.split('\t') for row in data.split('\n')]
        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
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
                            if row_idx == 0 and not use_first_row_as_header:
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
            context['graph'] = None
            context['results'] = None
            return render(request, "descriptive/descriptive.html", context)

        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results = []
        z_value = 1.96
        means = []
        confidence_intervals = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                mean = np.mean(valid_col)
                n_elements = len(valid_col)
                variance = np.var(valid_col, ddof=1)
                std_dev = np.std(valid_col, ddof=1)
                median = np.median(valid_col)
                std_error = std_dev / np.sqrt(n_elements)
                margin_of_error = z_value * std_error
                confidence_interval = stats.norm.interval(0.95, loc=mean, scale=std_error)
                confidence_interval_str = f"{format_str.format(confidence_interval[0])} ; {format_str.format(confidence_interval[1])}"

                minimum = np.min(valid_col)
                maximum = np.max(valid_col)
                range_value = maximum - minimum
                percentile_25 = np.percentile(valid_col, 25)
                percentile_75 = np.percentile(valid_col, 75)
                skewness = stats.skew(valid_col, bias=False)
                kurtosis = stats.kurtosis(valid_col, bias=False)
                shapiro_w, shapiro_p = stats.shapiro(valid_col)

                variable_name = headers[i] if use_first_row_as_header and i < len(headers) else f"var. {i + 1}"

                results.append({
                    'variable': variable_name,
                    'n_elements': n_elements,
                    'minimum': format_str.format(minimum),
                    'maximum': format_str.format(maximum),
                    'range': format_str.format(range_value),
                    'percentile_25': format_str.format(percentile_25),
                    'median': format_str.format(median),
                    'percentile_75': format_str.format(percentile_75),
                    'mean': format_str.format(mean),
                    'variance': format_str.format(variance),
                    'std_dev': format_str.format(std_dev),
                    'std_error': format_str.format(std_error),
                    'margin_of_error': format_str.format(margin_of_error),
                    'confidence_interval': confidence_interval_str,
                    'skewness': "{:.4g}".format(skewness),
                    'kurtosis': "{:.4g}".format(kurtosis),
                    'shapiro_w': "{:.4g}".format(shapiro_w),
                    'shapiro_p': "{:.4g}".format(shapiro_p),
                })

                means.append(mean)
                confidence_intervals.append((confidence_interval[0], confidence_interval[1]))

        context['results'] = results
        context['data'] = data
        context['headers'] = headers if use_first_row_as_header else None
        context['use_first_row_as_header'] = 'checked' if use_first_row_as_header else ''

        request.session['results'] = results
        request.session['data'] = data
        request.session['headers'] = headers if use_first_row_as_header else None
        request.session['use_first_row_as_header'] = use_first_row_as_header

        fig, ax = plt.subplots()
        bar_width = 0.5
        index = np.arange(len(means))
        ax.bar(index, means, bar_width, label='Mean', color='#9368E9', alpha=0.6, edgecolor='black', linewidth=1.0)
        for i, (ci_lower, ci_upper) in enumerate(confidence_intervals):
            ax.errorbar(index[i], means[i], yerr=[[means[i] - ci_lower], [ci_upper - means[i]]], 
                                                   fmt='o', color='black', capsize=3)
        ax.set_xlabel('Variables', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_ylabel('Values', fontsize=8, labelpad=10, fontweight='bold')
        ax.set_title('Mean Plot with 95% Confidence Intervals', fontsize=10, pad=10, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([result['variable'] for result in results], rotation=45, ha='right', fontsize=7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_data = base64.b64encode(buf.read()).decode('utf-8')
        context['graph'] = f'data:image/png;base64,{graph_data}'
        request.session['graph'] = context['graph']

##########################################################################################################################
    if request.method == "POST" and tab == "histograms":
        data = request.POST.get('data')
        use_first_row_as_header = request.POST.get('use_first_row_as_header') == 'on'
        
        if not data.strip():
            context['error'] = "Please enter data to generate histograms."
            context['graph_h'] = None
            return render(request, "descriptive/descriptive.html", context)

        data = data.replace('\r', '').strip()
        rows = [row.split('\t') for row in data.split('\n')]
        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
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
                            if row_idx == 0 and not use_first_row_as_header:
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
            context['graph_h'] = None
            return render(request, "descriptive/descriptive.html", context)
        
        context['data'] = data
        context['headers'] = headers if use_first_row_as_header else None
        context['use_first_row_as_header'] = 'checked' if use_first_row_as_header else ''

        request.session['data'] = data
        request.session['headers'] = headers if use_first_row_as_header else None
        request.session['use_first_row_as_header'] = use_first_row_as_header

        if not headers:
            headers = [f"var. {i+1}" for i in range(data_columns.shape[0])]

        num_variables = data_columns.shape[0]
        fig, axs = plt.subplots(num_variables, 4, figsize=(16, 4 * num_variables))

        if num_variables == 1:
            axs = np.array([axs])
        
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]
            
            if len(valid_col) > 0:

                variable_name = headers[i] if i < len(headers) else f"var. {i+1}"

                n1, bin_edges1, _1 = axs[i, 0].hist(valid_col, bins='auto', color='#A0C4FF', alpha=0.7, edgecolor='black')
                axs[i, 0].set_title(f'{variable_name}')
                axs[i, 0].set_xlabel('Values')
                axs[i, 0].set_ylabel('Frequency')
                axs[i, 0].set_xticks(bin_edges1)
                axs[i, 0].set_xticklabels([f'{edge:.2f}' for edge in bin_edges1], rotation=45)

                n2, bin_edges2, _2 = axs[i, 1].hist(valid_col, bins='auto', cumulative=True, color='#FFD6A5', alpha=0.7, edgecolor='black')
                axs[i, 1].set_title(f'{variable_name}')
                axs[i, 1].set_xlabel('Values')
                axs[i, 1].set_ylabel('Cumulative Frequency')
                axs[i, 1].set_xticks(bin_edges2)
                axs[i, 1].set_xticklabels([f'{edge:.2f}' for edge in bin_edges2], rotation=45)

                n3, bin_edges3, _3 = axs[i, 2].hist(valid_col, bins='auto', density=True, color='#A8E6CF', alpha=0.7, edgecolor='black')
                axs[i, 2].set_title(f'{variable_name}')
                axs[i, 2].set_xlabel('Values')
                axs[i, 2].set_ylabel('Relative Frequency')
                axs[i, 2].set_xticks(bin_edges3)
                axs[i, 2].set_xticklabels([f'{edge:.2f}' for edge in bin_edges3], rotation=45)

                counts, bin_edges = np.histogram(valid_col, bins='auto', density=True)
                cumulative_counts = np.cumsum(counts)
                cumulative_freq = cumulative_counts / cumulative_counts[-1]

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ojiva_x = np.concatenate([[bin_edges[0]], bin_centers])
                ojiva_y = np.concatenate([[0], cumulative_freq])

                n4, bin_edges4, _4 = axs[i, 3].hist(valid_col, bins=bin_edges, cumulative=True, density=True, color='#9368E9', alpha=0.7, edgecolor='black')

                axs[i, 3].plot(ojiva_x, ojiva_y, 'r--', marker='o')
                
                axs[i, 3].set_title(f'{variable_name}')
                axs[i, 3].set_xlabel('Values')
                axs[i, 3].set_ylabel('Cumulative Relative Frequency')
                axs[i, 3].set_xticks(bin_edges4)
                axs[i, 3].set_xticklabels([f'{edge:.2f}' for edge in bin_edges4], rotation=45)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        graph_h_data = base64.b64encode(buf.read()).decode('utf-8')
        context['graph_h'] = f'data:image/png;base64,{graph_h_data}'
        request.session['graph_h'] = context['graph_h']


#############################################################################################################################    
    if request.method == "POST" and tab == "boxplot":
        data = request.POST.get('data')
        use_first_row_as_header = request.POST.get('use_first_row_as_header') == 'on'

        if not data.strip():
            context['error'] = "Please enter data to generate boxplot/s."
            context['boxplot'] = None
            return render(request, "descriptive/descriptive.html", context)
        
        data = data.replace('\r', '').strip()
        rows = [row.split('\t') for row in data.split('\n')]
        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
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
                            if row_idx == 0 and not use_first_row_as_header:
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
            context['boxplot'] = None
            return render(request, "descriptive/descriptive.html", context)

        context['data'] = data
        context['headers'] = headers if use_first_row_as_header else None
        context['use_first_row_as_header'] = 'checked' if use_first_row_as_header else ''

        request.session['data'] = data
        request.session['headers'] = headers if use_first_row_as_header else None
        request.session['use_first_row_as_header'] = use_first_row_as_header

        fig, ax = plt.subplots()
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                ax.boxplot(valid_col, positions=[i + 1], widths=0.5)

        ax.set_title('Boxplot')
        ax.set_ylabel('Values')
        ax.set_xticks(range(1, len(data_columns) + 1))
        ax.set_xticklabels(headers if use_first_row_as_header else [f"var. {i + 1}" for i in range(len(data_columns))])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        boxplot_data = base64.b64encode(buf.read()).decode('utf-8')
        context['boxplot'] = f'data:image/png;base64,{boxplot_data}'
        request.session['boxplot'] = context['boxplot']


    return render(request, "descriptive/descriptive.html", context)


def count_significant_figures(num_str):
    if isinstance(num_str, (float, np.float64)):
        num_str = str(num_str)
    else:
        num_str = num_str
    
    num_str = num_str.strip()
    num_str = re.sub(r'[eE][+-]?\d+', '', num_str)
    num_str = num_str.replace('.', '')
    return len(num_str.lstrip('0'))