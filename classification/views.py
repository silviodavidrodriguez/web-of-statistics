from django.shortcuts import render

def classification(request):
    return render(request, "classification/classification.html")