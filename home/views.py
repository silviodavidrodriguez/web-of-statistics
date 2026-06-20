from django.shortcuts import render

def index(request):
    context = {'segment': 'home'}
    return render(request, "home/index.html", context)