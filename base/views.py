from django.conf import settings
from django.views.static import serve

def show_static(request):
    print('doc_classifier/static/')
    path = request.GET.get('img_name', None)
    # path = request.data.copy()['img_name']
    print('path',path)
    return serve(**{'request':request, 'document_root':settings.MEDIA_ROOT, 'path':path})