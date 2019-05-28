from django.contrib import admin

from .models import Label, Document, Project
from .models import SequenceAnnotation

admin.site.register(SequenceAnnotation)
admin.site.register(Label)
admin.site.register(Document)
admin.site.register(Project)
