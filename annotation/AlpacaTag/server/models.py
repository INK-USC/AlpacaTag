import json
from django.core.exceptions import ValidationError
from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.staticfiles.storage import staticfiles_storage
from .utils import get_key_choices
import spacy
nlp = spacy.load('en_core_web_sm')


class Project(models.Model):
    SEQUENCE_LABELING = 'SequenceLabeling'

    name = models.CharField(max_length=100)
    description = models.CharField(max_length=500)
    guideline = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    users = models.ManyToManyField(User, related_name='projects')

    def get_absolute_url(self):
        return reverse('upload', args=[self.id])

    def get_progress(self, user):
        docs = self.get_documents(is_null=True, user=user)
        total = self.documents.count()
        remaining = docs.count()
        return {'total': total, 'remaining': remaining}

    @property
    def image(self):
        url = staticfiles_storage.url('images/cat-3449999_640.jpg')
        return url

    def get_template_name(self):
        template_name = 'annotation/sequence_labeling.html'
        return template_name

    def get_documents(self, is_null=True, user=None):
        docs = self.documents.all()
        docs = docs.filter(annotated=False)
        return docs

    def get_index_documents(self, indices):
        docs = self.documents.all()
        docs = docs.filter(pk__in=indices)
        return docs

    def get_document_serializer(self):
        from .serializers import SequenceDocumentSerializer
        return SequenceDocumentSerializer

    def get_annotation_serializer(self):
        from .serializers import SequenceAnnotationSerializer
        return SequenceAnnotationSerializer

    def get_annotation_class(self):
        return SequenceAnnotation

    def __str__(self):
        return self.name


class Label(models.Model):
    KEY_CHOICES = get_key_choices()
    COLOR_CHOICES = ()

    text = models.CharField(max_length=100)
    shortcut = models.CharField(max_length=15, blank=True, null=True, choices=KEY_CHOICES)
    project = models.ForeignKey(Project, related_name='labels', on_delete=models.CASCADE)
    background_color = models.CharField(max_length=7, default='#209cee')
    text_color = models.CharField(max_length=7, default='#ffffff')

    def __str__(self):
        return self.text

    class Meta:
        unique_together = (
            ('project', 'text'),
            ('project', 'shortcut')
        )


class Document(models.Model):
    text = models.TextField()
    project = models.ForeignKey(Project, related_name='documents', on_delete=models.CASCADE)
    annotated = models.BooleanField(default=False)
    metadata = models.TextField(default='{}')

    def delete_annotations(self):
        self.seq_annotations.all().delete()

    def get_annotations(self):
        return self.seq_annotations.all()

    def to_csv(self):
        return self.make_dataset()

    def make_dataset(self):
        return self.make_dataset_for_sequence_labeling()

    def make_dataset_for_sequence_labeling(self):
        annotations = self.get_annotations()
        doc = nlp(self.text)
        words = [token.text for token in doc]
        dataset = [[self.id, word, 'O', self.metadata] for word in words]
        startoff_map = {}
        endoff_map = {}

        start_off = 0
        for word_index, word in enumerate(dataset):
            end_off = start_off + len(word[1])
            startoff_map[start_off] = word_index
            endoff_map[end_off] = word_index
            start_off = end_off + 1

        for a in annotations:
            if a.start_offset in startoff_map:
                dataset[startoff_map[a.start_offset]][2] = 'B-{}'.format(a.label.text)
            if a.end_offset in endoff_map:
                if endoff_map[a.end_offset] != startoff_map[a.start_offset]:
                    dataset[endoff_map[a.end_offset]][2] = 'I-{}'.format(a.label.text)
        return dataset

    def to_json(self):
        return self.make_dataset_json()

    def make_dataset_json(self):
        return self.make_dataset_for_sequence_labeling_json()

    def make_dataset_for_sequence_labeling_json(self):
        annotations = self.get_annotations()
        entities = [(a.start_offset, a.end_offset, a.label.text) for a in annotations]
        username = annotations[0].user.username
        dataset = {'doc_id': self.id, 'text': self.text, 'entities': entities, 'username': username, 'metadata': json.loads(self.metadata)}
        return dataset

    def __str__(self):
        return self.text[:50]


class Annotation(models.Model):
    prob = models.FloatField(default=0.0)
    manual = models.BooleanField(default=False)

    class Meta:
        abstract = True


class SequenceAnnotation(Annotation):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, related_name='seq_annotations', on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    start_offset = models.IntegerField()
    end_offset = models.IntegerField()

    def clean(self):
        if self.start_offset >= self.end_offset:
            raise ValidationError('start_offset is after end_offset')

    class Meta:
        unique_together = ('document', 'user', 'label', 'start_offset', 'end_offset')


class RecommendationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name='history', on_delete=models.CASCADE)
    word = models.TextField()
    label = models.ForeignKey(Label, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'project', 'word', 'label')


class Setting(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name='settings', on_delete=models.CASCADE)
    embedding = models.IntegerField() #1-glove 2-w2v 3-fasttext 4-bert 5-elmo 6-gpt
    nounchunk = models.BooleanField()
    onlinelearning = models.BooleanField()
    history = models.BooleanField()
    active = models.IntegerField()
    batch = models.IntegerField()
    epoch = models.IntegerField()
    acquire = models.IntegerField()

    def get_setting_serializer(self):
        from .serializers import SettingSerializer
        return SettingSerializer

    class Meta:
        unique_together = ('user', 'project')
