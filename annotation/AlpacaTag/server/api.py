from collections import Counter
from itertools import chain

from django.db.utils import IntegrityError
from django.shortcuts import get_object_or_404, get_list_or_404
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, generics, filters, mixins
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .models import Project, Label, Document, Setting, RecommendationHistory
from .permissions import IsAdminUserAndWriteOnly, IsProjectUser, IsOwnAnnotation
from .serializers import ProjectSerializer, LabelSerializer, DocumentSerializer, SettingSerializer, RecommendationHistorySerializer

import spacy
from spacy.tokens import Doc

import time
from alpaca_serving_client.client import AlpacaClient
alpaca_client = None

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def alpaca_recommend(text):
    global alpaca_client
    response = alpaca_client.predict(text)
    if response == 'error':
        print('error')
        time.sleep(2)
        return alpaca_recommend(text)
    return response


def alpaca_initiate(project_id):
    global alpaca_client
    response = alpaca_client.initiate(project_id)
    if response == 'error':
        print('error')
        time.sleep(2)
        return alpaca_client.initiate(project_id)
    return response


def alpaca_online_initiate(train_docs, predefined_label):
    global alpaca_client
    response = alpaca_client.online_initiate(train_docs, [predefined_label])
    if response == 'error':
        print('error')
        time.sleep(2)
        alpaca_client.online_initiate(train_docs, [predefined_label])


def alpaca_online_learning(train_docs, annotations, epoch, batch):
    global alpaca_client
    response = alpaca_client.online_learning(train_docs, annotations, epoch, batch)
    if response == 'error':
        print('error')
        time.sleep(2)
        alpaca_client.online_learning(train_docs, annotations, epoch, batch)


def alpaca_active_learning(train_docs, acquire):
    global alpaca_client
    response = alpaca_client.active_learning(train_docs, acquire)
    if response == 'error':
        print('error')
        time.sleep(2)
        return alpaca_client.active_learning(train_docs, acquire)
    return response


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsAdminUserAndWriteOnly)

    def get_queryset(self):
        return self.request.user.projects

    @action(methods=['get'], detail=True)
    def progress(self, request, pk=None):
        project = self.get_object()
        return Response(project.get_progress())


class LabelList(generics.ListCreateAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])
        return queryset

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project)


class ProjectStatsAPI(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        labels = [label.text for label in p.labels.all()]
        users = [user.username for user in p.users.all()]
        docs = [doc for doc in p.documents.all()]
        nested_labels = [[a.label.text for a in doc.get_annotations()] for doc in docs]
        nested_users = [[a.user.username for a in doc.get_annotations()] for doc in docs]

        label_count = Counter(chain(*nested_labels))
        label_data = [label_count[name] for name in labels]

        user_count = Counter(chain(*nested_users))
        user_data = [user_count[name] for name in users]

        response = {'label': {'labels': labels, 'data': label_data},
                    'user': {'users': users, 'data': user_data}}

        return Response(response)


class LabelDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUser)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])

        return queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['label_id'])
        self.check_object_permissions(self.request, obj)

        return obj


class DocumentList(generics.ListCreateAPIView):
    queryset = Document.objects.all()
    filter_backends = (DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter)
    search_fields = ('text', )
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.serializer_class = project.get_document_serializer()

        return self.serializer_class

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])

        if not self.request.query_params.get('active_indices'):
            return queryset
        active_indices = self.request.query_params.get('active_indices')
        active_indices = list(map(int, active_indices.split(",")))

        queryset = project.get_index_documents(active_indices)

        if not self.request.query_params.get('is_checked'):
            return queryset

        is_null = self.request.query_params.get('is_checked') == 'true'
        queryset = project.get_documents(is_null).distinct()
        return queryset


class DocumentDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUser)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])
        return queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['doc_id'])
        self.check_object_permissions(self.request, obj)
        return obj

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)


class AnnotationList(generics.ListCreateAPIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.serializer_class = project.get_annotation_serializer()

        return self.serializer_class

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        document = project.documents.get(id=self.kwargs['doc_id'])
        self.queryset = document.get_annotations()
        self.queryset = self.queryset.filter(user=self.request.user)

        return self.queryset

    def perform_create(self, serializer):
        doc = get_object_or_404(Document, pk=self.kwargs['doc_id'])
        serializer.save(document=doc, user=self.request.user)

    def delete(self, request, *args, **kwargs):
        doc = get_object_or_404(Document, pk=self.kwargs['doc_id'])
        doc.delete_annotations()
        return Response(status=status.HTTP_204_NO_CONTENT)


class AnnotationDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated, IsProjectUser, IsOwnAnnotation)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.serializer_class = project.get_annotation_serializer()

        return self.serializer_class

    def get_queryset(self):
        document = get_object_or_404(Document, pk=self.kwargs['doc_id'])
        self.queryset = document.get_annotations()

        return self.queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['annotation_id'])
        self.check_object_permissions(self.request, obj)

        return obj


class SettingList(generics.GenericAPIView, mixins.CreateModelMixin, mixins.UpdateModelMixin):
    queryset = Setting.objects.all()
    serializer_class = SettingSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.queryset = self.queryset.filter(project=project, user=self.request.user)

        return self.queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset)
        self.check_object_permissions(self.request, obj)
        return obj

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project, user=self.request.user)

    def get(self, request, *args, **kwargs):
        return Response(self.serializer_class(self.get_object()).data)

    def put(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        _, created = Setting.objects.get_or_create(project=project, user=self.request.user,
                                                   defaults=self.request.data)
        if not created:
            return self.update(request, *args, **kwargs)
        return Response(created)


class RecommendationHistoryList(generics.ListCreateAPIView):
    queryset = RecommendationHistory.objects.all()
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)
    serializer_class = RecommendationHistorySerializer

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.queryset = self.queryset.filter(project=project, user=self.request.user)
        return self.queryset

    def perform_create(self, serializer):
        try:
            project = get_object_or_404(Project, pk=self.kwargs['project_id'])
            serializer.save(project=project, user=self.request.user)
        except IntegrityError:
            print("The word with that label is already exist in history.")


class RecommendationHistoryDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = RecommendationHistory.objects.all()
    permission_classes = (IsAuthenticated, IsProjectUser)
    serializer_class = RecommendationHistorySerializer

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        self.queryset = self.queryset.filter(project=project, user=self.request.user)
        return self.queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['history_id'])
        self.check_object_permissions(self.request, obj)

        return obj


class ConnectToServer(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser,)

    def get(self, request, *args, **kwargs):
        global alpaca_client
        try:
            alpaca_client = AlpacaClient()
            response = {'connection': True}
        except TimeoutError:
            response = {'connection': False}

        return Response(response)


class RecommendationList(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def index_word2char(self, entities, words):
        res = []
        for i in range(len(entities)):
            startOffset = 0
            for j in range(entities[i]['beginOffset']):
                startOffset = startOffset + len(words[j])
                startOffset = startOffset + 1

            endOffset = startOffset

            for j in range(entities[i]['beginOffset'], entities[i]['endOffset']):
                endOffset = endOffset + len(words[j])
                endOffset = endOffset + 1

            endOffset = endOffset - 1

            recommend = {'document': self.kwargs['doc_id'], 'label': entities[i]['type'],
                         'start_offset': startOffset, 'end_offset': endOffset}
            res.append(recommend)
        return res

    def chunking(self, text):
        doc = nlp(text)
        words = [token.text for token in doc]
        chunklist = []
        for chunk in doc.noun_chunks:
            chunkdict = {}
            chunkdict['nounchunk'] = chunk.text
            chunkdict['left'] = chunk.start
            chunkdict['right'] = chunk.end
            chunklist.append(chunkdict)

        res = {
            'words': words,
            'entities': [

            ]
        }

        for chunkdict in chunklist:
            entity = {
                'text': chunkdict['nounchunk'],
                'type': None,
                'score': None,
                'beginOffset': chunkdict['left'],
                'endOffset': chunkdict['right']
            }
            res['entities'].append(entity)

        return res

    def get(self, request, *args, **kwargs):
        #project, document
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        document = project.documents.get(id=self.kwargs['doc_id'])

        #settings
        setting_queryset = Setting.objects.all()
        serializer_class = SettingSerializer
        setting_queryset = setting_queryset.filter(project=project, user=self.request.user)
        setting_obj = get_object_or_404(setting_queryset)
        setting_data = serializer_class(setting_obj).data
        opt_n = setting_data['nounchunk']
        opt_o = setting_data['onlinelearning']
        opt_h = setting_data['history']
        final_list = []
        n_list = []
        o_list = []
        h_list = []

        if opt_n:
            response = self.chunking(document.text)
            n_entities = response['entities']
            n_words = response['words']
            n_list = self.index_word2char(n_entities, n_words)

        if opt_o and alpaca_client is not None:
            response = alpaca_recommend(document.text)
            o_entities = response['entities']
            o_words = response['words']
            o_list = self.index_word2char(o_entities, o_words)


        if opt_h:
            history_queryset = RecommendationHistory.objects.all()
            serializer_class = RecommendationHistorySerializer
            history_queryset = history_queryset.filter(project=project, user=self.request.user)
            if len(history_queryset) > 0:
                history_obj = get_list_or_404(history_queryset)
                h_list = serializer_class(history_obj, many=True).data

        if opt_n:
            for n in n_list:
                is_h = False
                is_o = False
                tmp_h_list = []
                tmp_o_list = []
                tmp_list = []

                # noun chunk covers results from model
                if opt_o and alpaca_client is not None:
                    for o in o_list:
                        if n['start_offset'] <= o['start_offset'] and n['end_offset'] >= o['end_offset']:
                            tmp_o_list.append(o)
                            is_o = True

                # noun chunk covers results from user annotation
                if opt_h:
                    for h in h_list:
                        if h['word'].lower() in document.text[n['start_offset']:n['end_offset']].lower():
                            start_offset = n['start_offset'] + document.text[n['start_offset']:n['end_offset']].lower().find(h['word'].lower())
                            end_offset = start_offset + len(h['word'])
                            if document.text[end_offset] != ' ' or (document.text[start_offset-1] != ' ' and start_offset != 0):
                                continue
                            else:
                                label_queryset = Label.objects.all()
                                serializer_class = LabelSerializer
                                label_queryset = label_queryset.filter(project=self.kwargs['project_id'])
                                label_obj = get_object_or_404(label_queryset, pk=h['label'])
                                label_data = serializer_class(label_obj).data
                                h_dict = {'document': self.kwargs['doc_id'], 'label': label_data['text'], 'start_offset': start_offset, 'end_offset': end_offset}
                                for tmp_h in tmp_h_list:
                                    if h_dict['start_offset'] <= tmp_h['start_offset'] and h_dict['end_offset'] >= tmp_h['end_offset']:
                                        tmp_h_list.remove(tmp_h)
                                        continue
                                tmp_h_list.append(h_dict)
                                is_h = True

                if len(tmp_h_list) > 0 and len(tmp_o_list) > 0:
                    for tmp_h in tmp_h_list:
                        for tmp_o in tmp_o_list[:]:
                            o_range = range(tmp_o['start_offset'], tmp_o['end_offset'])
                            h_range = range(tmp_h['start_offset'], tmp_h['end_offset'])
                            h_range_s = set(h_range)
                            if len(h_range_s.intersection(o_range)) > 0:
                                tmp_o_list.remove(tmp_o)
                    tmp_list.extend(tmp_h_list)
                    tmp_list.extend(tmp_o_list)

                if len(tmp_h_list) > 0 and len(tmp_o_list) == 0:
                    tmp_list = tmp_h_list

                if len(tmp_h_list) == 0 and len(tmp_o_list) > 0:
                    tmp_list = tmp_o_list

                final_list.extend(tmp_list)
                if not is_o and not is_h:
                    final_list.append(n)
        else:
            tmp_h_list = []
            tmp_o_list = []
            tmp_list = []
            if opt_o and alpaca_client is not None:
                for o in o_list:
                    tmp_o_list.append(o)

            if opt_h:
                for h in h_list:
                    if h['word'].lower() in document.text.lower():
                        label_queryset = Label.objects.all()
                        serializer_class = LabelSerializer
                        label_queryset = label_queryset.filter(project=self.kwargs['project_id'])
                        label_obj = get_object_or_404(label_queryset, pk=h['label'])
                        label_data = serializer_class(label_obj).data
                        start_offset = document.text.lower().find(h['word'].lower())
                        end_offset = start_offset + len(h['word'])
                        h_dict = {'document': self.kwargs['doc_id'], 'label': label_data['text'],
                                  'start_offset': start_offset, 'end_offset': end_offset}
                        tmp_h_list.append(h_dict)

            if len(tmp_h_list) > 0 and len(tmp_o_list) > 0:
                for tmp_h in tmp_h_list:
                    for tmp_o in tmp_o_list[:]:
                        o_range = range(tmp_o['start_offset'], tmp_o['end_offset'])
                        h_range = range(tmp_h['start_offset'], tmp_h['end_offset'])
                        h_range_s = set(h_range)
                        if len(h_range_s.intersection(o_range)) > 0:
                            tmp_o_list.remove(tmp_o)
                tmp_list.extend(tmp_h_list)
                tmp_list.extend(tmp_o_list)

            if len(tmp_h_list) > 0 and len(tmp_o_list) == 0:
                tmp_list = tmp_h_list
            if len(tmp_h_list) == 0 and len(tmp_o_list) > 0:
                tmp_list = tmp_o_list

            final_list.extend(tmp_list)
        return Response({"recommendation": final_list})


class LearningInitiate(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser,)

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        labels = [label.text for label in p.labels.all()]

        if alpaca_initiate(self.kwargs['project_id']) == "Model Loaded":
            response = {'initiated': False, 'loaded': True}

        else:
            predefined_label = []
            for i in labels:
                predefined_label.append('B-' + str(i))
                predefined_label.append('I-' + str(i))
            predefined_label.append('O')
            docs = [doc for doc in p.documents.all()]
            train_docs = [str.split(doc.text) for doc in docs]

            alpaca_online_initiate(train_docs, predefined_label)
            response = {'initiated': True, 'loaded': False}

        return Response(response)


class OnlineLearning(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def post(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        setting_queryset = Setting.objects.all()
        serializer_class = SettingSerializer
        setting_queryset = setting_queryset.filter(project=p, user=self.request.user)
        setting_obj = get_object_or_404(setting_queryset)
        setting_data = serializer_class(setting_obj).data

        docs_num = request.data.get('indices')
        docs = [doc for doc in p.documents.filter(pk__in=docs_num)]
        annotations = [[label[2] for label in doc.make_dataset_for_sequence_labeling(self.request.user.id)] for doc in docs]
        train_docs = [str.split(doc.text) for doc in docs]

        alpaca_online_learning(train_docs, annotations, setting_data['epoch'], setting_data['batch'])
        response = {'docs': train_docs,
                    'annotations': annotations}

        return Response(response)


class ActiveLearning(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser,)

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        docs = [doc for doc in p.documents.all()]
        train_docs = [str.split(doc.text) for doc in docs]

        setting_queryset = Setting.objects.all()
        serializer_class = SettingSerializer
        setting_queryset = setting_queryset.filter(project=p, user=self.request.user)
        setting_obj = get_object_or_404(setting_queryset)
        setting_data = serializer_class(setting_obj).data

        active_data = alpaca_active_learning(train_docs, setting_data['acquire'])

        response = {'indices': active_data['indices'], 'scores': active_data['scores']}

        return Response(response)
