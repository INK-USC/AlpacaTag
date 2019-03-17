from rest_framework import serializers

from .models import SequenceAnnotation, SequenceRecommendation
from .models import Label, Project, Document


class LabelSerializer(serializers.ModelSerializer):

    class Meta:
        model = Label
        fields = ('id', 'text', 'shortcut', 'background_color', 'text_color')


class DocumentSerializer(serializers.ModelSerializer):

    class Meta:
        model = Document
        fields = ('id', 'text')


class ProjectSerializer(serializers.ModelSerializer):

    class Meta:
        model = Project
        fields = ('id', 'name', 'description', 'guideline', 'users', 'image', 'updated_at')


class ProjectFilteredPrimaryKeyRelatedField(serializers.PrimaryKeyRelatedField):

    def get_queryset(self):
        view = self.context.get('view', None)
        request = self.context.get('request', None)
        queryset = super(ProjectFilteredPrimaryKeyRelatedField, self).get_queryset()
        if not request or not queryset or not view:
            return None
        return queryset.filter(project=view.kwargs['project_id'])


class SequenceAnnotationSerializer(serializers.ModelSerializer):
    label = ProjectFilteredPrimaryKeyRelatedField(queryset=Label.objects.all())

    class Meta:
        model = SequenceAnnotation
        fields = ('id', 'prob', 'label', 'start_offset', 'end_offset')

    def create(self, validated_data):
        annotation = SequenceAnnotation.objects.create(**validated_data)
        return annotation


class SequenceRecommendationSerializer(serializers.ModelSerializer):
    def __init__(self, *args, **kwargs):
        many = kwargs.pop('many', True)
        super(SequenceRecommendationSerializer, self).__init__(many=many, *args, **kwargs)

    class Meta:
        model = SequenceRecommendation
        fields = ('id', 'prob', 'label', 'start_offset', 'end_offset', 'document')

    def create(self, validated_data):
        # recommends = [SequenceRecommendation(**item) for item in validated_data]
        # for recommend in recommends:
        #     SequenceAnnotation.objects.create(recommend)
        recommendation = SequenceRecommendation.objects.create(**validated_data)
        return recommendation


class SequenceDocumentSerializer(serializers.ModelSerializer):
    annotations = serializers.SerializerMethodField()
    recommendations = serializers.SerializerMethodField()

    def get_annotations(self, instance):
        request = self.context.get('request')
        if request:
            annotations = instance.seq_annotations.filter(user=request.user)
            serializer = SequenceAnnotationSerializer(annotations, many=True)
            return serializer.data

    def get_recommendations(self, instance):
        request = self.context.get('request')
        if request:
            recommendations = instance.seq_recommendations
            serializer = SequenceRecommendationSerializer(recommendations, many=True)
            return serializer.data

    class Meta:
        model = Document
        fields = ('id', 'text', 'annotations', 'recommendations')

