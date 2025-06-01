from rest_framework import serializers
from .models import URL, CloudStorage, Document

class URLSerializer(serializers.ModelSerializer):
    """Сериализатор для модели URL."""
    class Meta:
        model = URL
        fields = ['id', 'url', 'parser', 'response_status', 'last_updated', 'last_parsed', 'status',
                  'raw_content', 'cleaned_content', 'language', 'categories', 'error_message',
                  'created_at', 'updated_at']

    def validate_categories(self, value):
        """Валидация формата категорий."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Categories must be a list.")
        for category in value:
            if not isinstance(category, dict) or 'name' not in category:
                raise serializers.ValidationError("Each category must be a dict with 'name'.")
        return value

class CloudStorageSerializer(serializers.ModelSerializer):
    """Сериализатор для модели CloudStorage."""
    class Meta:
        model = CloudStorage
        fields = ['id', 'name', 'api_type', 'credentials', 'created_at', 'updated_at']

class DocumentSerializer(serializers.ModelSerializer):
    """Сериализатор для модели Document."""
    class Meta:
        model = Document
        fields = ['id', 'cloud_storage', 'file_id', 'file_name', 'file_path', 'file_type',
                  'description', 'language', 'categories', 'last_synced', 'last_modified',
                  'status', 'raw_content', 'cleaned_content', 'error_message', 'created_at',
                  'updated_at']

    def validate_categories(self, value):
        """Валидация формата категорий."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Categories must be a list.")
        for category in value:
            if not isinstance(category, dict) or 'name' not in category:
                raise serializers.ValidationError("Each category must be a dict with 'name'.")
        return value