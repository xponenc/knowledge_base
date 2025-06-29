class HierarchicalContextMixin:
    """
    Mixin для подмешивания полной иерархии объектов в context_data:
    - content
    - document (URL / LocalDocument / NetworkDocument)
    - storage (WebSite / LocalStorage / CloudStorage)
    - kb (KnowledgeBase)

    Добавляет в context:
    - объекты по ключам: 'content', 'document', 'storage', 'kb'
    - названия типа объекта на русском: например 'storage_type_ru' = 'Облачный диск'
    - упорядоченный список 'breadcrumbs_hierarchy' для удобного построения breadcrumbs

    Работает для DetailView/UpdateView и кастомных View с self.object.
    """

    # Соответствие model_name -> ключ в контексте
    CONTEXT_ALIASES = {
        'urlcontent': 'content',
        'rawcontent': 'content',
        'cleanedcontent': 'content',

        'url': 'document',
        'localdocument': 'document',
        'networkdocument': 'document',

        'website': 'storage',
        'localstorage': 'storage',
        'cloudstorage': 'storage',

        'knowledgebase': 'kb',
    }

    # Русские типы для удобства отображения
    TYPE_NAMES_RU = {
        'urlcontent': 'Контент веб-страницы',
        'rawcontent': 'Исходный контент',
        'cleanedcontent': 'Очищенный контент',

        'url': 'Веб-страница',
        'localdocument': 'Локальный документ',
        'networkdocument': 'Сетевой документ',

        'website': 'Веб-сайт',
        'localstorage': 'Локальное хранилище',
        'cloudstorage': 'Облачный диск',

        'knowledgebase': 'База знаний',
    }

    # Порядок следования для breadcrumbs
    BREADCRUMB_ORDER = ['kb', 'storage', 'document', 'content']

    def get_context_data(self, **kwargs):
        """
        Подмешивает иерархические объекты и дополнительную информацию в context.
        """
        context = super().get_context_data(**kwargs)

        obj = getattr(self, 'object', None)
        if obj is None:
            return context

        visited_models = set()
        allowed_models = set(self.CONTEXT_ALIASES.keys())
        collected = {}

        while obj:
            model_name = obj._meta.model_name
            alias = self.CONTEXT_ALIASES.get(model_name, model_name)

            # Добавляем объект в контекст, если ещё нет
            if alias not in collected:
                collected[alias] = obj
                visited_models.add(model_name)

                # Добавляем русский тип объекта
                type_ru_key = f"{alias}_type_ru"
                type_ru_value = self.TYPE_NAMES_RU.get(model_name)
                if type_ru_value:
                    context[type_ru_key] = type_ru_value

            # Ищем следующий объект вверх по FK из разрешённых моделей, пропуская уже посещённые
            next_obj = None
            for field in obj._meta.fields:
                if field.is_relation and field.related_model:
                    related_model_name = field.related_model._meta.model_name
                    if related_model_name in allowed_models and related_model_name not in visited_models:
                        related = getattr(obj, field.name, None)
                        if related:
                            next_obj = related
                            break
            obj = next_obj

        # Формируем упорядоченный список для breadcrumbs
        breadcrumbs = []
        for key in self.BREADCRUMB_ORDER:
            if key in collected:
                breadcrumbs.append((key, collected[key]))
        context['breadcrumbs_hierarchy'] = breadcrumbs

        # Добавляем сами объекты по ключам в контекст
        context.update(collected)

        return context
