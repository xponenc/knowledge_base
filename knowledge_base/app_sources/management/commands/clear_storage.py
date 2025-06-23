from django.core.management.base import BaseCommand

from app_sources.content_models import CleanedContent, RawContent
from app_sources.report_models import CloudStorageUpdateReport
from app_sources.source_models import NetworkDocument


class Command(BaseCommand):
    help = ("Удаляет все связанные объекты хранения: CleanedContent, RawContent, "
            "NetworkDocument и CloudStorageUpdateReport")

    def handle(self, *args, **kwargs):
        confirm = input("ВНИМАНИЕ: Это удалит ВСЕ данные хранилища. Продолжить? (y/n): ")
        if confirm.lower() != "y":
            self.stdout.write(self.style.ERROR("Операция отменена."))
            return
        self.stdout.write(self.style.WARNING("Очистка хранилища началась..."))

        cleaned_count = CleanedContent.objects.count()
        CleanedContent.objects.all().delete()
        self.stdout.write(f"Удалено очищенного контента: {cleaned_count}")

        raw_count = RawContent.objects.count()
        RawContent.objects.all().delete()
        self.stdout.write(f"Удалено raw-контента: {raw_count}")

        doc_count = NetworkDocument.objects.count()
        NetworkDocument.objects.all().delete()
        self.stdout.write(f"Удалено сетевых документов: {doc_count}")

        report_count = CloudStorageUpdateReport.objects.count()
        CloudStorageUpdateReport.objects.all().delete()
        self.stdout.write(f"Удалено отчётов CloudStorage: {report_count}")

        self.stdout.write(self.style.SUCCESS("Очистка завершена успешно."))