from fastapi import Request, HTTPException, status, Depends

from app_api.models import ApiClient


def get_api_client(request: Request) -> ApiClient:
    """
    Авторизация по заголовку Authorization: Bearer <token>
    """
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    token = auth.removeprefix("Bearer ").strip()
    try:
        client = ApiClient.objects.select_related("knowledge_base").get(token=token, is_active=True)
        return client
    except ApiClient.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API token")