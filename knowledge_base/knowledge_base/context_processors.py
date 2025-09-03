def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()  # Первый IP в списке — клиентский
    else:
        ip = request.META.get('REMOTE_ADDR', 'Unknown')
    return ip

def request_data(request):
    return {
        'client_ip': get_client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', 'Unknown'),
    }