def request_data(request):
    return {
        'client_ip': request.META.get('REMOTE_ADDR', 'Unknown'),
        'user_agent': request.META.get('HTTP_USER_AGENT', 'Unknown'),
    }