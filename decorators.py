from django.middleware.cache import CacheMiddleware
from django.utils.cache import get_cache_key
from django.utils.decorators import decorator_from_middleware_with_args


class PostDeleterCacheMiddleware(CacheMiddleware):
    def process_request(self, request):
        if request.method == 'POST':
            # https://github.com/django/django/blob/3.0.9/django/middleware/cache.py#L137
            cache_key = get_cache_key(request, self.key_prefix, 'GET', cache=self.cache)
            self.cache.delete(cache_key)
        return super().process_request(request)


def post_deleter_cache_page(timeout, *, cache=None, key_prefix=None):
    return decorator_from_middleware_with_args(PostDeleterCacheMiddleware)(
        page_timeout=timeout, cache_alias=cache, key_prefix=key_prefix,
    )