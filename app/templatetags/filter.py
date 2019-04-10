"""
template filters for raw json claim data
https://docs.djangoproject.com/en/dev/howto/custom-template-tags/#writing-custom-template-filters
"""

from django import template
import math

register = template.Library()


@register.filter
def dict_get_item(dict, key):
    return dict[key]

@register.filter
def abs_val(val):
    return math.fabs(val)

@register.filter
def blank_abs(val):
    return val + 1