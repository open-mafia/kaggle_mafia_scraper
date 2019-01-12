

from ast import literal_eval


def quote_str_to_list(txt):
    try:
        return literal_eval(txt)
    except Exception:
        return ['ERROR PARSING QUOTES']

