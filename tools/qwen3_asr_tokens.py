"""Validate actual chat token work without accepting booleans or conflicting token layouts."""


def token_ids(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    ids = choices[0].get("token_ids")
    extension = response.get("tensorfold")
    native = extension.get("token_ids") if isinstance(extension, dict) else None
    for layout in (ids, native):
        if layout is not None and (not isinstance(layout, list)
                                   or any(type(token) is not int or token < 0 for token in layout)):
            return None
    if ids is not None and native is not None and ids != native:
        return None
    return native if ids is None else ids


def valid_token_work(ids, count):
    return (type(count) is int and count > 0 and isinstance(ids, list) and len(ids) == count
            and all(type(token) is int and token >= 0 for token in ids))
