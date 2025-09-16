from htmltools import Tag, TagList, tags

def color_input(id: str, label: str, value: str = "#1E90FF") -> Tag:
    # This helper only returns markup; the JS binding is loaded once in app.py head.
    el = tags.input({
        "type": "color",
        "class": "py-color-input",   # matched by the JS binding
        "id": id,
        "data-input-id": id,
        "value": value,
        "style": "width: 90%; height: 30px; padding: 4px; margin-top: 5px; margin-left: 10%"
    })
    lab = tags.label({"for": id, "class": "form-label", "style": "margin-bottom: 0.25rem;"}, label) if label else None
    return TagList(*(item for item in (lab, el) if item is not None))