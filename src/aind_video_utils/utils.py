import datetime


def get_millisecond_string(seconds):
    ms = datetime.timedelta(seconds=seconds) / datetime.timedelta(
        milliseconds=1
    )
    return f"{ms:f}".rstrip("0").rstrip(".") + "ms"
