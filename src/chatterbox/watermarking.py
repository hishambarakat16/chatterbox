import perth


class PassthroughWatermarker:
    def apply_watermark(self, wav, sample_rate=None, watermark=None):
        return wav

    def get_watermark(self, wav, sample_rate=None):
        return 0.0


def create_watermarker():
    implicit_cls = getattr(perth, "PerthImplicitWatermarker", None)
    if callable(implicit_cls):
        try:
            return implicit_cls()
        except Exception:
            pass

    dummy_cls = getattr(perth, "DummyWatermarker", None)
    if callable(dummy_cls):
        try:
            return dummy_cls()
        except Exception:
            pass

    return PassthroughWatermarker()
