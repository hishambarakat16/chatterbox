from importlib import import_module

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # For Python <3.8

try:
    __version__ = version("chatterbox-tts")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


_LAZY_EXPORTS = {
    "ChatterboxTTS": (".tts", "ChatterboxTTS"),
    "ChatterboxVC": (".vc", "ChatterboxVC"),
    "ChatterboxMultilingualTTS": (".mtl_tts", "ChatterboxMultilingualTTS"),
    "SUPPORTED_LANGUAGES": (".mtl_tts", "SUPPORTED_LANGUAGES"),
    "ChatterboxMultilingualStreamingTTS": (".mtl_tts_streaming", "ChatterboxMultilingualStreamingTTS"),
    "ChatterboxMultilingualConcurrentTTS": (".mtl_tts_concurrent", "ChatterboxMultilingualConcurrentTTS"),
    "ChatterboxMultilingualScheduledTTS": (".mtl_tts_scheduled", "ChatterboxMultilingualScheduledTTS"),
    "ChatterboxMultilingualScheduledTurboS3TTS": (".mtl_tts_scheduled_turbo_s3", "ChatterboxMultilingualScheduledTurboS3TTS"),
    "ChatterboxMultilingualVllmTurboS3TTS": (".mtl_tts_vllm_turbo_s3", "ChatterboxMultilingualVllmTurboS3TTS"),
}


__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
