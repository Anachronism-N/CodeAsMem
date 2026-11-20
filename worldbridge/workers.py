"""Importable worker entry points required by Windows spawn / python -m."""
import atexit
from .pipeline import convert_case

_GPU=None


def convert_job(args):
    return convert_case(*args)


def initialize_renderer(width,height):
    global _GPU
    from .wgl import GLRenderer
    _GPU=GLRenderer(width,height)
    atexit.register(_GPU.close)


def render_job(args):
    from .render import render_episode
    episode,view,width,height=args
    return render_episode(episode,view,width,height,renderer=_GPU)


def track_job(episode):
    from .tracks import export_tracks
    return export_tracks(episode)
