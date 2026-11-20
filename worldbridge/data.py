"""Frame-aligned NumPy training pairs, independent of preview-video clocks."""
from pathlib import Path
import numpy as np
from .common import read_json


def iter_pairs(episode,view='main',stride=1,condition='control',target_size=None):
    import cv2
    episode=Path(episode);meta=read_json(episode/'episode.json')
    out=episode/'proxy'/view;report=read_json(out/'render_report.json')
    source=cv2.VideoCapture(meta['views'][view]['source_video'])
    if condition not in ['white','semantic','actors','control']:raise ValueError(condition)
    proxy=cv2.VideoCapture(str(out/f'{condition}.mp4'))
    try:
        for descriptor in report['chunks']:
            with np.load(out/descriptor['path'],allow_pickle=False) as z:
                for j in range(descriptor['frames']):
                    idx=descriptor['start_frame']+j
                    ok1,bgr=source.read();ok2,pbgr=proxy.read()
                    if not(ok1 and ok2):raise ValueError(f'Training-pair decode failed at frame {idx}')
                    if idx%stride:continue
                    w,h=target_size or report['resolution']
                    target=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(w,h))
                    yield dict(frame=idx,time_us=int(z['time_us'][j]),rgb=target,
                               proxy=cv2.cvtColor(pbgr,cv2.COLOR_BGR2RGB),depth=z['depth'][j].copy(),
                               instance=z['instance'][j].copy(),semantic=z['semantic'][j].copy(),
                               covered=z['covered'][j].copy(),actor_depth=z['actor_depth'][j].copy(),
                               actor_instance=z['actor_instance'][j].copy(),actor_semantic=z['actor_semantic'][j].copy(),
                               condition_type=condition,depth_scope='full_geometric_proxy_not_native_depth')
    finally:source.release();proxy.release()
