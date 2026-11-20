"""Read-only local dataset browser. Only web review annotations are writable."""
from __future__ import annotations
import ast
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile
from functools import lru_cache
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get('CODEASMEM_DATA_ROOT', ROOT / 'data/unified_worlds')).resolve()
CODE = ROOT / 'worldbridge'
STATE = HERE / '.local'
APP_ID = 'codeasmem-local-v1'
NAMES = {'cp2077': '赛博朋克 2077', 'gta5': 'GTA V', 'rdr2': '荒野大镖客 2'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


audit_path = os.environ.get('CODEASMEM_AUDIT')
AUDIT = {x['case_id']: x for x in read_json(audit_path).get('cases', [])} if audit_path else {}
validation_path = DATA / 'batch_validation.json'
VALIDATION = read_json(validation_path) if validation_path.is_file() else {}
VALID_CASES = {(x['game'], x['case_id']): x for x in VALIDATION.get('cases', [])}
CATALOG = {}
for game in NAMES:
    for path in sorted((DATA / game).glob('*/episode.json')):
        ep = read_json(path)
        qa = read_json(path.parent / 'qa.json')
        report_path = path.parent / 'proxy/main/render_report.json'
        report = read_json(report_path) if report_path.is_file() else {}
        if not report.get('complete_episode'):
            continue  # Browser displays complete main-view pairs; conversion-only data stays on disk.
        key = (game, ep['case_id'])
        CATALOG[key] = dict(game=game, game_name=NAMES[game], case_id=ep['case_id'],
            frames=ep['frame_count'], duration_s=ep['duration_s'],
            site=ep.get('site_id') or AUDIT.get(ep['case_id'], {}).get('location_id') or '未提供',
            triangles=qa['static_triangles'], track_only=qa.get('track_only_entities', 0),
            structure_passed=key in VALID_CASES and not VALID_CASES[key]['issues'],
            structurally_validated=key in VALID_CASES,
            visual_review_status=qa.get('visual_review_status'),
            source_training_eligibility=ep.get('source_training_eligibility'),
            path=path.parent, ep=ep, qa=qa, report=report)

app = FastAPI(title='WorldBridge 本地研究工作台', docs_url=None, redoc_url=None)
review_lock = threading.Lock()


def case(game, case_id):
    try:
        return CATALOG[(game, case_id)]
    except KeyError:
        raise HTTPException(404, '案例不存在')


def safe_json_array(a):
    a = np.asarray(a)
    if a.dtype.kind == 'f':
        result = np.round(a, 4).astype(object)
        result[~np.isfinite(a)] = None
        return result.tolist()
    return a.tolist()


def public(c):
    return {k: v for k, v in c.items() if k not in ('path', 'ep', 'qa', 'report')}


def get_reviews():
    p = STATE / 'reviews.local.json'
    return read_json(p) if p.exists() else {}


@app.middleware('http')
async def local_requests(request: Request, call_next):
    # A local service should not accept cross-origin writes from unrelated pages.
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail': '仅允许本地工作台提交'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': '请求来源不匹配'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/api/health')
def health():
    return {'app': APP_ID, 'episodes': len(CATALOG), 'pid': os.getpid(), 'data_root': str(DATA), 'web_root': str(HERE)}


@app.get('/api/summary')
def summary():
    games = {}
    for game, name in NAMES.items():
        rows = [c for c in CATALOG.values() if c['game'] == game]
        games[game] = {'name': name, 'episodes': len(rows), 'frames': sum(c['frames'] for c in rows)}
    return {'episodes': len(CATALOG), 'frames': sum(c['frames'] for c in CATALOG.values()),
        'games': games, 'proxy_videos': sum(4 for c in CATALOG.values()), 'rendered_views': len(CATALOG),
        'indexed_views': sum(len(c['ep']['views']) for c in CATALOG.values()),
        'point_records': sum(c['frames'] * c['ep'].get('point_track_semantics', {}).get('entities', 0)
            * c['ep'].get('point_track_semantics', {}).get('points_per_entity', 0) for c in CATALOG.values()),
        'track_only_entities': sum(c['track_only'] for c in CATALOG.values()),
        'structural_passed': sum(c['structure_passed'] for c in CATALOG.values()),
        'structural_unvalidated': sum(not c['structurally_validated'] for c in CATALOG.values()),
        'structural_failures': sum(c['structurally_validated'] and not c['structure_passed'] for c in CATALOG.values()),
        'visual_approved': sum(VALID_CASES.get(k, {}).get('visual_quality_approved', False) for k in CATALOG),
        'validation_scope': VALIDATION.get('scope', '尚未运行结构校验'), 'model_trained': False,
        'paths': {'data': str(DATA), 'code': str(CODE), 'web': str(HERE)}}


@app.get('/api/cases')
def cases():
    reviews = get_reviews()
    return [dict(public(c), review=reviews.get('/'.join(k), {})) for k, c in CATALOG.items()]


@app.get('/api/cases/{game}/{case_id}')
def detail(game: str, case_id: str):
    c = case(game, case_id)
    report = c['report']
    return dict(public(c), episode=c['ep'], qa=c['qa'],
        render={k: v for k, v in report.items() if k != 'chunks'},
        path=str(c['path']), review=get_reviews().get(f'{game}/{case_id}', {}),
        program=read_json(c['path'] / 'program.json'),
        source_segments=AUDIT.get(case_id, {}).get('segments'))


@lru_cache(maxsize=8)
def timing_data(game, case_id):
    c = case(game, case_id)
    with np.load(c['path'] / 'timeline.npz', allow_pickle=False) as z:
        times = z['time_us'].tolist()
        frames = z['source_video_frame'].tolist()
    # Video packet PTS are sorted into presentation order. Do not equate the
    # source VFR playback clock with the generated CFR preview clock.
    executable = 'ffprobe'
    import shutil
    if not shutil.which(executable):
        raise HTTPException(503, '未找到 ffprobe，无法精确对齐源视频时间')
    source = c['ep']['views']['main']['source_video']
    p = subprocess.run([executable, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'packet=pts_time', '-of', 'json', source],
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if p.returncode:
        raise HTTPException(503, '源视频时间读取失败；请确认 episode.json 中的本地路径可访问')
    pts = sorted(float(x['pts_time']) for x in json.loads(p.stdout)['packets'] if 'pts_time' in x)
    if len(pts) != c['frames']:
        raise HTTPException(409, f'源视频时间数量 {len(pts)} 与统一帧数 {c["frames"]} 不一致')
    # MP4 presentation time includes its edit-list/start offset in HTMLVideo.
    # Subtracting the first packet PTS shifts CP2077 by ~16 ms and can select
    # the preceding decoded frame even when currentTime matches our target.
    pts = [round(x, 7) for x in pts]
    return {'time_us': times, 'source_frame': frames,
        'source_pts_s': pts, 'source_first_pts_s': pts[0], 'proxy_fps': c['report']['preview_fps'],
        'mode': 'frame_index_join; source_presentation_pts; proxy_CFR'}


@app.get('/api/cases/{game}/{case_id}/timing')
def timing(game: str, case_id: str):
    return timing_data(game, case_id)


@lru_cache(maxsize=6)
def track_data(game, case_id):
    c = case(game, case_id)
    with np.load(c['path'] / 'joints.npz', allow_pickle=False) as z:
        roots = z['root_world'].copy()
        root_valid = z['root_valid'].copy()
        joints = z['world_xyz'].copy()
        valid = z['valid'].copy()
        ids = z['instance_ids'].tolist()
        names = z['joint_names'].tolist()
        parents = z['parent_joint'].tolist()
    entities = read_json(c['path'] / 'entities.json')['entities']
    registry = {x['instance_id']: x for x in entities}
    series = []
    for entity_idx, iid in enumerate(ids):
        e = registry.get(iid, {})
        series.append({'instance_id': iid, 'source_id': e.get('source_id'),
            'category': e.get('category'), 'is_focus': e.get('is_focus', False),
            'appearance': e.get('appearance') or e.get('model_id'),
            'roots': safe_json_array(roots[:, entity_idx]),
            'root_valid': root_valid[:, entity_idx].tolist(),
            'joints': safe_json_array(joints[:, entity_idx]),
            'joint_valid': valid[:, entity_idx].tolist()})
    view, projection = [], []
    with gzip.open(c['path'] / 'frames.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            camera = json.loads(line)['cameras']['main']
            view.append(camera['view'])
            projection.append(camera['projection'])
    # Coarse point-in-frustum indicator, not an occlusion or pixel-visibility label.
    focus = next((i for i, e in enumerate(series) if e['is_focus']), 0)
    states = []
    for t in range(c['frames']):
        points = joints[t, focus][valid[t, focus]]
        if not len(points) and root_valid[t, focus]:
            points = roots[t, focus:focus+1]
        if not len(points):
            states.append('unknown')
            continue
        homogeneous = np.c_[points, np.ones(len(points))]
        clip = homogeneous @ np.asarray(view[t]).T @ np.asarray(projection[t]).T
        w = clip[:, 3]
        ndc = clip[:, :3] / np.where(abs(w[:, None]) > 1e-8, w[:, None], np.nan)
        inside = (w > 0) & np.isfinite(ndc).all(axis=1) & (np.abs(ndc) <= 1).all(axis=1)
        fraction = float(inside.mean())
        states.append('oos' if fraction == 0 else 'visible' if fraction >= .5 else 'partial')
    segments = []
    for i, state in enumerate(states):
        if segments and segments[-1]['state'] == state:
            segments[-1]['end_frame'] = i
        else:
            segments.append({'state': state, 'start_frame': i, 'end_frame': i})
    with np.load(c['path'] / 'points.npz', allow_pickle=False) as z:
        point_shape = list(z['world_xyz'].shape)
        tracking_kind = str(z['tracking_kind'])
    return {'entities': series, 'joint_names': names, 'parent_joint': parents,
        'segments': AUDIT.get(case_id, {}).get('segments') or segments,
        'segment_provenance': 'source_capture_frustum_audit' if case_id in AUDIT else 'canonical_focus_joint_frustum_estimate',
        'segment_scope': 'frustum_only_not_pixel_occlusion',
        'point_shape': point_shape, 'tracking_kind': tracking_kind,
        'track_only_entities': c['track_only'], 'coordinate_space': 'episode_local_world_Z_up'}


@app.get('/api/cases/{game}/{case_id}/tracks')
def tracks(game: str, case_id: str):
    return track_data(game, case_id)


@app.get('/media/{game}/{case_id}/{channel}')
def media(game: str, case_id: str, channel: str):
    c = case(game, case_id)
    if channel == 'rgb':
        path = Path(c['ep']['views']['main']['source_video'])
    elif channel in ('white', 'semantic', 'actors', 'control'):
        path = c['path'] / 'proxy/main' / f'{channel}.mp4'
    elif channel == 'preview':
        return FileResponse(c['path'] / 'proxy/main/preview.jpg', media_type='image/jpeg')
    else:
        raise HTTPException(404)
    if not path.is_file():
        raise HTTPException(404, '媒体不存在；请确认原始数据路径仍可访问')
    return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'private, max-age=3600'})


@lru_cache(maxsize=4)
def buffer_chunk(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ('depth', 'instance', 'actor_depth', 'actor_instance')}


def get_buffer(game, case_id, frame, scope):
    c = case(game, case_id)
    if not 0 <= frame < c['frames']:
        raise HTTPException(422, '帧编号超出范围')
    if scope not in ('full', 'actor'):
        raise HTTPException(422, '无效缓冲范围')
    for chunk in c['report']['chunks']:
        if chunk['start_frame'] <= frame < chunk['start_frame'] + chunk['frames']:
            arrays = buffer_chunk(str(c['path'] / 'proxy/main' / chunk['path']))
            prefix = '' if scope == 'full' else 'actor_'
            j = frame - chunk['start_frame']
            return arrays[prefix + 'depth'][j], arrays[prefix + 'instance'][j]
    raise HTTPException(404)


@app.get('/api/cases/{game}/{case_id}/buffer/{frame}')
def buffer_image(game: str, case_id: str, frame: int, kind: str = 'depth', scope: str = 'full'):
    import cv2
    d, ids = get_buffer(game, case_id, frame, scope)
    if kind == 'depth':
        finite = np.isfinite(d) & (d > 0)
        lo, hi = np.percentile(d[finite], [2, 98]) if finite.any() else (0, 1)
        normalized = np.clip((np.nan_to_num(d, posinf=hi) - lo) / max(hi - lo, 1e-5), 0, 1)
        rgb = cv2.applyColorMap((normalized * 255).astype('uint8'), cv2.COLORMAP_TURBO)
        rgb[~finite] = (26, 22, 18)
    elif kind == 'instance':
        rgb = np.stack([(ids * 97 + 43) % 220 + 35, (ids * 57 + 121) % 220 + 35,
                        (ids * 139 + 71) % 220 + 35], axis=-1).astype('uint8')
        rgb[ids == 0] = (26, 22, 18)
    else:
        raise HTTPException(422, '无效缓冲类型')
    ok, encoded = cv2.imencode('.png', rgb)
    if not ok:
        raise HTTPException(500)
    return Response(encoded.tobytes(), media_type='image/png')


@app.get('/api/cases/{game}/{case_id}/sample/{frame}')
def pixel_sample(game: str, case_id: str, frame: int, x: int, y: int, scope: str = 'full'):
    depth, ids = get_buffer(game, case_id, frame, scope)
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise HTTPException(422, '像素坐标超出范围')
    value = float(depth[y, x])
    return {'x': x, 'y': y, 'depth_camera_z_m': value if np.isfinite(value) else None,
            'instance_id': int(ids[y, x]), 'scope': scope, 'native_depth': False}


class Review(BaseModel):
    status: str
    note: str = Field(default='', max_length=3000)


@app.post('/api/cases/{game}/{case_id}/review')
def review(game: str, case_id: str, body: Review):
    case(game, case_id)
    if body.status not in ('pending', 'candidate', 'repair'):
        raise HTTPException(422, '无效复核状态')
    with review_lock:
        data = get_reviews()
        value = dict(status=body.status, note=body.note,
            updated_at=datetime.now(timezone.utc).isoformat(),
            scope='local_user_review_only; original_quality_eligibility_unchanged')
        data[f'{game}/{case_id}'] = value
        STATE.mkdir(parents=True, exist_ok=True)
        target = STATE / 'reviews.local.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(target)
    return value


CODE_SYMBOLS = {
    'cp': ('adapters.py', 'CP2077Adapter.frames'),
    'gta': ('adapters.py', 'GTA5Adapter.frames'),
    'rdr': ('adapters.py', 'RDR2Adapter.frames'),
    'camera': ('common.py', 'camera_record'),
    'renderer': ('wgl.py', 'GLRenderer.render'),
    'tracks': ('tracks.py', 'export_tracks'),
    'runtime': ('runtime.py', 'ProgramWorld.execute'),
    'loader': ('data.py', 'iter_pairs'),
}


@app.get('/api/code/{symbol}')
def source_code(symbol: str):
    if symbol not in CODE_SYMBOLS:
        raise HTTPException(404)
    filename, qualified = CODE_SYMBOLS[symbol]
    text = (CODE / filename).read_text(encoding='utf-8')
    nodes = ast.parse(text).body
    node = None
    for name in qualified.split('.'):
        node = next(x for x in nodes if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name)
        nodes = node.body
    return {'file': str(CODE / filename), 'symbol': qualified, 'line': node.lineno,
            'text': '\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno])}


class Window(BaseModel):
    task: str = 'dynamics'
    start: int = Field(ge=0)
    cutoff: int = Field(ge=0)
    end: int = Field(ge=0)
    condition: str = 'control'


@app.post('/api/cases/{game}/{case_id}/recipe')
def recipe(game: str, case_id: str, w: Window):
    c = case(game, case_id)
    if not w.start <= w.cutoff < w.end < c['frames']:
        raise HTTPException(422, '需要 0 ≤ 起始帧 ≤ 历史截止帧 < 目标终帧 < 总帧数')
    if w.task not in ('dynamics', 'renderer') or w.condition not in ('control', 'actors', 'white', 'semantic'):
        raise HTTPException(422, '无效训练任务或条件通道')
    dynamics = w.task == 'dynamics'
    student = ['past_rgb_or_estimated_state', 'past_time_us']
    if not dynamics:
        student += ['recorded_future_proxy', 'future_camera_as_explicit_condition']
    return {'schema': 'worldbridge.training_window.v1', 'task': w.task,
        'game': game, 'case_id': case_id, 'episode': str(c['path']),
        'context': {'start': w.start, 'end_inclusive': w.cutoff},
        'target': {'start': w.cutoff + 1, 'end_inclusive': w.end},
        'condition': w.condition, 'student_inputs': student,
        'target_only': ['future_world_points', 'future_joint_valid', 'future_rgb'],
        'excluded_student_inputs': ['future_entity_state', 'future_velocity', 'future_events',
            'future_teacher_views', 'future_ground_truth_proxy'] if dynamics else [],
        'future_camera_policy': 'separate_known_camera_condition_or_predict_it_explicitly',
        'alignment': 'frame_index; native time_us for dynamics',
        'mask_policy': 'valid & finite; proxy covered is not native visibility',
        'split_policy': 'group_by_scene_identity_capture_session_before_windowing',
        'split': 'unassigned', 'source_training_eligibility': c['source_training_eligibility'],
        'visual_review': get_reviews().get(f'{game}/{case_id}', {'status': 'pending'}),
        'training_ready': False, 'approval_required_before_training': 'data_quality_review_and_split_assignment',
        'note': '窗口定义与数据准备配置；未训练模型。动力学输入与标签在 prepare_window.py 中分文件输出。'}


GLOBAL_DOWNLOADS = {
    'index': DATA / 'training_pairs.jsonl',
    'validation': DATA / 'batch_validation.json',
    'report': ROOT / 'docs/data.md',
    'guide': HERE / 'training_guide.zh.txt',
}
CASE_DOWNLOADS = {'episode.json', 'qa.json', 'program.json', 'joints.npz', 'points.npz', 'timeline.npz'}


@app.get('/download/global/{name}')
def download_global(name: str):
    if name in ('code', 'web_code'):
        exports = STATE / 'exports'
        exports.mkdir(parents=True, exist_ok=True)
        target = exports / f'{name}.zip'
        folders = ['worldbridge', 'schemas'] if name == 'code' else ['web', 'scripts']
        with review_lock:
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                for file in [ROOT/'pyproject.toml', ROOT/'README.md', ROOT/'configs/example.json']:
                    archive.write(file, file.relative_to(ROOT).as_posix())
                for folder in folders:
                    for file in sorted((ROOT / folder).rglob('*')):
                        if file.is_file() and file.suffix in {'.py', '.json', '.html', '.css', '.js', '.ps1', '.cmd', '.txt'} and not {'.local', '__pycache__'} & set(file.parts):
                            archive.write(file, file.relative_to(ROOT).as_posix())
        return FileResponse(target, filename=f'codeasmem_{name}.zip')
    p = GLOBAL_DOWNLOADS.get(name)
    if p is None or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name)


@app.get('/download/{game}/{case_id}/{name}')
def download_case(game: str, case_id: str, name: str):
    c = case(game, case_id)
    if name not in CASE_DOWNLOADS:
        raise HTTPException(404)
    return FileResponse(c['path'] / name, filename=f'{game}_{case_id}_{name}')


@app.get('/overview.jpg')
def overview():
    return FileResponse(ROOT / 'docs/assets/proxy_overview.jpg', media_type='image/jpeg')


app.mount('/', StaticFiles(directory=HERE / 'static', html=True), name='static')


if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
