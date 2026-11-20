from __future__ import annotations
import gzip
import json
import os
import uuid
from pathlib import Path
import numpy as np

SCHEMA = 'worldbridge.scene.v1'
CLASSES = {'unknown': 0, 'person': 1, 'vehicle': 2, 'building': 3,
           'tree': 4, 'bush': 5, 'road': 6, 'terrain': 7, 'prop': 8,
           'facility': 9, 'door': 10, 'animal': 11}
PALETTE = np.array([[110, 85, 120], [233, 137, 75], [77, 155, 209],
                    [190, 181, 157], [87, 145, 100], [135, 171, 97],
                    [145, 147, 151], [172, 161, 136], [153, 160, 174],
                    [194, 151, 182], [210, 185, 103], [158, 111, 75]], np.uint8)
JOINTS = ['SKEL_Pelvis', 'SKEL_Spine0', 'SKEL_Spine1', 'SKEL_Spine2',
          'SKEL_Neck_1', 'SKEL_Head', 'SKEL_L_Clavicle', 'SKEL_L_UpperArm',
          'SKEL_L_Forearm', 'SKEL_L_Hand', 'SKEL_R_Clavicle', 'SKEL_R_UpperArm',
          'SKEL_R_Forearm', 'SKEL_R_Hand', 'SKEL_L_Thigh', 'SKEL_L_Calf',
          'SKEL_L_Foot', 'SKEL_R_Thigh', 'SKEL_R_Calf', 'SKEL_R_Foot']
PARENTS = [-1, 0, 1, 2, 3, 4, 4, 6, 7, 8, 4, 10, 11, 12, 0, 14, 15, 0, 17, 18]
GTA_BONES = {11816: 'SKEL_Pelvis', 23553: 'SKEL_Spine0', 24816: 'SKEL_Spine1',
             24817: 'SKEL_Spine2', 39317: 'SKEL_Neck_1', 31086: 'SKEL_Head',
             64729: 'SKEL_L_Clavicle', 45509: 'SKEL_L_UpperArm', 61163: 'SKEL_L_Forearm',
             18905: 'SKEL_L_Hand', 10706: 'SKEL_R_Clavicle', 40269: 'SKEL_R_UpperArm',
             28252: 'SKEL_R_Forearm', 57005: 'SKEL_R_Hand', 58271: 'SKEL_L_Thigh',
             63931: 'SKEL_L_Calf', 14201: 'SKEL_L_Foot', 51826: 'SKEL_R_Thigh',
             36864: 'SKEL_R_Calf', 52301: 'SKEL_R_Foot'}


def canonical_joint(name):
    return {'SKEL_Neck0':'SKEL_Neck_1'}.get(str(name),str(name))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def unwrap(v):
    return v.get('value') if isinstance(v, dict) and 'value' in v else v


def category(value):
    s = str(value or '').lower()
    if any(w in s for w in ('npc', 'pedestrian', 'human', 'person', 'playerpuppet')):
        return 'person'
    if any(w in s for w in ('animal', 'horse')):
        return 'animal'
    if any(w in s for w in ('vehicle', 'car', 'wagon')):
        return 'vehicle'
    if 'building' in s:
        return 'building'
    if s=='tree' or s.startswith('tree_') or s.endswith('_tree'):
        return 'tree'
    if 'bush' in s:
        return 'bush'
    if 'road' in s and 'roadblock' not in s:
        return 'road'
    if s in ('terrain', 'coarse_mesh'):
        return 'terrain'
    if 'door' in s:
        return 'door'
    if any(w in s for w in ('facility', 'light', 'device', 'turret')):
        return 'facility'
    if s in ('unknown', 'none', ''):
        return 'unknown'
    return 'prop'


def finite_vector(value, n=3):
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    return a.tolist() if a.shape == (n,) and np.isfinite(a).all() else None


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    n = x*x + y*y + z*z + w*w
    if n < 1e-15:
        raise ValueError('Zero quaternion')
    s = 2/n
    return np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                     [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                     [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])


def transform(position, rotation=None):
    t = np.eye(4)
    t[:3, 3] = position
    if rotation is not None:
        t[:3, :3] = rotation
    return t


def rockstar_camera(position, rotation_deg, fov, width, height):
    """Rockstar order-2 camera, canonical camera axes x right/y up/z forward."""
    pitch, roll, yaw = np.deg2rad(rotation_deg)
    right = np.array([np.cos(yaw), np.sin(yaw), 0.])
    forward = np.array([-np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)])
    up = np.cross(right, forward)
    right, up = right*np.cos(roll)+up*np.sin(roll), up*np.cos(roll)-right*np.sin(roll)
    v = np.linalg.inv(transform(position, np.column_stack([right, up, forward])))
    f = height/(2*np.tan(np.deg2rad(fov)/2))
    return v, np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])


def perspective(k, width, height, near=.05, far=300.):
    """Camera z-forward, y-up; OpenGL NDC z in [-1,1]."""
    p = np.zeros((4, 4))
    p[0] = [2*k[0, 0]/width, 0, 2*k[0, 2]/width-1, 0]
    p[1] = [0, 2*k[1, 1]/height, 1-2*k[1, 2]/height, 0]
    p[2] = [0, 0, (far+near)/(far-near), -2*far*near/(far-near)]
    p[3, 2] = 1
    return p


def camera_record(v, p, width, height, origin, source, **extra):
    v = np.asarray(v, float).reshape(4, 4)
    p = np.asarray(p, float).reshape(4, 4)
    v = v @ transform(origin)
    if not np.isfinite(v).all() or not np.isfinite(p).all():
        raise ValueError('Non-finite camera')
    return dict(view=v.tolist(), projection=p.tolist(), resolution=[int(width), int(height)],
                camera_axes='x_right_y_up_z_forward', ndc_depth='minus_one_to_one',
                provenance=source, **extra)


class Registry:
    def __init__(self, game, case):
        self.game, self.case = game, case
        self.records = []
        self.by_key = {}

    def get(self, source_id, cat, **metadata):
        key = str(source_id)
        if key not in self.by_key:
            i = len(self.records)+1
            self.by_key[key] = i
            self.records.append(dict(instance_id=i, id=f'{self.game}/{self.case}/{key}',
                                     source_id=key, category=cat, semantic_id=CLASSES[cat], **metadata))
        return self.by_key[key]


def entity_record(registry, source_id, cat, matrix, origin, *, bounds=None,
                  velocity=None, joints=None, state=None, present=True, **metadata):
    t = np.asarray(matrix, float).reshape(4, 4).copy()
    t[:3, 3] -= origin
    if not np.isfinite(t).all():
        raise ValueError('Non-finite entity transform')
    result = dict(instance_id=registry.get(source_id, cat, **metadata), transform=t.tolist(),
                  present=bool(present), velocity=finite_vector(velocity), state=state or {})
    if bounds is not None:
        lo, hi = (np.asarray(x, float) for x in bounds)
        if np.isfinite([lo, hi]).all() and np.all(hi > lo):
            result['bounds'] = [lo.tolist(), hi.tolist()]
    if joints:
        result['joints'] = {k: (np.asarray(v)-origin).tolist() for k, v in joints.items()
                            if k in JOINTS and finite_vector(v) is not None}
    return result


def project(camera, points):
    pts = np.asarray(points)
    clip = np.c_[pts, np.ones(len(pts))] @ (np.asarray(camera['projection']) @ np.asarray(camera['view'])).T
    good = clip[:, 3] > 0
    ndc = clip[:, :3] / np.where(np.abs(clip[:, 3:4]) > 1e-12, clip[:, 3:4], np.nan)
    w, h = camera['resolution']
    uv = np.c_[(ndc[:, 0]+1)*w/2, (1-ndc[:, 1])*h/2]
    return uv, good & np.all(np.abs(ndc) <= 1, axis=1)
