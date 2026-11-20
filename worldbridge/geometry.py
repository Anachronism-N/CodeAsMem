"""Game-independent, coarse proxy primitives. No RGB is used for geometry."""
from __future__ import annotations
import numpy as np
from .common import CLASSES, JOINTS, PARENTS

BOX_V = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)/2
BOX_F = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], np.uint32)


class MeshBuilder:
    def __init__(self):
        self.vertices, self.faces, self.instances, self.semantics = [], [], [], []
        self.count = 0

    def add(self, v, f, instance, semantic):
        v, f = np.asarray(v, np.float32), np.asarray(f, np.uint32)
        if not len(v) or not len(f):
            return
        self.vertices.append(v); self.faces.append(f+self.count)
        self.instances.append(np.full(len(f), instance, np.uint32))
        self.semantics.append(np.full(len(f), semantic, np.uint32)); self.count += len(v)

    def box(self, matrix, bounds, instance, semantic):
        lo, hi = np.asarray(bounds, float)
        if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            return
        v = BOX_V*(hi-lo)+(hi+lo)/2
        t = np.asarray(matrix)
        self.add(v@t[:3, :3].T+t[:3, 3], BOX_F, instance, semantic)

    def bone(self, a, b, radius, instance, semantic):
        a, b = np.asarray(a), np.asarray(b)
        z = b-a; length = np.linalg.norm(z)
        if not .005 < length < 3:
            return
        z /= length
        ref = [0., 0., 1.] if abs(z[2]) < .9 else [1., 0., 0.]
        x = np.cross(ref, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        mat = np.eye(4); mat[:3,:3] = np.column_stack([x, y, z]); mat[:3,3] = (a+b)/2
        self.box(mat, [[-radius,-radius,-length/2],[radius,radius,length/2]], instance, semantic)

    def wire_box(self,matrix,bounds,instance,semantic):
        lo,hi=np.asarray(bounds,float);v=BOX_V*(hi-lo)+(hi+lo)/2;t=np.asarray(matrix)
        v=v@t[:3,:3].T+t[:3,3]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            # Long source bounds edges are split so the body-bone outlier gate is not applied.
            delta=v[b]-v[a];length=np.linalg.norm(delta)
            pieces=max(1,int(np.ceil(length/2.)))
            for j in range(pieces):self.bone(v[a]+delta*j/pieces,v[a]+delta*(j+1)/pieces,.025,instance,semantic)

    def packed(self):
        if not self.vertices:
            return dict(vertices=np.empty((0,3), np.float32), faces=np.empty((0,3), np.uint32),
                        instance_id=np.empty(0,np.uint32), semantic_id=np.empty(0,np.uint32))
        return dict(vertices=np.concatenate(self.vertices), faces=np.concatenate(self.faces),
                    instance_id=np.concatenate(self.instances), semantic_id=np.concatenate(self.semantics))


def dynamic_mesh(frame, registry, hide=()):
    b = MeshBuilder()
    for e in frame['entities']:
        if not e['present'] or e['instance_id'] in hide:
            continue
        i = e['instance_id']; meta = registry[i-1]; sem = meta['semantic_id']
        if meta.get('excluded_from_proxy'):
            continue
        joints = e.get('joints', {})
        edges = 0
        if meta['category'] == 'person' and joints:
            for j, name in enumerate(JOINTS):
                parent = PARENTS[j]
                if parent < 0 or name not in joints or JOINTS[parent] not in joints:
                    continue
                a, point = joints[JOINTS[parent]], joints[name]
                radius = .075 if any(w in name for w in ['Thigh','Spine','Neck']) else .045
                b.bone(a, point, radius, i, sem); edges += 1
            # Sparse RDR2/CP rigs expose pelvis and neck but no spine/clavicle chain.
            for a,z in [('SKEL_Pelvis','SKEL_Neck_1'),('SKEL_Neck_1','SKEL_L_UpperArm'),('SKEL_Neck_1','SKEL_R_UpperArm')]:
                if a in joints and z in joints and ('SKEL_Spine1' not in joints):
                    b.bone(joints[a],joints[z],.12 if a=='SKEL_Pelvis' else .06,i,sem);edges+=1
            if 'SKEL_Head' in joints:
                mat=np.eye(4);mat[:3,3]=joints['SKEL_Head']
                b.box(mat,[[-.10,-.10,-.12],[.10,.10,.14]],i,sem)
        if edges < 4 and 'bounds' in e:
            b.box(e['transform'],e['bounds'],i,sem)
    return b.packed()
