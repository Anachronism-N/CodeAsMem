"""Unified masked 3D tensors; proxy/skeleton semantics are never conflated."""
from pathlib import Path
import numpy as np
from .common import JOINTS,PARENTS,read_json,rows,write_json


def export_tracks(episode):
    episode=Path(episode);ep=read_json(episode/'episode.json');registry=read_json(episode/'entities.json')['entities']
    source=Path(ep['source_root'])
    if ep['game']=='rdr2':
        known={e['source_id'] for e in registry}
        with np.load(source/'skeleton/bones.npz',allow_pickle=False) as native:
            for k,sid in enumerate(map(str,native['entity_ids'])):
                if sid in known:continue
                model=str(native['model_names'][k])
                cat='animal' if model.startswith('A_C_') else 'person'
                registry.append(dict(instance_id=len(registry)+1,id=f'rdr2/{ep["case_id"]}/{sid}',
                    source_id=sid,category=cat,semantic_id=11 if cat=='animal' else 1,
                    is_focus=False,track_only=True,appearance=model,model_id=int(native['model_hash'][k]),
                    geometry_provenance='native_skeleton_track_without_complete_state',
                    category_source='recorded_model_name_prefix'))
                known.add(sid)
        write_json(episode/'entities.json',dict(schema=ep['schema'],entities=registry))
    focus=[e for e in registry if e.get('is_focus')];ids=[e['instance_id'] for e in focus]
    lookup={ident:i for i,ident in enumerate(ids)};t=ep['frame_count'];n=len(focus)
    root=np.full((t,n,3),np.nan,np.float32);velocity=root.copy();root_valid=np.zeros((t,n),bool)
    joints=np.full((t,n,len(JOINTS),3),np.nan,np.float32);joint_valid=np.zeros(joints.shape[:-1],bool)
    times=[]
    for f in rows(episode/'frames.jsonl.gz'):
        i=f['index'];times.append(f['time_us'])
        for e in f['entities']:
            if e['instance_id'] not in lookup:continue
            j=lookup[e['instance_id']];root[i,j]=np.asarray(e['transform'])[:3,3];root_valid[i,j]=e['present']
            if e.get('velocity') is not None:velocity[i,j]=e['velocity']
            for name,p in e.get('joints',{}).items():
                if name in JOINTS:
                    k=JOINTS.index(name);joints[i,j,k]=p;joint_valid[i,j,k]=e['present']
    np.savez_compressed(episode/'joints.npz',world_xyz=joints,valid=joint_valid,root_world=root,
        root_valid=root_valid,velocity_world=velocity,velocity_valid=np.isfinite(velocity).all(-1)&root_valid,
        instance_ids=np.asarray(ids,np.uint32),joint_names=np.asarray(JOINTS),parent_joint=np.asarray(PARENTS,np.int16),
        time_us=np.asarray(times,np.int64),coordinate_space=np.asarray('episode_local_world'))
    origin=np.asarray(ep['world']['origin_native_xyz'])
    if ep['game'] in ['cp2077','gta5']:
        with np.load(source/'track/focus_surface256.npz',allow_pickle=False) as z:
            points=(z['world_xyz'].astype(np.float64)-origin).astype(np.float32)[:,None]
            valid=z['valid'][:,None].copy()
            point_ids=np.asarray([str(x) for x in z['point_ids']]) if 'point_ids' in z else np.asarray([str(i) for i in range(points.shape[2])])
            source_time=z['sync_pts_us'].copy();source_time-=source_time[0]
            kind=str(z['tracking_kind']) if 'tracking_kind' in z else 'bone_driven_proxy_surface'
        source_ids=ids[:1]
    else:
        with np.load(source/'skeleton/bones.npz',allow_pickle=False) as z:
            points=(z['world_xyz'].astype(np.float64)-origin).astype(np.float32)
            valid=z['joint_valid'].copy()&z['entity_present'][:,:,None]
            point_ids=z['joint_names'].copy();source_time=z['video_pts_us'].copy();source_time-=source_time[0]
            source_ids=[next(e['instance_id'] for e in registry if e['source_id']==str(s)) for s in z['entity_ids']]
            kind='native_skeleton_points_not_surface'
    if len(points)!=t or not np.array_equal(source_time,np.asarray(times)):
        raise ValueError('Original point tracks do not exactly match the normalized simulation timeline')
    valid &= np.isfinite(points).all(-1)
    np.savez_compressed(episode/'points.npz',world_xyz=points,valid=valid,point_ids=point_ids,
        instance_ids=np.asarray(source_ids,np.uint32),time_us=np.asarray(times,np.int64),tracking_kind=np.asarray(kind),
        coordinate_space=np.asarray('episode_local_world'),strict_surface_ground_truth=np.asarray(False))
    ep['files'].update(joints='joints.npz',points='points.npz')
    ep['point_track_semantics']=dict(kind=kind,points_per_entity=points.shape[2],entities=points.shape[1],
                                     track_only_entities=sum(bool(e.get('track_only')) for e in registry),
                                     strict_surface_ground_truth=False,missing_values='NaN_with_explicit_valid_mask')
    write_json(episode/'episode.json',ep)
    if (episode/'qa.json').exists():
        qa=read_json(episode/'qa.json');qa['entity_registry_size']=len(registry)
        qa['track_only_entities']=sum(bool(e.get('track_only')) for e in registry);write_json(episode/'qa.json',qa)
    return dict(game=ep['game'],case_id=ep['case_id'],frames=t,tracked_entities=points.shape[1],points=points.shape[2],kind=kind)
