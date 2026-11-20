from __future__ import annotations
import concurrent.futures
import json
from pathlib import Path
import zipfile
import numpy as np
from .common import read_json,rows,write_json
from .render import probe


def npz_headers(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                result[name.removesuffix('.npy')]=dict(shape=list(shape),dtype=str(dtype))
    return result


def validate_case(case,probe_source=True):
    ep=read_json(case/'episode.json');registry=read_json(case/'entities.json')['entities'];issues=[]
    with np.load(case/'timeline.npz',allow_pickle=False) as z:times=z['time_us'].copy()
    if len(times)!=ep['frame_count'] or (len(times)>1 and not np.all(np.diff(times)>0)):
        issues.append('Timeline count/order mismatch')
    if [e['instance_id'] for e in registry]!=list(range(1,len(registry)+1)):
        issues.append('Registry IDs are not contiguous and one-based')
    for file in ['points.npz','joints.npz']:
        if not (case/file).is_file():issues.append('Missing unified 3D tracks: '+file);continue
        with np.load(case/file,allow_pickle=False) as z:
            if not np.array_equal(z['time_us'],times):issues.append('3D track time mismatch: '+file)
            xyz=z['world_xyz'];valid=z['valid']
            if not np.isfinite(xyz[valid]).all():issues.append('Non-finite valid track: '+file)
            if xyz.shape[:3]!=valid.shape:issues.append('3D track mask shape: '+file)
            if len(z['instance_ids'])!=xyz.shape[1] or z['instance_ids'].max()>len(registry):issues.append('3D track entity mapping: '+file)
    n=0
    for frame in rows(case/'frames.jsonl.gz'):
        if frame['index']!=n or frame['time_us']!=times[n]:issues.append('Frame/timeline mismatch');break
        for entity in frame['entities']:
            if not 1<=entity['instance_id']<=len(registry):issues.append('Entity reference out of range')
            if not np.isfinite(entity['transform']).all():issues.append('Invalid entity transform')
        n+=1
    if n!=ep['frame_count']:issues.append('Frame stream truncated')
    h=npz_headers(case/'static.npz')
    if h['vertices']['shape'][1:]!=[3] or h['faces']['shape'][1:]!=[3]:issues.append('Static geometry shape')
    reportpath=case/'proxy/main/render_report.json'
    output=dict(game=ep['game'],case_id=ep['case_id'],frames=ep['frame_count'],instances=len(registry),
                static_triangles=h['faces']['shape'][0],issues=issues)
    if not reportpath.exists():issues.append('No complete main proxy');return output
    report=read_json(reportpath)
    if not report.get('complete_episode') or report['frames']!=ep['frame_count']:issues.append('Incomplete main proxy')
    chunk_times=[];chunk_sources=[];covered=[]
    expected={'depth':'float32','instance':'uint32','semantic':'uint16','covered':'bool',
              'actor_depth':'float32','actor_instance':'uint32','actor_semantic':'uint16'}
    for i,chunk in enumerate(report['chunks']):
        path=reportpath.parent/chunk['path'];headers=npz_headers(path)
        shape=[chunk['frames'],report['resolution'][1],report['resolution'][0]]
        for key,dtype in expected.items():
            if headers[key]!=dict(shape=shape,dtype=dtype):issues.append('Buffer shape/type mismatch: '+key)
        with np.load(path,allow_pickle=False) as z:
            chunk_times.extend(z['time_us'].tolist());chunk_sources.extend(z['source_frame'].tolist())
            if i==0:
                depth=z['depth'][0];owner=z['instance'][0];mask=z['covered'][0]
                if not np.array_equal(mask,owner>0):issues.append('Coverage/instance mismatch')
                if owner.max()>len(registry):issues.append('Rendered instance out of registry range')
                if np.any(~np.isfinite(depth[mask])) or np.any(depth[mask]<=0):issues.append('Invalid foreground proxy depth')
                if not np.isposinf(depth[~mask]).all():issues.append('Background depth must be +inf')
    if not np.array_equal(chunk_times,times):issues.append('Buffer/source simulation timestamps differ')
    if chunk_sources!=list(range(ep['frame_count'])):issues.append('Unexpected source video frame mapping')
    for name in ['white','semantic','actors','control']:
        if not (reportpath.parent/(name+'.mp4')).exists():issues.append('Missing '+name+' video')
        if int(report['encoded'][name]['nb_frames'])!=ep['frame_count']:issues.append('Proxy encoded frame count mismatch')
    if probe_source:
        try:
            media=probe(ep['views']['main']['source_video']);output['source_main_video']=media
            if int(media['nb_frames'])!=ep['frame_count']:issues.append('Source RGB frame count differs')
        except Exception as e:issues.append('Source RGB probe failed: '+repr(e))
    output.update(chunks=len(report['chunks']),render_seconds=report['total_seconds'],resolution=report['resolution'],
                  source_quality=ep.get('source_quality'),visual_quality_approved=False)
    return output


def validate_dataset(root,workers=3):
    root=Path(root);cases=sorted(p.parent for p in root.glob('*/*/episode.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(validate_case,cases))
    games={}
    for result in results:
        item=games.setdefault(result['game'],dict(episodes=0,frames=0,proxy_videos=0,failed_checks=0))
        item['episodes']+=1;item['frames']+=result['frames'];item['proxy_videos']+=4
        item['failed_checks']+=bool(result['issues'])
    report=dict(schema='worldbridge.batch_validation.v1',episodes=len(results),games=games,
                structural_failures=sum(bool(x['issues']) for x in results),
                visual_quality_approval_inferred=False,source_data_modified=False,
                scope='all normalized frame streams; all unified 3D tracks and masks; all buffer headers/timestamps; first depth/ID frame per episode; source main video metadata',
                cases=results)
    write_json(root/'batch_validation.json',report)
    return report
