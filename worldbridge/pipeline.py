from __future__ import annotations
import gzip
import json
import time
from pathlib import Path
import numpy as np
from .adapters import discover, make_adapter
from .common import SCHEMA, CLASSES, JOINTS, PARENTS, rows, write_json


def controller_evidence(adapter):
    """Common controller envelope; source tasks are not silently executable rules."""
    if adapter.game=='rdr2':
        native=json.loads((adapter.case/'code/state_machine.json').read_text(encoding='utf-8-sig'))
        regions=[]
        def command(raw):
            parameters=dict(raw.get('parameters') or {})
            target=parameters.get('target_raw')
            if target is not None:
                parameters['target_position']=(np.asarray(target)-adapter.origin).tolist()
            return dict(op='request_behavior',name=raw.get('name',raw.get('action')),
                        source_entity_id=raw.get('entity_id'),parameters=parameters,
                        source_kind='collector_command',native=raw)
        for name,r in native.get('parallel_regions',{}).items():
            regions.append(dict(name=name,initial_state=r.get('initial_state'),
                transitions=[dict(from_state=t.get('from'),to_state=t.get('to'),trigger=t.get('trigger'),
                                  command=command(t['action']) if t.get('action') else None,
                                  observed_evidence=t.get('observed_evidence')) for t in r.get('transitions',[])]))
        return dict(schema='worldbridge.controller.v1',source_kind='collector_declared',
                    executable_without_game_behavior_backend=False,initial_commands=[command(c) for c in native.get('case_initialization',{}).get('commands',[])],
                    regions=regions,native=native)
    if adapter.game=='gta5':
        return dict(schema='worldbridge.controller.v1',source_kind='collector_declared_and_observed',
                    executable_without_game_behavior_backend=False,initial_commands=[],regions=[],
                    native=dict(batch_case=adapter.episode.get('batch_case'),camera_motion_plans=adapter.episode.get('camera_motion_plans')))
    return dict(schema='worldbridge.controller.v1',source_kind='collector_camera_only',
                executable_without_game_behavior_backend=False,initial_commands=[],regions=[],
                native=adapter.episode.get('collector_configuration'))


def convert_case(game, source, output_root):
    source, output_root=Path(source).resolve(),Path(output_root).resolve()
    target=output_root/game/source.name
    if (target/'episode.json').exists():
        return dict(game=game,case_id=source.name,status='existing',path=str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=target.with_name(target.name+'.partial')
    if stage.exists():
        old=stage.with_name(stage.name+f'.previous_{time.time_ns()}')
        stage.rename(old)
    stage.mkdir()
    adapter=make_adapter(game,source)
    count=0;last=-1;times=[];source_ids=[];first=None;entity_counts=[];joints_seen=0
    start=time.perf_counter()
    try:
        with gzip.open(stage/'frames.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as f:
            for frame in adapter.frames():
                if frame['index']!=count or (count and frame['time_us']<=last):
                    raise ValueError('Non-consecutive index or non-increasing simulation timestamp')
                for camera in frame['cameras'].values():
                    if not np.isfinite(camera['view']).all() or not np.isfinite(camera['projection']).all():
                        raise ValueError('Non-finite normalized camera')
                if first is None:first=frame
                last=frame['time_us'];times.append(last);source_ids.append(frame['source_frame_id'])
                entity_counts.append(len(frame['entities']))
                joints_seen+=sum(bool(e.get('joints')) for e in frame['entities'])
                f.write(json.dumps(frame,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                count+=1
        if count<1:raise ValueError('Empty episode')
        static=adapter.static_mesh()
        if len(static['faces']):
            if static['faces'].max()>=len(static['vertices']) or not np.isfinite(static['vertices']).all():
                raise ValueError('Invalid static mesh')
        np.savez_compressed(stage/'static.npz',**static)
        write_json(stage/'entities.json',dict(schema=SCHEMA,entities=adapter.registry.records))
        with (stage/'events.jsonl').open('w',encoding='utf-8') as f:
            for event in adapter.events:f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        np.savez_compressed(stage/'timeline.npz',time_us=np.asarray(times,np.int64),source_frame_id=np.asarray(source_ids),
                            source_video_frame=np.arange(count,dtype=np.int32))
        header=adapter.header()
        header.update(schema=SCHEMA,frame_count=count,duration_s=(last-times[0])/1e6,
                      classes=CLASSES,joint_names=JOINTS,parent_joint=PARENTS,
                      files=dict(frames='frames.jsonl.gz',static='static.npz',entities='entities.json',
                                 events='events.jsonl',timeline='timeline.npz',program='program.json'))
        write_json(stage/'episode.json',header)
        write_json(stage/'first_frame.json',first)
        write_json(stage/'program.json',dict(schema='worldbridge.program.v1',mode='recorded_replay',
                    initial_state=dict(frame=0),instructions=[dict(op='replay',timeline='frames.jsonl.gz')],
                    event_stream='events.jsonl',source_commands_are_evidence=True,
                    controller_evidence=controller_evidence(adapter),
                    predictive_rules_recovered=False,available_runtime_ops=['replay','set_velocity','set_state','advance']))
        from .tracks import export_tracks
        export_tracks(stage)
        qa=dict(schema='worldbridge.qa.v1',converted=True,source_unchanged=True,rgb_used_for_geometry=False,
                frame_count=count,time_strictly_increasing=True,exact_frame_join=True,
                static_triangles=len(static['faces']),static_vertices=len(static['vertices']),
                entity_registry_size=len(json.loads((stage/'entities.json').read_text(encoding='utf-8'))['entities']),dynamic_entity_count_min=min(entity_counts),
                dynamic_entity_count_max=max(entity_counts),entity_frames_with_canonical_joints=joints_seen,
                event_count=len(adapter.events),warnings=adapter.warnings,static_coverage=adapter.static_info,
                native_camera_to_rgb_independently_verified=False,visual_review_status='required',
                dense_surface_ground_truth=False,source_training_eligibility=header['source_training_eligibility'],
                source_quality=header.get('source_quality'),
                elapsed_seconds=time.perf_counter()-start)
        write_json(stage/'qa.json',qa)
        stage.rename(target)
        return dict(game=game,case_id=source.name,status='converted',path=str(target),frames=count,
                    static_triangles=len(static['faces']),elapsed_seconds=qa['elapsed_seconds'])
    except Exception as exc:
        write_json(stage/'failure.json',dict(game=game,source=str(source),error=repr(exc)))
        raise


def inventory(config):
    results=[]
    for game,root in config['datasets'].items():
        cases=discover(root,game)
        results.extend(dict(game=game,source=str(p),case_id=p.name) for p in cases)
    return results


def write_training_index(output_root):
    output_root=Path(output_root).resolve();records=[]
    for path in sorted(output_root.glob('*/*/episode.json')):
        ep=json.loads(path.read_text(encoding='utf-8'));case=path.parent
        for view,entry in ep['views'].items():
            report=case/'proxy'/view/'render_report.json'
            rendered=report.is_file()
            if rendered:rendered=bool(json.loads(report.read_text(encoding='utf-8')).get('complete_episode'))
            records.append(dict(schema='worldbridge.training_pair.v1',game=ep['game'],case_id=ep['case_id'],
                                site_id=ep.get('site_id'),view=view,episode=str(case),frame_count=ep['frame_count'],
                                rgb_source=entry['source_video'],rgb_present=Path(entry['source_video']).exists(),
                                proxy_white=str(case/'proxy'/view/'white.mp4') if rendered else None,
                                proxy_semantic=str(case/'proxy'/view/'semantic.mp4') if rendered else None,
                                proxy_actors=str(case/'proxy'/view/'actors.mp4') if rendered else None,
                                proxy_control=str(case/'proxy'/view/'control.mp4') if rendered else None,
                                proxy_buffers=str(case/'proxy'/view/'buffers') if rendered else None,
                                points_3d=str(case/'points.npz'),canonical_joints=str(case/'joints.npz'),
                                alignment='exact_frame_index; native_simulation_time_us_in_timeline_and_buffers',
                                proxy_video_time_mode='CFR_preview; training_loader_uses_frame_indices',
                                generated_proxy_available=rendered,qa=str(case/'qa.json'),
                                label_scope='coarse_proxy_not_native_pixel_visibility',
                                training_eligibility='requires_visual_review',split='unassigned_group_by_site_and_identity',
                                teacher_only_views=['monitor','ego','follow']))
    with (output_root/'training_pairs.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=dict(episodes=len(set(r['episode'] for r in records)),pairs=len(records),
                 rendered_pairs=sum(r['generated_proxy_available'] for r in records),
                 games={g:len(set(r['episode'] for r in records if r['game']==g)) for g in ['cp2077','gta5','rdr2']})
    write_json(output_root/'dataset_summary.json',summary)
    return summary
