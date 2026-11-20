"""Native exports -> one renderable scene contract. All input is read-only."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .common import (CLASSES, GTA_BONES, JOINTS, Registry, camera_record, category, canonical_joint,
                     entity_record, finite_vector, perspective, quaternion_matrix,
                     read_json, rockstar_camera, rows, transform, unwrap)
from .geometry import MeshBuilder


def discover(root, game):
    root = Path(root).resolve()
    if game == 'cp2077':
        return sorted({p.parent.parent.parent for p in root.rglob('code/data/episode.json')})
    if game == 'gta5':
        return sorted(p.parent.parent for p in root.rglob('video/main.mp4'))
    if game == 'rdr2':
        return sorted(p.parent.parent for p in root.rglob('code/dynamics.npz'))
    raise ValueError(game)


def native_state(values):
    aliases = {'walking': ('walking','isWalking'), 'running': ('running','isRunning'),
               'sprinting': ('sprinting','isSprinting'), 'dead': ('dead','IsDead','is_dead_or_dying'),
               'in_combat': ('in_combat','IsPuppetInCombat','isInCombat'),
               'shooting': ('shooting','isShooting'), 'ragdoll': ('ragdoll','isRagdoll')}
    out = {}
    for target, sources in aliases.items():
        out[target] = next((unwrap(values[k]) for k in sources if k in values), None)
    return out


class Adapter:
    views = ['main','monitor','ego','follow']
    def __init__(self, case, game):
        self.case = Path(case).resolve(); self.game = game
        self.registry = Registry(game, self.case.name)
        self.origin = np.zeros(3)
        self.warnings = []; self.events = []
        self.native_sidecars = {}; self.source_videos = {}; self.site_id = None
        self.static_info = {}

    def header(self):
        if self.game=='rdr2':
            q=read_json(self.case/'code/metadata/binding.json')
            eligible=q.get('training_eligible')
            quality={k:q.get(k) for k in ['layout_status','training_eligible','final_code_qualified','inventory_complete']}
        else:
            result=read_json(self.case/'result.json');eligible=result.get('case_package',{}).get('training_eligible')
            quality=dict(status=result.get('case_package',{}).get('status'),
                         automated_capture_accepted=result.get('case_diagnostics',{}).get('automated_capture_accepted'),
                         batch_classification=result.get('batch_classification'))
        return dict(game=self.game, case_id=self.case.name, source_root=str(self.case),
                    world=dict(origin_native_xyz=self.origin.tolist(), axes='right_handed_z_up',
                               distance_unit='meter', unit_policy='one_game_world_unit_as_one_meter',
                               canonical_from_native='translation_by_negative_origin; native horizontal axes retained'),
                    camera_axes='x_right_y_up_z_forward', camera_ndc_depth='minus_one_to_one',
                    frame_semantics='scoped_observation_not_complete_world_state',
                    missing_entity_semantics='unknown_not_destroyed',
                    views={v: dict(source_video=str(p), camera_source=self.camera_sources[v])
                           for v,p in self.source_videos.items()},
                    site_id=self.site_id, source_training_eligibility=eligible,source_quality=quality,
                    native_semantics_preserved=True, predictive_native_rules_claimed=False)

    def static_mesh(self):
        return MeshBuilder().packed()


class CP2077Adapter(Adapter):
    def __init__(self, case):
        super().__init__(case, 'cp2077')
        self.episode = read_json(self.case/'code/data/episode.json')
        self.native_sidecars = {v:list(rows(self.case/f'video/{v}_present.jsonl')) for v in self.views}
        self.source_videos = {v:self.case/f'video/{v}.mp4' for v in self.views}
        self.origin = np.asarray(self.native_sidecars['main'][0]['camera']['T_world_from_camera_raw']).reshape(4,4)[:3,3].copy()
        self.camera_sources = {v:'native_GPU_render_descriptor' for v in self.views}
        self.focus_ids = set(map(str,self.episode['collector_configuration']['focus_entity_ids']))
        self.warnings += ['Static shared archive geometry is absent from the supplied batch.',
                          'Native depth is referenced by source manifests but absent.',
                          'Bone-derived proxy is not a skinned/clothing surface or true pixel visibility.']
        self.site_id = None
        self.static_info = dict(coverage='partial_nearby_entity_bounds', full_static_scene_available=False,
                                geometry='per-frame native loaded visual bounds; focused canonical joints')

    def frames(self):
        src = iter(rows(self.case/'code/data/states.jsonl'))
        first_t = int(self.native_sidecars['main'][0]['sync_pts_us'])
        last = None
        for i,main in enumerate(self.native_sidecars['main']):
            wanted = str(main['bundle_id'])
            s = next(src)
            if str(s['native_engine_frame_id']) != wanted:
                raise ValueError(f'CP2077 exact state/frame mismatch at {i}')
            cams = {}
            for v in self.views:
                row = self.native_sidecars[v][i]
                if str(row['bundle_id']) != wanted or row['sync_pts_us'] != main['sync_pts_us']:
                    raise ValueError(f'Cross-view timestamp mismatch: {v}/{i}')
                c = row['camera']; p = np.asarray(c['projection_raw']).reshape(4,4).copy()
                p[2] = 2*p[2]-p[3]  # native [0,1] depth -> OpenGL [-1,1]
                cams[v] = camera_record(c['view_raw'],p,row['width'],row['height'],self.origin,
                                        self.camera_sources[v], near_far_source='native_projection',
                                        source_video_frame=int(row['video_frame']))
            entities = []
            for e in s['scene']['entities']:
                ident = str(e['entity_id'])
                if ident in self.focus_ids:
                    continue
                matrix=e.get('T_world_from_entity_raw')
                if matrix is None:continue
                bounds=(e['bbox_local_min'],e['bbox_local_max']) if 'bbox_local_min' in e else None
                cat=category(e.get('semantic_class_candidate') or e.get('runtime_class'))
                item=entity_record(self.registry,ident,cat,matrix,self.origin,bounds=bounds,
                                   velocity=e.get('velocity_raw'),model_id=e.get('model_id'),
                                   excluded_from_proxy=e.get('runtime_class')=='PlayerPuppet',
                                   geometry_provenance=e.get('geometry_source','native_bounds'))
                entities.append(item)
            for e in s['entities']:
                joints={canonical_joint(b['canonical_name']):b['position_world_raw'] for b in e.get('bones',[])
                        if canonical_joint(b.get('canonical_name')) in JOINTS and b.get('valid')}
                context=next((r for r in s['scene']['entities'] if str(r['entity_id'])==str(e['entity_id'])),{})
                bounds=(context['bbox_local_min'],context['bbox_local_max']) if 'bbox_local_min' in context else None
                values=e.get('native_state',{})
                ent=entity_record(self.registry,e['entity_id'],'person',e['T_world_from_entity_raw'],self.origin,
                                  bounds=bounds,velocity=e.get('velocity_raw'),joints=joints,
                                  state=dict(motion=native_state(values),native=values),present=e.get('valid',True),
                                  model_id=e.get('model_id'),appearance=e.get('appearance'),is_focus=True,
                                  source_generation=e.get('generation'),geometry_provenance='canonical_joint_proxy')
                entities.append(ent)
            t=int(main['sync_pts_us'])-first_t
            states={e['instance_id']:e['state'] for e in entities if e['state']}
            if last is not None and states!=last:
                self.events.append(dict(time_us=t,kind='observed_state_delta',source_kind='observed',states=states))
            last=states
            yield dict(index=i,time_us=t,source_pts_us=int(main['sync_pts_us']),source_frame_id=wanted,
                       cameras=cams,entities=entities)
        for event in rows(self.case/'code/data/events.jsonl'):
            self.events.append(dict(time_us=None,kind='source_command',source_kind='collector_camera_command',native=event))


class GTA5Adapter(Adapter):
    def __init__(self,case):
        super().__init__(case,'gta5')
        self.episode=read_json(self.case/'code/data/episode.json')
        self.native_sidecars={v:list(rows(self.case/f'video/{v}_present.jsonl')) for v in self.views}
        self.source_videos={v:self.case/f'video/{v}.mp4' for v in self.views}
        first=self.native_sidecars['main'][0]
        self.origin=np.asarray(first['commanded_camera_position'],float)
        self.camera_sources={v:'capture_sidecar_commanded_pose_not_sample_camera' for v in self.views}
        self.site_id=self.episode.get('batch_case',{}).get('site_id')
        self.warnings += ['Commanded capture-slot camera is used; native render observer may be unavailable.',
                          'Sample camera and sample bone UV can refer to a different pose than packaged main RGB.',
                          'Static bounds/primitives approximate occupied surfaces; source coverage claims are retained separately.']
        self.static_info=dict(coverage='source_scene_layout', geometry='oriented native bounds',
                              source_complete_claim=None, normalized_proxy_complete_claim=False)

    def frames(self):
        native={str(r['timeline']['bundle_id']):r for r in rows(self.case/'code/data/frames.jsonl') if r.get('type')=='frame'}
        first_t=int(self.native_sidecars['main'][0]['sync_pts_us']);last={}
        for i,m in enumerate(self.native_sidecars['main']):
            wanted=str(m['bundle_id'])
            if wanted not in native:raise ValueError(f'GTA state missing bundle {wanted}')
            s=native[wanted];cams={}
            for v in self.views:
                row=self.native_sidecars[v][i]
                if str(row['bundle_id'])!=wanted:raise ValueError(f'GTA view mismatch {v}/{i}')
                source=s.get('view_cameras',{}).get(v,{})
                pos=row.get('commanded_camera_position',source.get('commanded_camera_position'))
                rot=row.get('commanded_camera_rotation',source.get('commanded_camera_rotation'))
                fov=row.get('commanded_camera_fov',source.get('commanded_camera_fov'))
                if pos is None or rot is None or fov is None:raise ValueError('Missing GTA capture camera')
                width,height=row['width'],row['height'];vmat,k=rockstar_camera(pos,rot,float(fov),width,height)
                cams[v]=camera_record(vmat,perspective(k,width,height),width,height,self.origin,self.camera_sources[v],
                                       source_video_frame=int(row['video_frame']),intrinsics=k.tolist(),
                                       render_observer_available=bool(row.get('render',{}).get('render_observer_available')))
            entities=[];focus=s.get('tracking',{}).get('focus_entities',[])
            primary=s.get('tracking',{}).get('primary_entity',{})
            focus_handles={str(e['handle']) for e in focus}
            for e in focus:
                cat=category(e.get('kind','person'));p=primary if primary.get('handle')==e.get('handle') else e
                mat=p.get('world_from_entity')
                if mat is None:
                    yaw=float(e.get('rotation_deg_xyz_order2',[0,0,0])[2]);q=[0,0,np.sin(np.deg2rad(yaw)/2),np.cos(np.deg2rad(yaw)/2)]
                    mat=transform(e['root'],quaternion_matrix(q))
                bb=p.get('model_aabb_local');bounds=(bb['min'],bb['max']) if bb else None
                joints={GTA_BONES[b['id']]:b['world'] for b in e.get('bones',[]) if b['id'] in GTA_BONES and b.get('valid',True)}
                state=dict(motion=native_state(p.get('behavior',{})),
                           collector_command=e.get('activity_controller'),native_task=p.get('ai_task_observation'))
                entities.append(entity_record(self.registry,e.get('focus_id',e['handle']),cat,mat,self.origin,
                                               bounds=bounds,velocity=e.get('velocity'),joints=joints,state=state,
                                               present=e.get('valid',True),model_id=e.get('model_hash'),is_focus=True,
                                               source_handle=e.get('handle'),geometry_provenance='native_bone_proxy'))
            for e in s.get('scene',{}).get('npcs',{}).get('entities',[]):
                if str(e.get('handle')) in focus_handles:continue
                pos=e.get('position_world');
                if pos is None:continue
                yaw=np.deg2rad(e.get('facing_heading_deg_gta',0));mat=transform(pos,quaternion_matrix([0,0,np.sin(yaw/2),np.cos(yaw/2)]))
                entities.append(entity_record(self.registry,e.get('npc_id',e['handle']),'person',mat,self.origin,
                                               velocity=e.get('velocity_world_mps'),state=dict(motion=native_state(e)),
                                               model_id=e.get('model_hash'),geometry_provenance='root_only_no_fabricated_bounds'))
            for e in s.get('scene',{}).get('vehicles',{}).get('entities',[]):
                bb=e.get('model_aabb_local');mat=e.get('world_from_entity')
                if mat is not None:
                    entities.append(entity_record(self.registry,'vehicle:'+str(e['handle']),'vehicle',mat,self.origin,
                                                   bounds=(bb['min'],bb['max']) if bb else None,
                                                   state=dict(native=e.get('vehicle_state')),model_id=e.get('model_hash'),
                                                   geometry_provenance='native_local_bounds'))
            t=int(m['sync_pts_us'])-first_t
            for e in entities:
                if e['state'] and last.get(e['instance_id'])!=e['state']:
                    self.events.append(dict(time_us=t,kind='state_observation',instance_id=e['instance_id'],
                                            source_kind='observed_with_separate_collector_command',state=e['state']))
                last[e['instance_id']]=e['state']
            for event in s.get('interactions',{}).get('events',[]):
                self.events.append(dict(time_us=t,kind='interaction',source_kind='native_observation',native=event))
            yield dict(index=i,time_us=t,source_pts_us=int(m['sync_pts_us']),source_frame_id=wanted,cameras=cams,entities=entities)

    def static_mesh(self):
        b=MeshBuilder();path=self.case/'code/scene/scene_layout.jsonl'
        if not path.exists():
            self.warnings.append('No packaged static scene layout.');return b.packed()
        collision_path=self.case/'code/scene/occluder_membrane.json'
        collision=read_json(collision_path) if collision_path.exists() else {}
        has_collision=bool(collision.get('collision_segments') or collision.get('collision_surfaces'))
        self.static_info.update(geometry='source_collision_planes_plus_conservative_object_bounds',
                                occupied_surface_truth=False,collision_available=has_collision,
                                camera_inside_bounds_policy='wire_only',large_bounds_policy='wire_only')
        for r in rows(path):
            if r.get('type')=='meta':
                self.static_info['source_complete_claim']=r.get('coverage_complete');continue
            if r.get('type')!='scene_instance':continue
            cat=category(r.get('semantic_class'));i=self.registry.get('static:'+str(r['id']),cat,
                           geometry_provenance='source_bounds_approximation',static=True)
            if has_collision and cat in ('building','road'):
                continue  # Bounding volumes enclose courtyards/interiors; use source collision planes instead.
            tr=r.get('transform',{});lo=r.get('bounds_min_local');hi=r.get('bounds_max_local')
            if lo is not None and hi is not None and tr.get('position_world') is not None:
                mat=transform(np.asarray(tr['position_world'])-self.origin,
                              quaternion_matrix(tr.get('rotation_quaternion_xyzw',[0,0,0,1])))
                scale=tr.get('scale_xy',1);sz=tr.get('scale_z',1)
                if not np.isscalar(scale):scale=float(scale[0])
                mat[:3,:3] = mat[:3,:3] @ np.diag([scale,scale,sz])
                corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
                world=corners@mat[:3,:3].T+mat[:3,3]
                camera_inside=bool(np.all(world.min(0)<=0) and np.all(world.max(0)>=0))
                if camera_inside or (world.max(0)-world.min(0)).max()>40:
                    b.wire_box(mat,[lo,hi],i,CLASSES[cat])
                else:b.box(mat,[lo,hi],i,CLASSES[cat])
            elif r.get('bounds_min_world') is not None:
                b.box(np.eye(4),[np.asarray(r['bounds_min_world'])-self.origin,
                                 np.asarray(r['bounds_max_world'])-self.origin],i,CLASSES[cat])
        for r in collision.get('collision_segments',[]):
            a=np.asarray(r['a_world_xy']);end=np.asarray(r['b_world_xy'])
            z0,z1=r['z_min'],r['z_max']
            if z1-z0<.01:continue
            ident='collision_wall:'+Path(r.get('source_ybn','unknown').replace('\\','/')).name+':'+str(r.get('geometry_index',''))
            i=self.registry.get(ident,'unknown',static=True,geometry_provenance='source_collision_vertical_extrusion')
            v=np.array([[*a,z0],[*end,z0],[*end,z1],[*a,z1]])-self.origin
            b.add(v,[[0,1,2],[0,2,3]],i,0)
        for r in collision.get('collision_surfaces',[]):
            xy=r.get('points_world_xy',[])
            if len(xy)<3:continue
            z=r.get('z_mean',(r['z_min']+r['z_max'])/2)
            ident='collision_surface:'+Path(r.get('source_ybn','unknown').replace('\\','/')).name+':'+str(r.get('geometry_index',''))
            i=self.registry.get(ident,'unknown',static=True,geometry_provenance='source_collision_mean_height_surface')
            v=np.column_stack([np.asarray(xy),np.full(len(xy),z)])-self.origin
            b.add(v,[[0,j,j+1] for j in range(1,len(xy)-1)],i,0)
        return b.packed()


class RDR2Adapter(Adapter):
    names={'main':'main_oos','monitor':'monitor_npc_b','ego':'ego_npc_a','follow':'follow_npc_a'}
    def __init__(self,case):
        super().__init__(case,'rdr2')
        self.scene=read_json(self.case/'code/scene.json')
        self.site_id=self.scene.get('site_id')
        self.native_sidecars={v:list(rows(self.case/f'code/cameras/{n}.jsonl')) for v,n in self.names.items()}
        self.source_videos={v:self.case/f'videos/{n}.mp4' for v,n in self.names.items()}
        self.origin=np.asarray(self.native_sidecars['main'][0]['position_world_raw'],float)
        self.camera_sources={v:'packaged_final_render_camera' for v in self.views}
        self.warnings += ['Static scene remains a coarse, partial source inventory; unknown geometry is explicit.',
                          'Collector commands and observed transitions are preserved separately; hidden native rules are not claimed.']
        binding=self.case/'code/metadata/binding.json'
        self.static_info=read_json(binding) if binding.exists() else {}
        self.static_info={k:v for k,v in self.static_info.items() if k not in ['video_proofs']}
        self.static_info['normalized_proxy_complete_claim']=False

    def frames(self):
        with np.load(self.case/'code/dynamics.npz',allow_pickle=False) as z:
            xyz=z['world_xyz'].copy();valid=z['joint_valid'].copy();names=list(z['joint_names']);ids=list(z['entity_ids'])
            cycles=z['cycle_id'].copy()
        src=(r for r in rows(self.case/'code/runtime_observations/world_code.jsonl') if r.get('record_type')=='observed_state')
        first_t=int(self.native_sidecars['main'][0]['video_pts'])
        for i,m in enumerate(self.native_sidecars['main']):
            s=next(src)
            if int(s['time']['cycle_id'])!=int(m['cycle_id']) or int(cycles[i])!=int(m['cycle_id']):
                raise ValueError(f'RDR exact state/camera/dynamics mismatch at {i}')
            cams={}
            for v in self.views:
                row=self.native_sidecars[v][i]
                if row['cycle_id']!=m['cycle_id']:raise ValueError(f'RDR view mismatch {v}/{i}')
                width,height=self.scene['source_resolution'];k=np.asarray(row['intrinsics_K'])
                cams[v]=camera_record(row['extrinsics_T_camera_from_world'],perspective(k,width,height),
                                       width,height,self.origin,self.camera_sources[v],intrinsics=k.tolist(),
                                       source_video_frame=int(row['video_frame']),
                                       final_render_camera_verified=row.get('final_render_camera_verified'))
            ents=[]
            for e in s['entities']:
                eid=e['entity_id'];ei=ids.index(eid) if eid in ids else None
                joints={canonical_joint(name):xyz[i,ei,j].tolist() for j,name in enumerate(names)
                        if ei is not None and valid[i,ei,j] and canonical_joint(name) in JOINTS}
                mat=unwrap(e['pose']['T_world_from_entity_raw'])
                state=dict(motion=native_state(e.get('motion',{})),collector_command=e.get('collector_command'),
                           health=unwrap(e.get('health')),native_task=e.get('observed_native_task'))
                ents.append(entity_record(self.registry,eid,'person',mat,self.origin,
                                          velocity=unwrap(e.get('motion',{}).get('velocity_raw')),joints=joints,
                                          state=state,present=bool(unwrap(e.get('valid',True))),
                                          model_id=unwrap(e.get('model_hash')),appearance=unwrap(e.get('model_name')),
                                          is_focus=True,geometry_provenance='native_canonical_joint_proxy'))
            yield dict(index=i,time_us=int(m['video_pts'])-first_t,source_pts_us=int(m['video_pts']),
                       source_frame_id=str(m['cycle_id']),cameras=cams,entities=ents)
        for kind,name in [('command','actions.jsonl'),('observed_transition','transitions.jsonl')]:
            for event in rows(self.case/f'code/{name}'):
                t=event.get('video_pts')
                self.events.append(dict(time_us=int(t)-first_t if t is not None else None,kind=kind,
                                        source_kind='collector_command' if kind=='command' else 'observed_delta',native=event))

    def static_mesh(self):
        path=self.case/'code/layout/static_compiled.npz'
        mapping=read_json(self.case/'code/layout/static_mapping.json')
        lut_id=np.zeros(max(map(int,mapping))+1,np.uint32);lut_sem=np.zeros_like(lut_id)
        for old,meta in mapping.items():
            cat=category(meta.get('category'));old=int(old)
            lut_id[old]=self.registry.get('static:'+str(meta['object_id']),cat,static=True,
                                        geometry_provenance='packaged_coarse_template',source_instance_id=meta.get('source_instance_id'))
            lut_sem[old]=CLASSES[cat]
        with np.load(path,allow_pickle=False) as z:
            ids=z['ids'];result=dict(vertices=(z['v']-self.origin).astype(np.float32),
                                     faces=z['f'].astype(np.uint32),instance_id=lut_id[ids],semantic_id=lut_sem[ids])
        return result


def make_adapter(game,case):
    return {'cp2077':CP2077Adapter,'gta5':GTA5Adapter,'rdr2':RDR2Adapter}[game](case)
