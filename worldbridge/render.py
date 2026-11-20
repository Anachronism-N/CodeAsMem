from __future__ import annotations
import json
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import read_json, rows, write_json
from .geometry import dynamic_mesh
from .wgl import GLRenderer


class VideoWriter:
    def __init__(self,path,width,height,fps):
        exe=shutil.which('ffmpeg')
        if not exe:raise RuntimeError('ffmpeg is required on PATH')
        self.path=Path(path);self.log=self.path.with_suffix('.encoder.log').open('wb')
        self.proc=subprocess.Popen([exe,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate',f'{fps:.9f}',
             '-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','18',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
        self.closed=False

    def write(self,image):
        self.proc.stdin.write(np.ascontiguousarray(image,np.uint8).tobytes())

    def close(self):
        if self.closed:return
        self.closed=True
        self.proc.stdin.close();rc=self.proc.wait();self.log.close()
        if rc:raise RuntimeError(f'ffmpeg failed ({rc}): {self.path}')


def probe(path):
    exe=shutil.which('ffprobe')
    r=subprocess.run([exe,'-v','error','-select_streams','v:0','-show_entries',
                       'stream=nb_frames,width,height,avg_frame_rate','-of','json',str(path)],
                      capture_output=True,text=True,timeout=30)
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)['streams'][0]


def render_episode(episode,view='main',width=384,height=216,chunk_frames=32,limit=None,renderer=None):
    episode=Path(episode).resolve();meta=read_json(episode/'episode.json')
    out=episode/'proxy'/view
    if (out/'render_report.json').exists() and limit is None:
        report=read_json(out/'render_report.json')
        if report.get('render_version')==2 and report.get('complete_episode') and report.get('resolution')==[width,height]:
            return dict(game=meta['game'],case_id=meta['case_id'],view=view,status='existing',frames=report['frames'])
    out.mkdir(parents=True,exist_ok=True);buffers=out/'buffers';buffers.mkdir(exist_ok=True)
    registry=read_json(episode/'entities.json')['entities']
    timeline=np.load(episode/'timeline.npz',allow_pickle=False)['time_us']
    count=min(len(timeline),limit) if limit else len(timeline)
    fps=(count-1)/((timeline[count-1]-timeline[0])/1e6) if count>1 else 30.
    own=renderer is None;gpu=renderer or GLRenderer(width,height)
    if (gpu.width,gpu.height)!=(width,height):raise ValueError('Renderer resolution differs')
    start=time.perf_counter();writers=[];records=[];chunks=[];chunk={k:[] for k in ['depth','instance','semantic','covered','actor_depth','actor_instance','actor_semantic','time_us','source_frame']}
    snapshots={0,count//2,count-1};images={};render_times=[];root_fov=[]
    try:
        with np.load(episode/'static.npz',allow_pickle=False) as z:gpu.set_static({k:z[k] for k in z.files})
        writers=[VideoWriter(out/f'{name}.mp4',width,height,fps) for name in ['white','semantic','actors','control']]
        def flush():
            if not chunk['time_us']:return
            name=f'chunk_{len(chunks):05d}.npz';n=len(chunk['time_us']);begin=len(records)-n
            arrays={k:np.stack(v) for k,v in chunk.items()}
            np.savez_compressed(buffers/name,**arrays)
            chunks.append(dict(path='buffers/'+name,start_frame=begin,frames=n))
            for v in chunk.values():v.clear()
        for frame in rows(episode/'frames.jsonl.gz'):
            idx=frame['index']
            if idx>=count:break
            cam=frame['cameras'][view]
            hide=[next((r['instance_id'] for r in registry if r.get('is_focus')),0)] if view=='ego' else []
            mesh=dynamic_mesh(frame,registry,hide=hide)
            tick=time.perf_counter();result=gpu.render(cam,mesh)
            actors=gpu.render(cam,mesh,include_static=False)
            control=np.where(actors['covered'][...,None],actors['white'],result['white'])
            render_times.append(time.perf_counter()-tick)
            writers[0].write(result['white']);writers[1].write(result['semantic_rgb'])
            writers[2].write(actors['white']);writers[3].write(control)
            for key in ['depth','instance','semantic','covered']:chunk[key].append(result[key])
            for key in ['depth','instance','semantic']:chunk['actor_'+key].append(actors[key])
            chunk['time_us'].append(np.int64(frame['time_us']));chunk['source_frame'].append(np.int32(cam['source_video_frame']))
            visible_ids,px=np.unique(result['instance'],return_counts=True)
            records.append(dict(frame=idx,time_us=frame['time_us'],source_video_frame=cam['source_video_frame'],
                                covered_pixels=int(result['covered'].sum()),
                                actor_only_pixels=int(actors['covered'].sum()),
                                actor_pixels_occluded_by_proxy_geometry=int((actors['covered'] & (actors['instance']!=result['instance'])).sum()),
                                rendered_instance_pixels={str(int(i)):int(n) for i,n in zip(visible_ids,px) if i}))
            if idx in snapshots:
                images[idx]=(result['white'],control)
                Image.fromarray(result['white']).save(out/f'white_{idx:04d}.png')
            if len(chunk['time_us'])>=chunk_frames:flush()
        flush()
        for w in writers:w.close()
        if len(records)!=count:raise ValueError(f'Render count mismatch {len(records)} != {count}')
        encoded={name:probe(out/f'{name}.mp4') for name in ['white','semantic','actors','control']}
        if any(int(v['nb_frames'])!=count for v in encoded.values()):raise ValueError('Encoded frame count mismatch')
        report=dict(schema='worldbridge.proxy.v1',render_version=2,game=meta['game'],case_id=meta['case_id'],view=view,
                    frames=count,complete_episode=count==meta['frame_count'],resolution=[width,height],
                    preview_fps=fps,video_time_mode='CFR_preview; exact_native_time_us_in_buffers',
                    exact_frame_index_alignment=True,source_video=meta['views'][view]['source_video'],
                    source_rgb_used_for_geometry=False,renderer=gpu.info,
                    channels=dict(white='white.mp4',semantic_display='semantic.mp4',depth='float32_meters',
                                  actor_only='actors.mp4',dynamic_priority_control='control.mp4',
                                  actor_depth='float32_meters_ignores_static_proxy_occlusion',
                                  instance='uint32_0_is_uncovered',semantic='uint16_0_is_unknown',
                                  covered='proxy_geometry_coverage_not_native_visibility'),
                    background_depth='positive_infinity',chunks=chunks,encoded=encoded,
                    control_occlusion_policy='dynamic layer composited over coarse static geometry; not physical pixel visibility',
                    geometric_depth_policy='depth/instance retain full z-buffer; actor_depth/actor_instance retain separate dynamic layer',
                    total_seconds=time.perf_counter()-start,
                    mean_raster_seconds=float(np.mean(render_times)),p95_raster_seconds=float(np.percentile(render_times,95)),
                    source_quality_status='pending_visual_review',dense_surface_ground_truth=False)
        with (out/'frames.jsonl').open('w',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row)+'\n')
        make_preview(meta,view,images,out,width,height)
        write_json(out/'render_report.json',report)
        return dict(game=meta['game'],case_id=meta['case_id'],view=view,status='rendered',frames=count,
                    seconds=report['total_seconds'],mean_raster_seconds=report['mean_raster_seconds'])
    finally:
        for w in writers:
            if not w.closed:
                try:w.close()
                except Exception:pass
        if own:gpu.close()


def make_preview(meta,view,images,out,width,height):
    import cv2
    canvas=Image.new('RGB',(width*3,(height+28)*len(images)),(22,26,34));d=ImageDraw.Draw(canvas)
    source=meta['views'][view]['source_video'];cap=cv2.VideoCapture(source)
    try:
        for row,(idx,(white,control)) in enumerate(sorted(images.items())):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,bgr=cap.read()
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB) if ok else np.zeros((height,width,3),np.uint8)
            for col,(title,image) in enumerate([('Source RGB',rgb),('Geometric white',white),('Dynamic-priority control',control)]):
                canvas.paste(Image.fromarray(image).resize((width,height)),(col*width,row*(height+28)+28))
                d.text((col*width+5,row*(height+28)+5),f'{meta["game"]} {title} frame={idx}',fill='white')
        canvas.save(out/'preview.jpg',quality=90)
    finally:cap.release()
