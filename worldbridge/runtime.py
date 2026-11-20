"""Shared replay and a deliberately small explicit program runtime.

The predictive operators below are user-defined rules, not recovered native AI.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np
from .common import read_json,rows


class World:
    def __init__(self,path):
        self.path=Path(path);self.episode=read_json(self.path/'episode.json')
        self.registry=read_json(self.path/'entities.json')['entities']

    @classmethod
    def open(cls,path):return cls(path)

    def frames(self):return rows(self.path/'frames.jsonl.gz')

    def seek(self,index):
        if index<0:raise IndexError(index)
        for frame in self.frames():
            if frame['index']==index:return frame
        raise IndexError(index)


class ProgramWorld:
    def __init__(self,frame):
        self.state=deepcopy(frame)
        self.rules={}

    def execute(self,instruction):
        op=instruction['op']
        if op=='advance':
            dt=float(instruction['dt_s'])
            if dt<0 or not np.isfinite(dt):raise ValueError('Invalid dt_s')
            for e in self.state['entities']:
                if not e['present']:continue
                velocity=e.get('velocity')
                if velocity is None:continue
                displacement=np.asarray(velocity)*dt
                t=np.asarray(e['transform']);t[:3,3]+=displacement;e['transform']=t.tolist()
                # Translation of recorded pose; no new articulation is invented.
                e['joints']={k:(np.asarray(p)+displacement).tolist() for k,p in e.get('joints',{}).items()}
            self.state['time_us']+=round(dt*1e6)
            self.state['index']+=1
            self.state['source_frame_id']='program:'+str(self.state['index'])
            self.state['source_pts_us']=None
            self.state['state_source']='explicit_constant_velocity_program'
        elif op in ('set_velocity','set_state'):
            entity=next(e for e in self.state['entities'] if e['instance_id']==instruction['instance_id'])
            if op=='set_velocity':
                v=np.asarray(instruction['value'],float)
                if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid velocity')
                entity['velocity']=v.tolist()
            else:entity['state'][instruction['key']]=deepcopy(instruction['value'])
        else:raise ValueError('Unsupported program operation: '+str(op))
        return deepcopy(self.state)


def execute_program(world,program):
    """Interpret one common JSON program, with no execution of native Python."""
    state=ProgramWorld(world.seek(int(program.get('initial_state',{}).get('frame',0))))
    for command in program['instructions']:
        if command['op']=='replay':
            begin=command.get('start_frame',0);end=command.get('end_frame',world.episode['frame_count']-1)
            for frame in world.frames():
                if begin<=frame['index']<=end:yield frame
        else:
            result=state.execute(command)
            if command['op']=='advance':yield result
