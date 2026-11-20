"""Render an explicit local constant-velocity program from one observed state.

No future entity state is read. The observed camera and joint pose are held;
this is a transparent baseline, not a learned physics or animation system.
"""
import argparse,json,gzip
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from worldbridge.runtime import World,ProgramWorld
from worldbridge.wgl import GLRenderer
from worldbridge.geometry import dynamic_mesh
from worldbridge.render import VideoWriter
from worldbridge.common import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frame',type=int,default=0);p.add_argument('--steps',type=int,default=90)
    p.add_argument('--dt',type=float,default=1/30)
    a=p.parse_args()
    if a.steps<1 or a.dt<=0:raise ValueError('Positive steps/dt required')
    a.out.mkdir(parents=True,exist_ok=False)
    world=World.open(a.episode);runtime=ProgramWorld(world.seek(a.frame));gpu=GLRenderer(384,216)
    with np.load(a.episode/'static.npz',allow_pickle=False) as z:gpu.set_static({k:z[k] for k in z.files})
    writer=VideoWriter(a.out/'predicted_white.mp4',384,216,1/a.dt)
    try:
        with gzip.open(a.out/'predicted_states.jsonl.gz','wt',encoding='utf-8') as f:
            for _ in range(a.steps):
                state=runtime.execute({'op':'advance','dt_s':a.dt})
                f.write(json.dumps(state,allow_nan=False)+'\n')
                pixels=gpu.render(state['cameras']['main'],dynamic_mesh(state,world.registry))
                writer.write(pixels['white'])
    finally:writer.close();gpu.close()
    write_json(a.out/'prediction_report.json',dict(initial_frame=a.frame,steps=a.steps,dt_s=a.dt,
        future_entity_states_read=False,initialization='recorded_engine_state_oracle',camera='held_observed_camera',
        dynamics='explicit_constant_velocity',articulation='held_observed_pose_translated_with_root'))


if __name__=='__main__':main()
