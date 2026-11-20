import numpy as np
import pytest
import sys
from worldbridge.common import camera_record,perspective,project,transform,category
from worldbridge.adapters import native_state
from worldbridge.geometry import MeshBuilder
from worldbridge.runtime import ProgramWorld


def test_world_origin_does_not_change_projection():
    origin=np.array([1370.,-920.,23.])
    v=transform(-origin);k=np.array([[200.,0,160],[0,200.,90],[0,0,1]])
    cam=camera_record(v,perspective(k,320,180),320,180,origin,'test')
    uv,inside=project(cam,[[0,0,5],[1,1,5],[0,0,-1]])
    np.testing.assert_allclose(uv[:2],[[160,90],[200,50]],atol=1e-9)
    assert inside.tolist()==[True,True,False]


def test_missing_state_is_distinct_from_false():
    v=native_state({'isWalking':None,'IsDead':False,'isRunning':{'value':True}})
    assert v['walking'] is None and v['dead'] is False and v['running'] is True


def test_street_furniture_is_not_a_tree():
    assert category('street_furniture')=='prop'
    assert category('tree')=='tree'


def test_program_moves_offscreen_entities_and_keeps_unknown_velocity():
    frame={'index':0,'time_us':0,'entities':[
        {'instance_id':1,'present':True,'transform':np.eye(4).tolist(),'velocity':[2,0,0],
         'joints':{'SKEL_Head':[0,0,1.7]},'state':{},'in_frustum':False},
        {'instance_id':2,'present':True,'transform':np.eye(4).tolist(),'velocity':None,'state':{}}]}
    runtime=ProgramWorld(frame)
    state=runtime.execute({'op':'advance','dt_s':3})
    assert state['entities'][0]['transform'][0][3]==6
    assert state['entities'][0]['joints']['SKEL_Head']==[6,0,1.7]
    assert state['entities'][1]['transform'][0][3]==0
    assert frame['entities'][0]['transform'][0][3]==0


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != 'win32', reason='Native WGL renderer requires Windows')
def test_opengl_depth_and_ids_resolve_actual_occlusion():
    from worldbridge.wgl import GLRenderer
    gpu=GLRenderer(64,64)
    try:
        back=MeshBuilder();back.box(transform([0,0,8]),[[-2,-2,-.5],[2,2,.5]],1,3)
        front=MeshBuilder();front.box(transform([0,0,4]),[[-.5,-.5,-.5],[.5,.5,.5]],2,1)
        gpu.set_static(back.packed())
        k=np.array([[40.,0,32],[0,40.,32],[0,0,1]])
        camera=camera_record(np.eye(4),perspective(k,64,64),64,64,np.zeros(3),'test')
        result=gpu.render(camera,front.packed())
        assert result['instance'][32,32]==2 and result['semantic'][32,32]==1
        assert abs(float(result['depth'][32,32])-3.5)<.001
        assert not result['covered'][0,0] and np.isinf(result['depth'][0,0])
        assert result['instance'].dtype==np.uint32
    finally:gpu.close()
