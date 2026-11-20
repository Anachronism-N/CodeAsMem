"""Native Windows OpenGL 4.3 MRT rasterizer; no downloaded GL packages.

The window is never shown. Geometry comes only from normalized scene records.
"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import os
import numpy as np


class PFD(c.Structure):
    _fields_=[('nSize',w.WORD),('nVersion',w.WORD),('dwFlags',w.DWORD),
        ('iPixelType',c.c_byte),('cColorBits',c.c_byte),('cRedBits',c.c_byte),
        ('cRedShift',c.c_byte),('cGreenBits',c.c_byte),('cGreenShift',c.c_byte),
        ('cBlueBits',c.c_byte),('cBlueShift',c.c_byte),('cAlphaBits',c.c_byte),
        ('cAlphaShift',c.c_byte),('cAccumBits',c.c_byte),('cAccumRedBits',c.c_byte),
        ('cAccumGreenBits',c.c_byte),('cAccumBlueBits',c.c_byte),('cAccumAlphaBits',c.c_byte),
        ('cDepthBits',c.c_byte),('cStencilBits',c.c_byte),('cAuxBuffers',c.c_byte),
        ('iLayerType',c.c_byte),('bReserved',c.c_byte),('dwLayerMask',w.DWORD),
        ('dwVisibleMask',w.DWORD),('dwDamageMask',w.DWORD)]


class Context:
    def __init__(self,width,height):
        if os.name!='nt':raise RuntimeError('The dependency-free WGL backend requires Windows.')
        self.user=c.WinDLL('user32');self.gdi=c.WinDLL('gdi32');self.lib=c.WinDLL('opengl32')
        u,g,l=self.user,self.gdi,self.lib
        u.CreateWindowExW.restype=w.HWND
        u.CreateWindowExW.argtypes=[w.DWORD,w.LPCWSTR,w.LPCWSTR,w.DWORD,c.c_int,c.c_int,c.c_int,c.c_int,w.HWND,w.HMENU,w.HINSTANCE,c.c_void_p]
        u.GetDC.argtypes=[w.HWND];u.GetDC.restype=w.HDC
        g.ChoosePixelFormat.argtypes=[w.HDC,c.POINTER(PFD)];g.SetPixelFormat.argtypes=[w.HDC,c.c_int,c.POINTER(PFD)]
        l.wglCreateContext.argtypes=[w.HDC];l.wglCreateContext.restype=c.c_void_p
        l.wglMakeCurrent.argtypes=[w.HDC,c.c_void_p]
        l.wglGetProcAddress.argtypes=[c.c_char_p];l.wglGetProcAddress.restype=c.c_void_p
        self.hwnd=u.CreateWindowExW(0,'STATIC','WorldBridge offline hidden rasterizer',0x80000000,0,0,width,height,None,None,None,None)
        if not self.hwnd:raise OSError('Cannot create hidden render window')
        self.dc=u.GetDC(self.hwnd);p=PFD();p.nSize=c.sizeof(p);p.nVersion=1
        p.dwFlags=0x25;p.cColorBits=24;p.cAlphaBits=8;p.cDepthBits=32
        fmt=g.ChoosePixelFormat(self.dc,c.byref(p))
        if not fmt or not g.SetPixelFormat(self.dc,fmt,c.byref(p)):raise OSError('Cannot set OpenGL pixel format')
        self.rc=l.wglCreateContext(self.dc)
        if not self.rc or not l.wglMakeCurrent(self.dc,self.rc):raise OSError('Cannot create OpenGL context')
        self._functions=[]

    def function(self,name,restype,*args):
        try:fn=getattr(self.lib,name)
        except AttributeError:
            address=self.lib.wglGetProcAddress(name.encode())
            if address in (None,0,1,2,3,c.c_void_p(-1).value):raise RuntimeError('OpenGL function unavailable: '+name)
            fn=c.WINFUNCTYPE(restype,*args)(address)
        fn.restype=restype;fn.argtypes=list(args);self._functions.append(fn)
        return fn

    def close(self):
        if not getattr(self,'rc',None):return
        self.lib.wglMakeCurrent(self.dc,None)
        self.lib.wglDeleteContext.argtypes=[c.c_void_p];self.lib.wglDeleteContext(self.rc)
        self.user.ReleaseDC.argtypes=[w.HWND,w.HDC];self.user.ReleaseDC(self.hwnd,self.dc)
        self.user.DestroyWindow.argtypes=[w.HWND];self.user.DestroyWindow(self.hwnd)
        self.rc=None


VERTEX='''#version 430
layout(location=0) in vec3 position;
uniform mat4 vp;
out vec3 world_position;
void main(){world_position=position;gl_Position=vp*vec4(position,1.);}
'''
FRAGMENT='''#version 430
layout(std430,binding=0) readonly buffer Faces { uvec2 face[]; };
in vec3 world_position;
layout(location=0) out vec4 white_color;
layout(location=1) out uint instance_id;
layout(location=2) out uint semantic_id;
layout(location=3) out vec4 semantic_color;
const vec3 palette[12]=vec3[12](vec3(110,85,120),vec3(233,137,75),vec3(77,155,209),
 vec3(190,181,157),vec3(87,145,100),vec3(135,171,97),vec3(145,147,151),
 vec3(172,161,136),vec3(153,160,174),vec3(194,151,182),vec3(210,185,103),vec3(158,111,75));
void main(){
 uvec2 d=face[gl_PrimitiveID];instance_id=d.x;semantic_id=d.y;
 vec3 n=normalize(cross(dFdx(world_position),dFdy(world_position)));
 float shade=.58+.42*abs(dot(n,normalize(vec3(.35,-.5,.79))));
 white_color=vec4(vec3(.9*shade),1.);
 semantic_color=vec4(palette[min(d.y,11u)]/255.*shade,1.);
}
'''


class GLRenderer:
    def __init__(self,width=384,height=216):
        self.width,self.height=width,height;self.context=Context(width,height);f=self.context.function
        U,I,F,P=c.c_uint,c.c_int,c.c_float,c.c_void_p
        self.genvao=f('glGenVertexArrays',None,I,c.POINTER(U));self.bindvao=f('glBindVertexArray',None,U)
        self.delvao=f('glDeleteVertexArrays',None,I,c.POINTER(U))
        self.genbuf=f('glGenBuffers',None,I,c.POINTER(U));self.bindbuf=f('glBindBuffer',None,U,U)
        self.bufdata=f('glBufferData',None,U,c.c_ssize_t,P,U);self.delbuf=f('glDeleteBuffers',None,I,c.POINTER(U))
        self.bindbase=f('glBindBufferBase',None,U,U,U)
        self.attrib=f('glVertexAttribPointer',None,U,I,U,c.c_ubyte,I,P);self.enableattr=f('glEnableVertexAttribArray',None,U)
        self.shadercreate=f('glCreateShader',U,U);self.shadersource=f('glShaderSource',None,U,I,c.POINTER(c.c_char_p),c.POINTER(I))
        self.shadercompile=f('glCompileShader',None,U);self.shaderget=f('glGetShaderiv',None,U,U,c.POINTER(I))
        self.shaderlog=f('glGetShaderInfoLog',None,U,I,c.POINTER(I),P);self.shaderdelete=f('glDeleteShader',None,U)
        self.programcreate=f('glCreateProgram',U);self.attach=f('glAttachShader',None,U,U)
        self.link=f('glLinkProgram',None,U);self.programget=f('glGetProgramiv',None,U,U,c.POINTER(I))
        self.programlog=f('glGetProgramInfoLog',None,U,I,c.POINTER(I),P);self.programdelete=f('glDeleteProgram',None,U)
        self.use=f('glUseProgram',None,U);self.uniformloc=f('glGetUniformLocation',I,U,c.c_char_p)
        self.uniformmatrix=f('glUniformMatrix4fv',None,I,I,c.c_ubyte,P)
        self.gentex=f('glGenTextures',None,I,c.POINTER(U));self.bindtex=f('glBindTexture',None,U,U)
        self.teximage=f('glTexImage2D',None,U,I,I,I,I,I,U,U,P);self.texparam=f('glTexParameteri',None,U,U,I)
        self.deltex=f('glDeleteTextures',None,I,c.POINTER(U));self.genfbo=f('glGenFramebuffers',None,I,c.POINTER(U))
        self.bindfbo=f('glBindFramebuffer',None,U,U);self.delfbo=f('glDeleteFramebuffers',None,I,c.POINTER(U))
        self.attachtex=f('glFramebufferTexture2D',None,U,U,U,U,I);self.checkfbo=f('glCheckFramebufferStatus',U,U)
        self.drawbuffers=f('glDrawBuffers',None,I,c.POINTER(U));self.viewport=f('glViewport',None,I,I,I,I)
        self.clearf=f('glClearBufferfv',None,U,I,P);self.clearu=f('glClearBufferuiv',None,U,I,P)
        self.enable=f('glEnable',None,U);self.disable=f('glDisable',None,U);self.depthfunc=f('glDepthFunc',None,U)
        self.readbuffer=f('glReadBuffer',None,U);self.readpixels=f('glReadPixels',None,I,I,I,I,U,U,P)
        self.pixelstore=f('glPixelStorei',None,U,I);self.draw=f('glDrawElements',None,U,I,U,P)
        getstring=f('glGetString',c.c_char_p,U)
        self.info={k:getstring(v).decode() for k,v in [('vendor',0x1F00),('renderer',0x1F01),('version',0x1F02)]}
        self.program=self.programcreate()
        for kind,source in [(0x8B31,VERTEX),(0x8B30,FRAGMENT)]:
            sh=self.shadercreate(kind);s=c.c_char_p(source.encode());self.shadersource(sh,1,c.byref(s),None)
            self.shadercompile(sh);ok=I();self.shaderget(sh,0x8B81,c.byref(ok))
            if not ok.value:
                msg=c.create_string_buffer(8192);self.shaderlog(sh,8192,None,msg);raise RuntimeError(msg.value.decode())
            self.attach(self.program,sh);self.shaderdelete(sh)
        self.link(self.program);ok=I();self.programget(self.program,0x8B82,c.byref(ok))
        if not ok.value:
            msg=c.create_string_buffer(8192);self.programlog(self.program,8192,None,msg);raise RuntimeError(msg.value.decode())
        self.vp_location=self.uniformloc(self.program,b'vp')
        self.textures=[];self.fbo=U();self.genfbo(1,c.byref(self.fbo));self.bindfbo(0x8D40,self.fbo)
        for slot,internal,external,typ in [(0,0x8058,0x1908,0x1401),(1,0x8236,0x8D94,0x1405),
                                           (2,0x8234,0x8D94,0x1403),(3,0x8058,0x1908,0x1401),
                                           (4,0x8CAC,0x1902,0x1406)]:
            texture=U();self.gentex(1,c.byref(texture));self.textures.append(texture)
            self.bindtex(0x0DE1,texture);self.teximage(0x0DE1,0,internal,width,height,0,external,typ,None)
            self.texparam(0x0DE1,0x2801,0x2600);self.texparam(0x0DE1,0x2800,0x2600)
            self.attachtex(0x8D40,0x8CE0+slot if slot<4 else 0x8D00,0x0DE1,texture,0)
        attachments=(U*4)(*[0x8CE0+i for i in range(4)]);self.drawbuffers(4,attachments)
        if self.checkfbo(0x8D40)!=0x8CD5:raise RuntimeError('Incomplete MRT framebuffer')
        self.enable(0x0B71);self.depthfunc(0x0201);self.disable(0x0B44);self.disable(0x0BD0)
        self.pixelstore(0x0D05,1);self.static=None;self.dynamic=self.mesh()
        self._clear=(F*4)(.10,.12,.15,0);self._zeros=(U*4)(0,0,0,0);self._depth=(F*1)(1.)

    def mesh(self):
        vao=c.c_uint();self.genvao(1,c.byref(vao));self.bindvao(vao)
        bufs=(c.c_uint*3)();self.genbuf(3,bufs)
        self.bindbuf(0x8892,bufs[0]);self.attrib(0,3,0x1406,0,12,None);self.enableattr(0)
        self.bindbuf(0x8893,bufs[1])
        return {'vao':vao,'buffers':bufs,'count':0}

    def update(self,mesh,data):
        self.bindvao(mesh['vao'])
        desc=np.column_stack([data['instance_id'],data['semantic_id']]).astype(np.uint32)
        arrays=[np.ascontiguousarray(data['vertices'],np.float32),np.ascontiguousarray(data['faces'],np.uint32),np.ascontiguousarray(desc)]
        for target,buffer,array in zip([0x8892,0x8893,0x90D2],mesh['buffers'],arrays):
            self.bindbuf(target,buffer);self.bufdata(target,array.nbytes,c.c_void_p(array.ctypes.data),0x88E4)
        mesh['count']=len(data['faces'])*3

    def release_mesh(self,mesh):
        if mesh is not None:self.delvao(1,c.byref(mesh['vao']));self.delbuf(3,mesh['buffers'])

    def set_static(self,data):
        self.release_mesh(self.static);self.static=self.mesh();self.update(self.static,data)

    def draw_mesh(self,mesh):
        if mesh is None or not mesh['count']:return
        self.bindvao(mesh['vao']);self.bindbase(0x90D2,0,mesh['buffers'][2])
        self.draw(0x0004,mesh['count'],0x1405,None)

    def render(self,camera,dynamic,include_static=True):
        self.bindfbo(0x8D40,self.fbo);self.viewport(0,0,self.width,self.height)
        for slot in [0,3]:self.clearf(0x1800,slot,self._clear)
        for slot in [1,2]:self.clearu(0x1800,slot,self._zeros)
        self.clearf(0x1801,0,self._depth);self.use(self.program)
        vp=np.ascontiguousarray((np.asarray(camera['projection'])@np.asarray(camera['view'])).T,np.float32)
        self.uniformmatrix(self.vp_location,1,0,c.c_void_p(vp.ctypes.data))
        if include_static:self.draw_mesh(self.static)
        self.update(self.dynamic,dynamic);self.draw_mesh(self.dynamic)
        result={}
        for slot,key,dtype,channels,format_,type_ in [(0,'white',np.uint8,4,0x1908,0x1401),
                 (1,'instance',np.uint32,1,0x8D94,0x1405),(2,'semantic',np.uint16,1,0x8D94,0x1403),
                 (3,'semantic_rgb',np.uint8,4,0x1908,0x1401)]:
            a=np.empty((self.height,self.width,channels),dtype=dtype)
            self.readbuffer(0x8CE0+slot);self.readpixels(0,0,self.width,self.height,format_,type_,c.c_void_p(a.ctypes.data))
            result[key]=a[::-1,:,:3].copy() if channels==4 else a[::-1,:,0].copy()
        d=np.empty((self.height,self.width),np.float32)
        self.readpixels(0,0,self.width,self.height,0x1902,0x1406,c.c_void_p(d.ctypes.data))
        p=np.asarray(camera['projection']);ndc=2*d[::-1]-1
        with np.errstate(divide='ignore',invalid='ignore'):depth=p[2,3]/(ndc-p[2,2])
        depth[result['instance']==0]=np.inf
        result['depth']=depth.astype(np.float32);result['covered']=result['instance']>0
        return result

    def close(self):
        self.release_mesh(self.static);self.release_mesh(self.dynamic)
        self.programdelete(self.program);self.delfbo(1,c.byref(self.fbo))
        for tex in self.textures:self.deltex(1,c.byref(tex))
        self.context.close()
