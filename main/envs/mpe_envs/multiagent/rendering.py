"""
2D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version: # 获取python解释程序的版本信息
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym.utils import reraise
from gym import error

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np

RAD2DEG = 57.29577951308232

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    将显示规范(例如:0)转换为实际的显示对象
    Pyglet only supports multiple Displays on Linux.Pyglet只支持Linux上的多种显示。
    """
    if spec is None:
        return None
        # spec与six.string_types是否为同类型，即spec是否为str类型
    elif isinstance(spec, six.string_types):  # six.string_types在python2中，使用的为basestring；在python3中，使用的为str
        return pyglet.canvas.Display(spec)
    else:  # spec不是str类型
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

# 可视化类Viewer
class Viewer(object):
    # # 初始化 输入：width宽度，height高度，显示标志display
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        # pyglet.window.Window(width,height) 创建一个新的窗口
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user  # 定义了窗口关闭的参数
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()   # 获取当前变换的状态
        """
        # glEnable用于启用各种功能。功能由参数决定。
        # glDisable是用来关闭的。两个函数参数取值是一至的。
        glEnable(GL_BLEND)  # GL_BLEND：启用颜色混合。例如实现半透明效果
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)  # GL_LINE_SMOOTH ：	执行后，过虑线段的锯齿
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        # glHint(int target,int mode) 控制GL某些行为: 参数target是要控制的行为  mode是另一个由符号常量描述的想要执行的行为。
        # target=GL_LINE_SMOOTH_HINT——表明直线抗锯齿的效果。
        # mode=GL_NICEST 质量最好
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        # 根据glEnable(GL_LINE_SMOOTH)，结合width=2.0 启用了反走样处理
        glLineWidth(2.0)  # width=2.0 参数表示光栅化的线段的宽度
        # 颜色混合glBlendFunc，在RGBA模式下
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # 源因子sfactor=GL_SRC_ALPHA表示使用源颜色的alpha值来作为因子（alpha）
        # 目标因子dfactor=GL_ONE_MINUS_SRC_ALPHA表示用1.0-源颜色的alpha值来作为因子（1-alpha）
        """
        # 设置渲染画质-抗锯齿，半透明，反走样，颜色混合等
        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    # 设定边界： 左，右，下-底，上-顶
    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom  # 要求 右>左 上>下
        # 均匀刻度数scalex=总宽度/2*cam_range
        # 均匀刻度数scalex=总高度/2*cam_range
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        # 尺度变换参数
        # translation是平移：左-
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    # 追加几何图形
    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    # 渲染显示或数组
    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)  # 设置背景颜色 4个参数为 红，绿，蓝，alpha，不会清除颜色
        self.window.clear()  # 清屏
        self.window.switch_to()  # 切换窗口
        self.window.dispatch_events()  # 用于用于新事件和调用附加事件处理程序的操作系统事件队列
        self.transform.enable()  # 执行尺度变换和颜色变换
        # 渲染每个geom
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()  # 返回保存的做尺度变换的前一状态信息
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()  #
        self.onetime_geoms = []  # 清空
        return arr

    # Convenience  # 绘制函数
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        # 默认颜色为：红=0，绿=0，蓝=0，不透明度=100%
        self.attrs = [self._color]  # 列表形式
    def render(self):
        # reserved() 是 Pyton 内置函数之一，其功能是对于给定的序列（包括列表、元组、字符串以及 range(n) 区间），
        # 该函数可以返回一个逆序序列的迭代器（用于遍历该逆序序列）
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass
# Transform类
class Transform(Attr):
    # 初始定义：translation 平移， rotation 旋转 scale比例-用于放大缩小
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
        # guojing gai
        #self.color = Color((0, 0, 0, 1))  # self.color 是Color类
        # self.set_color(*color)  # 把color的值传递给self.color.vec4
        #self.atrr = [self.color]


    def enable(self):  # 生效
        # guojing gai
        # glColor4f(*self.color.vec4)
        glPushMatrix()  # 入栈
        # glTranslatef(x,y,z)是基于当前位置做平移
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)  # 旋转角度RAD2DEG * self.rotation，绕(0-x,0-y,1-z)轴旋转
        glScalef(self.scale[0], self.scale[1], 1)  # 尺度变换，放大缩小

    def disable(self):  # 失效
        glPopMatrix()  # 出栈
    def set_translation(self, newx, newy):  # 设置平移量
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):  # 设置旋转量
        self.rotation = float(new)
    def set_scale(self, newx, newy):  # 设置尺度变化量
        self.scale = (float(newx), float(newy))
    # guojing
    #def set_color(self,r,g,b,alpha):
    #    self.color.vec4 = (r, g, b, alpha)


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)  # vec4 = red、green、blue、alpha分别是红、绿、蓝、不透明度，

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):  # 点
    def __init__(self):
        Geom.__init__(self)
    def render1(self):  # 渲染点
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):  # 多边形顶点
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

        color = (self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()


# 制作圆
# 函数输入量：半径，微分角度
def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):  # 30等分360度
        ang = 2*math.pi*i / res  # 角度ang 2pi/30 = pi/15等分30个
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
"""guojing changing"""
# 前向扇形
# 函数输入量：半径，角度，微分角度
def make_forward_sector(radius=10, angle_start=-45,angle_end=45 ,res=30, filled=True):
    points = []
    angle = angle_end - angle_start  # 角度
    angle_pi = angle * math.pi/180  # 扇形角度转化弧度
    start_pi = (angle_start+angle_end)/2 * (math.pi/180)  # 起点角度转化弧度
    # 关于x轴对称扇形
    # 角度+
    for i in range(res+1):  # 30等分angle_pi度
        if i>=1:
            ang = 0.5*angle_pi*(res/2 - (i-1)) / (res/2) # +角度
            points.append((math.cos(start_pi + ang) * radius, math.sin(start_pi + ang) * radius))
        else:
            points.append((math.cos(math.pi/2) *radius, math.sin(0) * radius))  # 圆心
        # 绘制+直线
        #if i == res:
        #   make_line((0,0),(math.cos(angle_start+ang)*radius, math.sin(angle_start+ang)*radius))
    # 角度-
    #for i in range(res+1):  # 30等分angle_pi度
    #    if i>=1:
    #        ang = -0.5*angle_pi*(i-1) / res
    #        points.append((math.cos(-angle_start+ang)*radius, math.sin(-angle_start+ang)*radius))
    #    else:
    #        points.append((math.cos(math.pi/2)* radius, math.sin(0)*radius))  #圆心
        # 绘制-直线
        #if i == res:
        #    make_line((0, 0), (math.cos(-angle_start+ang) * radius, math.sin(-angle_start+ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
# 后向扇形
# def make_backward_sector(radius=10, angle_start=0,angle_end=90 ,res=30, filled=True):

# 绘制五角星-无人机
def make_uav(radius=10, res=11, filled=False):
    """
    五角星：  n=[1:2:11];x=sin(0.4*n*pi);y=cos(0.4*n*pi);
    :param radius: 长度
    :param filled:
    :return: 绘制图形
    """
    points = []  # 收集点
    for i in range(res):
        if i%2 == 1:
            ang = 0.4 * math.pi * i;
            points.append((math.sin(ang) * radius, math.cos(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
"""********************************"""
# 渲染多边形
def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

# 渲染直线
def make_polyline(v):
    return PolyLine(v, False)

def make_line(start,end):
    return Line(start,end)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()