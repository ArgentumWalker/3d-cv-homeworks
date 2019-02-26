#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_cam_box(aspect_ratio, fov_y, f=20, n=0.01):
    w, h = aspect_ratio, 1
    tan = np.tan(fov_y/2)
    nl, nr = - n * w / h * tan, n * w / h * tan
    nb, nt = - n * tan, n * tan
    fl, fr = - f * w / h * tan, f * w / h * tan
    fb, ft = - f * tan, f * tan
    return np.array([((nl, nb, n), (nl, nt, n)), ((nl, nt, n), (nr, nt, n)), ((nr, nt, n), (nr, nb, n)), ((nr, nb, n), (nl, nb, n)),
                     ((fl, fb, f), (fl, ft, f)), ((fl, ft, f), (fr, ft, f)), ((fr, ft, f), (fr, fb, f)), ((fr, fb, f), (fl, fb, f)),
                     ((nl, nb, n), (fl, fb, f)), ((nl, nt, n), (fl, ft, f)), ((nr, nt, n), (fr, ft, f)), ((nr, nb, n), (fr, fb, f))],
                    dtype=np.float32).reshape((-1,))


def _build_4d_rot_matrix(rot_mat, tr_vec):
    v4d = np.concatenate((rot_mat, np.zeros((1, 3))))
    v4d = np.concatenate((v4d, np.zeros((4, 1))), axis=1)
    v4d[3][3] = 1
    v4d[:3, 3] = tr_vec
    return v4d


def _build_view_matrix(rot_mat, tr_vec):
    rot_mat = np.concatenate((rot_mat.transpose(), np.zeros((1, 3))))
    rot_mat = np.concatenate((rot_mat, np.zeros((4, 1))), axis=1)
    rot_mat[3][3] = 1

    v4d = np.eye(4)
    v4d[:3, 3] = tr_vec
    return rot_mat.dot(v4d)


def _build_projection_matrix(w, h, fov_y, n=0.001, f=100):
    # base_n = h / 2 / tan
    tan = np.tan(fov_y/2)
    l, r = - n * w / h * tan, n * w / h * tan
    b, t = - n * tan, n * tan

    return np.array([
        [n / r, 0, 0, 0],
        [0, n / t, 0, 0],
        [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def _build_colored_program():
    vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position_in;
        in vec3 color_in;
        
        out vec3 color;

        void main() {
            vec4 camera_space_position = mvp * vec4(position_in, 1.0);
            gl_Position = camera_space_position;
            color = color_in;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 140
        
        in vec3 color;
        
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._ctn = len(point_cloud.points)
        self.inv_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.points = vbo.VBO(np.array(point_cloud.points, dtype=np.float32).reshape((-1,)))
        self.colors = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32).reshape((-1,)))
        self.cam_track = tracked_cam_track
        self.track = vbo.VBO(np.array([(tr1.t_vec, tr2.t_vec)
                                       for tr1, tr2 in zip(self.cam_track[1:], self.cam_track)],
                                      dtype=np.float32).reshape((-1,)))
        self.cam_box = vbo.VBO(_build_cam_box(tracked_cam_parameters.aspect_ratio, tracked_cam_parameters.fov_y))

        self._points_shader = _build_colored_program()
        self._track_shader = _build_colored_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        p = _build_projection_matrix(GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH), GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT), camera_fov_y)

        mvp = p.dot(_build_view_matrix(camera_rot_mat, -camera_tr_vec)).dot(self.inv_mat).astype(np.float32)

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._render(mvp, self.points, self._points_shader, self.colors)
        self._render(mvp, self.track, self._points_shader, points=False)
        self._render(mvp.dot(_build_4d_rot_matrix(self.cam_track[tracked_cam_track_pos].r_mat,
                                                  self.cam_track[tracked_cam_track_pos].t_vec)),
                     self.cam_box, self._points_shader, points=False)

        GLUT.glutSwapBuffers()

    def _render(self, mvp, v_buffer, shader, c_buffer=None, points=True):
        ctn = len(v_buffer) // 3

        shaders.glUseProgram(shader)
        v_buffer.bind()
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(shader, 'mvp'),
            1, True, mvp)

        position_loc = GL.glGetAttribLocation(shader, 'position_in')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0, v_buffer)
        if c_buffer is None:
            color_loc = GL.glGetAttribLocation(shader, 'color_in')
            GL.glVertexAttrib3fv(color_loc, np.array([1, 1, 0], dtype=np.float32))

        v_buffer.unbind()

        if c_buffer is not None:
            c_buffer.bind()
            color_loc = GL.glGetAttribLocation(shader, 'color_in')
            GL.glEnableVertexAttribArray(color_loc)
            GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT,
                                 False, 0, c_buffer)

            c_buffer.unbind()

        GL.glDrawArrays(GL.GL_POINTS if points else GL.GL_LINES, 0, ctn)

        GL.glDisableVertexAttribArray(position_loc)
        if c_buffer is not None:
            GL.glDisableVertexAttribArray(color_loc)
        shaders.glUseProgram(0)

