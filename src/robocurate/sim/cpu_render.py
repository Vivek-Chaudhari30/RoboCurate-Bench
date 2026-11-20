"""CPU-only MuJoCo rendering on macOS through Apple's software OpenGL renderer.

Why: tasks forbid GPU use, and dm_control classifies its GLFW/EGL backends as GPU
(`dm_control/_render/__init__.py:109`, USING_GPU = BACKEND in EGL + GLFW). OSMesa is not usable on macOS: MuJoCo's
Python bindings accept only `cgl`/`glfw` there (`mujoco/rendering/classic/gl_context.py:25-34`). Apple ships a
CPU renderer selectable through CGL: kCGLPFARendererID = 70 (SDK CGLTypes.h:69) with kCGLRendererGenericFloatID =
0x00020400, "Apple Software Renderer" (SDK CGLRenderers.h:15). This module creates such a context (no
kCGLPFAAccelerated, unlike mujoco/cgl/__init__.py:44) and installs it as dm_control's render context. The renderer
string is checked after creation and anything other than "Apple Software Renderer" is refused.

Use: set MUJOCO_GL=disable before importing mujoco/dm_control (accepted by both: gl_context.py:25 and
dm_control/_render/constants.py NO_RENDERER), then call `install()` before the first `physics.render`.
"""

from __future__ import annotations

import ctypes
import os
import sys

SOFTWARE_RENDERER = b"Apple Software Renderer"
_ATTRS = (99, 0x1000,       # kCGLPFAOpenGLProfile, kCGLOGLPVersion_Legacy (CGLTypes.h:78)
          70, 0x00020400,   # kCGLPFARendererID, kCGLRendererGenericFloatID (CGLTypes.h:69, CGLRenderers.h:15)
          8, 24, 11, 8,     # kCGLPFAColorSize 24, kCGLPFAAlphaSize 8 (CGLTypes.h:57-58)
          12, 24, 13, 8,    # kCGLPFADepthSize 24, kCGLPFAStencilSize 8 (CGLTypes.h:59-60)
          0)
GL_RENDERER = 0x1F01  # gl.h:1118


class CPURenderError(RuntimeError):
    pass


def _cgl():
    if sys.platform != "darwin":
        raise CPURenderError("the Apple software renderer exists only on macOS")
    lib = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
    lib.CGLChoosePixelFormat.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(ctypes.c_int))
    lib.CGLCreateContext.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    lib.CGLSetCurrentContext.argtypes = (ctypes.c_void_p,)
    lib.CGLReleaseContext.argtypes = (ctypes.c_void_p,)
    lib.CGLReleasePixelFormat.argtypes = (ctypes.c_void_p,)
    lib.glGetString.restype = ctypes.c_char_p
    lib.glGetString.argtypes = (ctypes.c_uint,)
    return lib


def _context_class():
    from dm_control._render import base, executor

    class AppleSoftwareCGLContext(base.ContextBase):
        """dm_control render context on the Apple Software Renderer (CPU rasterization)."""

        renderer = None

        def __init__(self, max_width, max_height):
            super().__init__(max_width, max_height, executor.PassthroughRenderExecutor)

        def _platform_init(self, max_width, max_height):
            self._lib = _cgl()
            attrs = (ctypes.c_int * len(_ATTRS))(*_ATTRS)
            self._pix, n = ctypes.c_void_p(), ctypes.c_int()
            if self._lib.CGLChoosePixelFormat(attrs, ctypes.byref(self._pix), ctypes.byref(n)) or not self._pix.value:
                raise CPURenderError("CGLChoosePixelFormat found no Apple Software Renderer pixel format")
            self._ctx = ctypes.c_void_p()
            if self._lib.CGLCreateContext(self._pix, None, ctypes.byref(self._ctx)):
                raise CPURenderError("CGLCreateContext failed")
            self._lib.CGLSetCurrentContext(self._ctx)
            r = self._lib.glGetString(GL_RENDERER)
            if r != SOFTWARE_RENDERER:
                raise CPURenderError(f"refusing renderer {r!r}: not {SOFTWARE_RENDERER!r} (CPU only)")
            type(self).renderer = r.decode()

        def _platform_make_current(self):
            self._lib.CGLSetCurrentContext(self._ctx)

        def _platform_free(self):
            if getattr(self, "_ctx", None):
                self._lib.CGLSetCurrentContext(None)
                self._lib.CGLReleaseContext(self._ctx)
                self._ctx = None
            if getattr(self, "_pix", None):
                self._lib.CGLReleasePixelFormat(self._pix)
                self._pix = None

    return AppleSoftwareCGLContext


def install() -> type:
    """Install the software context as dm_control's renderer (engine.py:539 looks `_render.Renderer` up per
    call). Requires MUJOCO_GL=disable at import time so no hardware backend was imported."""
    if os.environ.get("MUJOCO_GL", "").lower() not in ("disable", "disabled", "off", "false", "0"):
        raise CPURenderError("set MUJOCO_GL=disable before importing mujoco / dm_control (CPU-only rendering)")
    from dm_control import _render

    cls = _context_class()
    _render.Renderer = cls
    _render.BACKEND = "apple-software-cgl"
    _render.USING_GPU = False
    return cls
