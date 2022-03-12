import math
import time

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from strucs import *


def main():
    pygame.init()
    display = 800, 600
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL,vsync=1)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 800.0 / 600.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    ice = Iceberg()
    ice.state.pos = Vector(0, 8, 0)

    ice.context.mg = Vector(0, ice.context.mass * (-9.8), 0)
    ice.state.impulse = Vector(0, -10, 0)

    alpha = 3.14 * 45 / 180
    arr = [1, 0, 0, 0, math.cos(alpha), -math.sin(alpha), 0, math.sin(alpha), math.cos(alpha)]

    ice.state.R = Matrix(arr)
    ice.state.am = Vector(3, 0, 2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        glTranslatef(0, -5, -20)
        glRotatef(20, 1, 0, 0)

        glColor3f(0, 0.65, 0.65)

        '''water'''
        glBegin(GL_QUADS)
        glVertex3f(-10, 0, -10)
        glVertex3f(10, 0, -10)
        glVertex3f(10, 0, 10)
        glVertex3f(-10, 0, 10)
        glEnd()

        glPushMatrix()

        ice.state.am *= 0.9999
        ice.state.impulse *= 0.99
        ice = ice.rk4(ice, 0.016)
        ice.draw()

        glPopMatrix()
        glPopMatrix()
        pygame.display.flip()


if __name__ == "__main__":
    main()