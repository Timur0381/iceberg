import math
from OpenGL.GL import *
import copy


class Vector:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.z)

    @staticmethod
    def dot(l, r):
        return l.x * r.x + l.y * r.y + l.z * r.z

    @staticmethod
    def cross(l, r):
        res = Vector(l.y * r.z - l.z * r.y, l.z * r.x - l.x * r.z, l.x * r.y - l.y * r.x)
        return res

    def __truediv__(self, rv):
        if rv == 0:
            return Vector(float("nan"), float("nan"), float("nan"))
        return Vector(self.x / rv, self.y / rv, self.z / rv)

    def normalize(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        if length == 0:
            return Vector(float("nan"), float("nan"), float("nan"))
        return self / length

    def is_null(self):
        if self.x == 0 and self.y == 0 and self.z == 0:
            return True
        return False

    def __add__(self, rv):
        res = Vector(self.x + rv.x, self.y + rv.y, self.z + rv.z)
        return res

    def __sub__(self, rv):
        res = Vector(self.x - rv.x, self.y - rv.y, self.z - rv.z)
        return res

    def __mul__(self, rv):
        res = Vector(self.x * rv, self.y * rv, self.z * rv)
        return res

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + + self.z ** 2)

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        return -1

    def __getitem__(self, ind):
        if ind == 0:
            return self.x
        elif ind == 1:
            return self.y
        elif ind == 2:
            return self.z


class Matrix:
    def __init__(self, arr=[]):

        self.arr = copy.copy(arr)
        if len(arr) == 0:
            for i in range(9):
                self.arr.append(0)

    def set(self, e, x, y):
        self.arr[x + 3 * y] = e

    def __str__(self):
        s = ''
        for i in range(3):
            for j in range(3):
                s += str(round(self.get(i, j), 2)) + " "
            s += "\n"

        return s

    def t(self):
        c = Matrix()
        for i in range(3):
            for j in range(3):
                c.set(self.get(i, j), j, i)
        return c

    def get(self, x, y):
        return self.arr[x + 3 * y]

    def func_equal(self, m):
        self.arr = m.arr
        return self.arr

    def __mul__(self, m):
        if type(m) == Matrix:
            c = Matrix()
            for i in range(3):
                for j in range(3):
                    sum = 0
                    for k in range(3):
                        sum += self.get(i, k) * m.get(k, j)
                    c.set(sum, i, j)
            return c
        elif type(m) == Vector:
            c = Vector()
            for i in range(3):
                s = 0
                for k in range(3):
                    s += self.get(i, k) * m[k]
                c[i] = s
            return c
        elif type(m) == float or type(m) == int:
            t = []
            for i in range(9):
                t.append(self.arr[i] * m)
            return Matrix(t)

    def __add__(self, other):
        t = []
        for i in range(9):
            t.append(self.arr[i] + other.arr[i])
        return Matrix(t)

    @staticmethod
    def cross_matrix(v):
        t = [0, v.z, -v.y, -v.z, 0, v.x, v.y, -v.x, 0]
        return Matrix(t)

    def adj(self, x, y):
        t = [0, 0, 0, 0]
        index = 0
        for j in range(3):
            for i in range(3):
                if j != y and i != x:
                    t[index] = self.get(i, j)
                    index += 1
        det = t[0] * t[3] - t[1] * t[2]
        return (1 - ((x + y) % 2) * 2) * det

    def det(self):
        return self.get(0, 0) * self.adj(0, 0) + self.get(1, 0) * self.adj(1, 0) + self.get(2, 0) * self.adj(2, 0)

    @staticmethod
    def inv(m):
        d = m.det()
        t = []
        for j in range(9):
            t.append(m.adj(j % 3, j // 3))
        m1 = Matrix(t)
        tmp = m1.t()
        rt = tmp * (1 / d)
        return rt

    @staticmethod
    def ort_gram_schmidt(a):
        f1 = Vector(a.get(0, 0), a.get(1, 0), a.get(2, 0))
        f2 = Vector(a.get(0, 1), a.get(1, 1), a.get(2, 1))
        f3 = Vector(a.get(0, 2), a.get(1, 2), a.get(2, 2))

        e1 = f1
        if not e1.is_null():
            g12 = Vector.dot(f2, e1) / Vector.dot(e1, e1)
        else:
            g12 = 0

        e2 = f2 - e1 * g12

        if not e1.is_null():
            g13 = Vector.dot(f3, e1) / Vector.dot(e1, e1)
        else:
            g13 = 0

        if not e2.is_null():
            g23 = Vector.dot(f3, e2) / Vector.dot(e2, e2)
        else:
            g23 = 0

        e3 = f3 - e1 * g13 - e2 * g23

        if not e1.is_null():
            e1 /= math.sqrt(Vector.dot(e1, e1))
        if not e2.is_null():
            e2 /= math.sqrt(Vector.dot(e2, e2))
        if not e3.is_null():
            e3 /= math.sqrt(Vector.dot(e3, e3))

        m = Matrix()
        m.set(e1.x, 0, 0)
        m.set(e2.x, 0, 1)
        m.set(e3.x, 0, 2)

        m.set(e1.y, 1, 0)
        m.set(e2.y, 1, 1)
        m.set(e3.y, 1, 2)

        m.set(e1.z, 2, 0)
        m.set(e2.z, 2, 1)
        m.set(e3.z, 2, 2)
        return m


class Section:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def crossing(self):
        if self.a.x == self.b.x:
            ax = (self.a.y - self.b.y)
        else:
            ax = (self.a.y - self.b.y) / (self.a.x - self.b.x)
        if self.a.z == self.b.z:
            az = self.a.y - self.b.y
        else:
            az = (self.a.y - self.b.y) / (self.a.z - self.b.z)

        if ax != 0:
            x = self.a.x - self.a.y / ax
        else:
            x = self.a.x - self.a.y
        if az != 0:
            z = self.a.z - self.a.y / az
        else:
            z = self.a.z - self.a.y
        return Vector(x, 0, z)


class Face:
    def __init__(self):
        self.num_verts = 0
        self.norm = Vector()
        self.w = 0
        self.vertices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.polyhedron = 0


class Polyhedron:
    def __init__(self):
        self.num_verts = 0
        self.num_faces = 0
        self.vertices = []
        for i in range(100):
            self.vertices.append(Vector(float("nan"), float("nan"), float("nan")))
        self.faces = []
        for i in range(100):
            self.faces.append(Face())


class Volume_integral:
    def __init__(self):
        self.A = 0
        self.B = 0
        self.C = 0
        self.P1 = 0
        self.Pa = 0
        self.Pb = 0
        self.Paa = 0
        self.Pab = 0
        self.Pbb = 0
        self.Paaa = 0
        self.Paab = 0
        self.Pabb = 0
        self.Pbbb = 0
        self.Fa = 0
        self.Fb = 0
        self.Fc = 0
        self.Faa = 0
        self.Fbb = 0
        self.Fcc = 0
        self.Faaa = 0
        self.Fbbb = 0
        self.Fccc = 0
        self.Faab = 0
        self.Fbbc = 0
        self.Fcca = 0
        self.T0 = 0
        self.T1 = [0, 0, 0]
        self.T2 = [0, 0, 0]
        self.TP = [0, 0, 0]

    def read_polyhedron(self, name, p: Polyhedron):
        try:
            with open(name, "r") as file:
                lines = [str.strip() for str in file.readlines()]
            file.close()

        except:
            print("i/o error")
            return

        p.num_verts = int(lines[0])

        print("Reading " + str(p.num_verts) + " vertices")

        for i in range(1, p.num_verts + 1):
            p.vertices[i - 1] = Vector(float(lines[i].split()[0]), float(lines[i].split()[1]),
                                       float(lines[i].split()[2]))
        p.num_faces = int(lines[p.num_verts + 1])
        print("Reading " + str(p.num_faces) + " faces")
        index = 0
        for i in range(p.num_verts + 2, len(lines)):
            f = copy.deepcopy(p.faces[index])
            f.polyhedron = p
            f.num_verts = int(lines[i].split()[0])
            for j in range(f.num_verts):
                f.vertices[j] = int(lines[i].split()[j + 1])
            dx1 = p.vertices[f.vertices[1]][0] - p.vertices[f.vertices[0]][0]
            dy1 = p.vertices[f.vertices[1]][1] - p.vertices[f.vertices[0]][1]
            dz1 = p.vertices[f.vertices[1]][2] - p.vertices[f.vertices[0]][2]
            dx2 = p.vertices[f.vertices[2]][0] - p.vertices[f.vertices[1]][0]
            dy2 = p.vertices[f.vertices[2]][1] - p.vertices[f.vertices[1]][1]
            dz2 = p.vertices[f.vertices[2]][2] - p.vertices[f.vertices[1]][2]
            nx = dy1 * dz2 - dy2 * dz1
            ny = dz1 * dx2 - dz2 * dx1
            nz = dx1 * dy2 - dx2 * dy1
            len_ = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            if len_ == 0:
                f.norm[0] = float("nan")
                f.norm[1] = float("nan")
                f.norm[2] = float("nan")
            else:
                f.norm[0] = nx / len_
                f.norm[1] = ny / len_
                f.norm[2] = nz / len_
            f.w = -f.norm[0] * p.vertices[f.vertices[0]][0] - f.norm[1] * p.vertices[f.vertices[0]][1] - f.norm[2] * \
                  p.vertices[f.vertices[0]][2]
            p.faces[index] = f
            index += 1
        return p

    def comp_projection_integrals(self, f: Face):
        self.P1 = 0
        self.Pa = 0
        self.Pb = 0
        self.Paa = 0
        self.Pab = 0
        self.Pbb = 0
        self.Pbb = 0
        self.Paaa = 0
        self.Paab = 0
        self.Pabb = 0
        self.Pbbb = 0
        for i in range(f.num_verts):
            a0 = f.polyhedron.vertices[f.vertices[i]][self.A]
            b0 = f.polyhedron.vertices[f.vertices[i]][self.B]
            a1 = f.polyhedron.vertices[f.vertices[(i + 1) % f.num_verts]][self.A]
            b1 = f.polyhedron.vertices[f.vertices[(i + 1) % f.num_verts]][self.B]
            da = a1 - a0
            db = b1 - b0
            a0_2 = a0 * a0
            a0_3 = a0_2 * a0
            a0_4 = a0_3 * a0
            b0_2 = b0 * b0
            b0_3 = b0_2 * b0
            b0_4 = b0_3 * b0
            a1_2 = a1 * a1
            a1_3 = a1_2 * a1
            b1_2 = b1 * b1
            b1_3 = b1_2 * b1

            C1 = a1 + a0
            Ca = a1 * C1 + a0_2
            Caa = a1 * Ca + a0_3
            Caaa = a1 * Caa + a0_4
            Cb = b1 * (b1 + b0) + b0_2
            Cbb = b1 * Cb + b0_3
            Cbbb = b1 * Cbb + b0_4
            Cab = 3 * a1_2 + 2 * a1 * a0 + a0_2
            Kab = a1_2 + 2 * a1 * a0 + 3 * a0_2
            Caab = a0 * Cab + 4 * a1_3
            Kaab = a1 * Kab + 4 * a0_3
            Cabb = 4 * b1_3 + 3 * b1_2 * b0 + 2 * b1 * b0_2 + b0_3
            Kabb = b1_3 + 2 * b1_2 * b0 + 3 * b1 * b0_2 + 4 * b0_3

            self.P1 += db * C1
            self.Pa += db * Ca
            self.Paa += db * Caa
            self.Paaa += db * Caaa
            self.Pb += da * Cb
            self.Pbb += da * Cbb
            self.Pbbb += da * Cbbb
            self.Pab += db * (b1 * Cab + b0 * Kab)
            self.Paab += db * (b1 * Caab + b0 * Kaab)
            self.Pabb += da * (a1 * Cabb + a0 * Kabb)

        self.P1 /= 2
        self.Pa /= 6
        self.Paa /= 12
        self.Paaa /= 20
        self.Pb /= -6
        self.Pbb /= -12
        self.Pbbb /= -20
        self.Pab /= 24
        self.Paab /= 60
        self.Pabb /= -60

        return f

    def comp_face_integrals(self, f: Face):
        f = self.comp_projection_integrals(f)
        w = f.w
        n = f.norm
        if n[self.C] == 0:
            k1 = float("nan")
        else:
            k1 = 1 / n[self.C]
        k2 = k1 ** 2
        k3 = k2 * k1
        k4 = k3 * k1

        self.Fa = k1 * self.Pa
        self.Fb = k1 * self.Pb
        self.Fc = -k2 * (n[self.A] * self.Pa + n[self.B] * self.Pb + w * self.P1)

        self.Faa = k1 * self.Paa
        self.Fbb = k1 * self.Pbb
        self.Fcc = k3 * (
                n[self.A] ** 2 * self.Paa + 2 * n[self.A] * n[self.B] * self.Pab + n[self.B] ** 2 * self.Pbb + w * (
                2 * (n[self.A] * self.Pa + n[self.B] * self.Pb) + w * self.P1))

        self.Faaa = k1 * self.Paaa
        self.Fbbb = k1 * self.Pbbb
        self.Fccc = -k4 * (n[self.A] ** 3 * self.Paaa + 3 * n[self.A] ** 2 * n[self.B] * self.Paab
                           + 3 * n[self.A] * n[self.B] ** 2 * self.Pabb + n[self.B] ** 3 * self.Pbbb
                           + 3 * w * (n[self.A] ** 2 * self.Paa + 2 * n[self.A] * n[self.B] * self.Pab + n[
                    self.B] ** 2 * self.Pbb) + w * w * (
                                   3 * (n[self.A] * self.Pa + n[self.B] * self.Pb) + w * self.P1))
        self.Faab = k1 * self.Paab
        self.Fbbc = -k2 * (n[self.A] * self.Pabb + n[self.B] * self.Pbbb + w * self.Pbb)
        self.Fcca = k3 * (
                n[self.A] ** 2 * self.Paaa + 2 * n[self.A] * n[self.B] * self.Paab + n[self.B] ** 2 * self.Pabb
                + w * (2 * (n[self.A] * self.Paa + n[self.B] * self.Pab) + w * self.Pa))

        return f

    def comp_volume_integrals(self, p: Polyhedron):
        self.T0 = 0
        self.T1[0] = 0
        self.T1[1] = 0
        self.T1[2] = 0
        self.T2[0] = 0
        self.T2[1] = 0
        self.T2[2] = 0
        self.TP[0] = 0
        self.TP[1] = 0
        self.TP[2] = 0
        for i in range(p.num_faces):
            f = p.faces[i]
            nx = abs(f.norm[0])
            ny = abs(f.norm[1])
            nz = abs(f.norm[2])

            if nx > ny and nx > nz:
                self.C = 0
            else:
                if ny > nz:
                    self.C = 1
                else:
                    self.C = 2

            self.A = (self.C + 1) % 3
            self.B = (self.A + 1) % 3

            f = self.comp_face_integrals(f)

            if self.A == 0:
                c = self.Fa
            else:
                if self.B == 0:
                    c = self.Fb
                else:
                    c = self.Fc
            self.T0 += f.norm[0] * c
            self.T1[self.A] += f.norm[self.A] * self.Faa
            self.T1[self.B] += f.norm[self.B] * self.Fbb
            self.T1[self.C] += f.norm[self.C] * self.Fcc
            self.T2[self.A] += f.norm[self.A] * self.Faaa
            self.T2[self.B] += f.norm[self.B] * self.Fbbb
            self.T2[self.C] += f.norm[self.C] * self.Fccc
            self.TP[self.A] += f.norm[self.A] * self.Faab
            self.TP[self.B] += f.norm[self.B] * self.Fbbc
            self.TP[self.C] += f.norm[self.C] * self.Fcca

        self.T1[0] /= 2
        self.T1[1] /= 2
        self.T1[2] /= 2
        self.T2[0] /= 3
        self.T2[1] /= 3
        self.T2[2] /= 3
        self.TP[0] /= 2
        self.TP[1] /= 2
        self.TP[2] /= 2

        return p

    def solve(self):
        p = Polyhedron()
        p = self.read_polyhedron("settings.txt", p)
        self.comp_volume_integrals(p)
        density = 1
        mass = density * self.T0
        r = []
        '''center of mass'''

        r.append(self.T1[0] / self.T0)
        r.append(self.T1[1] / self.T0)
        r.append(self.T1[2] / self.T0)

        J = [[0] * 3] * 3
        '''inertia tensor'''
        J[0][0] = density * (self.T2[1] + self.T2[2])
        J[1][1] = density * (self.T2[2] + self.T2[0])
        J[2][2] = density * (self.T2[0] + self.T2[1])

        J[0][1] = J[1][0] = -density * self.Tp[0]
        J[1][2] = J[2][1] = -density * self.Tp[1]
        J[2][0] = J[0][2] = -density * self.Tp[2]

        '''inertia tensor to center of mass'''
        J[0][0] -= mass * (r[1] ** 2 + r[2] ** 2)
        J[1][1] -= mass * (r[2] ** 2 + r[0] ** 2)
        J[2][2] -= mass * (r[0] ** 2 + r[1] ** 2)

        J[1][0] += mass * r[0] * r[1]
        J[2][1] += mass * r[1] * r[2]
        J[0][2] += mass * r[2] * r[0]

        J[0][1] = J[1][0]
        J[1][2] = J[2][1]
        J[2][0] = J[0][2]

        return


class State:
    def __init__(self, pos=Vector(), impulse=Vector(), R=Matrix(), am=0):
        self.pos = pos
        self.impulse = impulse
        self.R = R
        self.am = am

    def __mul__(self, r):
        return State(self.pos * r, self.impulse * r, self.R * r, self.am * r)

    def __add__(self, r):
        return State(r.pos + self.pos, r.impulse + self.impulse, r.R + self.R, self.am + r.am)


class Context:
    def __init__(self):
        self.mg = 0
        self.mass = 0
        self.tensor_in = Matrix()


class Iceberg:

    def __init__(self):
        self.context = Context()
        self.state = State()
        self.c_m = Vector()
        self.body = Polyhedron()
        self.density = 1

        J = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        p = Volume_integral()

        self.body = p.read_polyhedron("settings.txt", self.body)
        self.body = p.comp_volume_integrals(self.body)
        self.context.mass = self.density * p.T0

        self.c_m.x = p.T1[0] / p.T0
        self.c_m.y = p.T1[1] / p.T0
        self.c_m.z = p.T1[2] / p.T0
        for i in range(self.body.num_verts):
            self.body.vertices[i][0] -= self.c_m.x
            self.body.vertices[i][1] -= self.c_m.y
            self.body.vertices[i][2] -= self.c_m.z

        self.c_m.x = 0
        self.c_m.y = 0
        self.c_m.z = 0
        J[0] = self.density * (p.T2[1] + p.T2[2])
        J[4] = self.density * (p.T2[2] + p.T2[0])
        J[8] = self.density * (p.T2[0] + p.T2[1])
        J[3] = -self.density * p.TP[0]
        J[1] = -self.density * p.TP[0]
        J[7] = -self.density * p.TP[1]
        J[5] = -self.density * p.TP[1]
        J[2] = -self.density * p.TP[2]
        J[6] = -self.density * p.TP[2]

        self.context.tensor_in = Matrix.inv(Matrix(J))

    def to_global1(self, vertex: Vector):
        return self.state.pos + self.state.R * (vertex - self.c_m)

    def to_global2(self, vertex: Vector, a: State):
        return a.pos + a.R * (vertex - self.c_m)

    def polyhedron_add_face(self, p1: Polyhedron, inf, i):
        g1 = Face()
        g1.num_verts = 3

        n = Vector()
        n.x = self.body.faces[i].norm[0]
        n.y = self.body.faces[i].norm[1]
        n.z = self.body.faces[i].norm[2]

        n = self.state.R * n

        v1 = Vector(p1.vertices[inf][0], p1.vertices[inf][1], p1.vertices[inf][2])
        v2 = Vector(p1.vertices[inf + 1][0], p1.vertices[inf + 1][1], p1.vertices[inf + 1][2])
        v3 = Vector(p1.vertices[inf + 2][0], p1.vertices[inf + 2][1], p1.vertices[inf + 2][2])

        v4 = v2 - v1
        v5 = v3 - v1
        v6 = Vector.cross(v4, v5)
        dot = Vector.dot(n, v6)

        g1.norm[0] = n.x
        g1.norm[1] = n.y
        g1.norm[2] = n.z

        if dot > 0:
            g1.vertices[0] = inf
            g1.vertices[1] = inf + 1
            g1.vertices[2] = inf + 2
        else:
            g1.vertices[0] = inf
            g1.vertices[1] = inf + 2
            g1.vertices[2] = inf + 1

        g1.w = -g1.norm[0] * p1.vertices[g1.vertices[0]][0] - g1.norm[1] * p1.vertices[g1.vertices[0]][1] - g1.norm[2] * \
               p1.vertices[g1.vertices[0]][2]

        g1.polyhedron = p1
        p1.faces[p1.num_faces] = g1
        p1.num_faces += 1
        return p1

    def func(self, res: State, input: State, context: Context):
        if context.mass == 0:
            res.pos = Vector(float("nan"), float("nan"), float("nan"))
        else:
            res.pos = input.impulse / context.mass

        volume = 0
        cm = Vector()
        volume, cm = self.comp_volume(volume, cm, input)

        if volume == 0:
            cm = input.pos

        f = Vector(0, 2.3 * 9.8 * volume, 0)
        r = cm - input.pos
        res.impulse = context.mg + f
        g = input.R.t()
        d = input.R * context.tensor_in * g * input.am
        res.R = Matrix.cross_matrix(d) * input.R
        res.am = Vector.cross(r, f)

        return res

    def rk4(self, ice, dt):
        k1 = State()
        k2 = State()
        k3 = State()
        k4 = State()
        copy1 = copy.deepcopy(ice.state)
        k1 = self.func(k1, copy1, ice.context)
        ice.state = copy1
        k2 = self.func(k2, copy1 + k1 * (dt / 2), ice.context)
        ice.state = copy1
        k3 = self.func(k3, copy1 + k2 * (dt / 2), ice.context)
        ice.state = copy1
        k4 = self.func(k4, copy1 + k3 * dt, ice.context)
        ice.state = copy1
        ice.state += k1 * (dt / 6)
        ice.state += k2 * (dt / 3)
        ice.state += k3 * (dt / 3)
        ice.state += k4 * (dt / 6)

        ice.state.R = Matrix.ort_gram_schmidt(ice.state.R)
        return ice

    def comp_volume(self, volume, c_m, a: State):
        p1 = self.polyhedron_volume_cross(a)
        p = Volume_integral()
        p.comp_volume_integrals(p1)
        if p.T0 != 0:
            c_m.x = p.T1[0] / p.T0
            c_m.y = p.T1[1] / p.T0
            c_m.z = p.T1[2] / p.T0
        else:
            c_m.x, c_m.y, c_m.z = float("-nan"), float("-nan"), float("-nan")

        volume = p.T0
        return volume, c_m

    def draw(self):
        light = Vector(1, -1, 0).normalize()
        glBegin(GL_TRIANGLES)
        for i in range(self.body.num_faces):
            for j in range(self.body.faces[i].num_verts):
                gl = self.to_global1(Vector(
                    self.body.vertices[self.body.faces[i].vertices[j]][0],
                    self.body.vertices[self.body.faces[i].vertices[j]][1],
                    self.body.vertices[self.body.faces[i].vertices[j]][2]))

                norm = Vector(self.body.faces[i].norm[0], self.body.faces[i].norm[1], self.body.faces[i].norm[2])
                c = 0.5 + 0.5 * max(-0.2, -Vector.dot(light, self.state.R * norm))

                glColor(c, c, c)
                glVertex3f(gl.x, gl.y, gl.z)
        glEnd()

    def polyhedron_add_vertex(self, p: Polyhedron, v: Vector) -> Polyhedron:
        p.vertices[p.num_verts][0] = v.x
        p.vertices[p.num_verts][1] = v.y
        p.vertices[p.num_verts][2] = v.z
        p.num_verts += 1
        return p

    def polyhedron_volume_cross(self, a: State) -> Polyhedron:
        p1 = Polyhedron()
        p1.num_verts = 0
        p1.num_faces = 0
        cross = []
        for i in range(self.body.num_faces):
            v_under_water = []
            v_above_water = []
            v = []

            v1 = Vector(self.body.vertices[self.body.faces[i].vertices[0]][0],
                        self.body.vertices[self.body.faces[i].vertices[0]][1],
                        self.body.vertices[self.body.faces[i].vertices[0]][2])

            v2 = Vector(self.body.vertices[self.body.faces[i].vertices[1]][0],
                        self.body.vertices[self.body.faces[i].vertices[1]][1],
                        self.body.vertices[self.body.faces[i].vertices[1]][2])

            v3 = Vector(self.body.vertices[self.body.faces[i].vertices[2]][0],
                        self.body.vertices[self.body.faces[i].vertices[2]][1],
                        self.body.vertices[self.body.faces[i].vertices[2]][2])

            v.append(self.to_global2(v1, a))
            v.append(self.to_global2(v2, a))
            v.append(self.to_global2(v3, a))

            for j in range(len(v)):
                if v[j].y < 0 or math.isnan(v[j].y):
                    v_under_water.append(v[j])
                else:
                    v_above_water.append(v[j])

            if len(v_under_water) == 1:
                s1 = Section(v_under_water[0], v_above_water[0])
                s2 = Section(v_under_water[0], v_above_water[1])

                point1 = s1.crossing()
                point2 = s2.crossing()

                inf = p1.num_verts

                p1 = self.polyhedron_add_vertex(p1, point1)
                p1 = self.polyhedron_add_vertex(p1, point2)
                p1 = self.polyhedron_add_vertex(p1, v_under_water[0])

                cross.append(inf)
                cross.append(inf + 1)

                p1 = self.polyhedron_add_face(p1, inf, i)

            elif len(v_under_water) == 2:
                s1 = Section(v_under_water[0], v_above_water[0])
                s2 = Section(v_under_water[1], v_above_water[0])

                point1 = s1.crossing()
                point2 = s2.crossing()

                inf = p1.num_verts

                p1 = self.polyhedron_add_vertex(p1, v_under_water[0])
                p1 = self.polyhedron_add_vertex(p1, v_under_water[1])

                p1 = self.polyhedron_add_vertex(p1, point1)
                p1 = self.polyhedron_add_vertex(p1, point2)

                p1 = self.polyhedron_add_face(p1, inf, i)
                p1 = self.polyhedron_add_face(p1, inf + 1, i)

                cross.append(inf + 2)
                cross.append(inf + 3)

            elif len(v_under_water) == 3:

                inf = p1.num_verts

                p1 = self.polyhedron_add_vertex(p1, v[0])
                p1 = self.polyhedron_add_vertex(p1, v[1])
                p1 = self.polyhedron_add_vertex(p1, v[2])

                p1 = self.polyhedron_add_face(p1, inf, i)

        sum = Vector()
        for vertex in cross:
            v1 = Vector(p1.vertices[vertex][0], p1.vertices[vertex][1], p1.vertices[vertex][2])
            sum += v1
        sum /= len(cross)
        c = p1.num_verts

        p1 = self.polyhedron_add_vertex(p1, sum)

        for i in range(0, len(cross), 2):
            g1 = Face()
            g1.num_verts = 3

            n = Vector(0, 1, 0)
            i1 = cross[i]
            i2 = cross[i + 1]

            v1 = Vector(p1.vertices[i1][0], p1.vertices[i1][1], p1.vertices[i1][2])
            v2 = Vector(p1.vertices[i2][0], p1.vertices[i2][1], p1.vertices[i2][2])
            v3 = Vector(p1.vertices[c][0], p1.vertices[c][1], p1.vertices[c][2])
            v4 = v2 - v1
            v5 = v3 - v1
            v6 = Vector.cross(v4, v5)
            dot = Vector.dot(n, v6)

            g1.norm[0] = n.x
            g1.norm[1] = n.y
            g1.norm[2] = n.z

            if dot > 0:
                g1.vertices = Vector(i1, i2, c)
            else:
                g1.vertices = Vector(i1, c, i2)

            g1.w = -g1.norm[0] * p1.vertices[g1.vertices[0]][0] - g1.norm[1] * p1.vertices[g1.vertices[0]][1] - g1.norm[
                2] * \
                   p1.vertices[g1.vertices[0]][2]

            g1.polyhedron = p1
            p1.faces[p1.num_faces] = g1
            p1.num_faces += 1

        return p1