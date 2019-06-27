import math




class L2R_LrFunction():

    def __init__(self, prob, C):
        l = prob.l
        self.prob = prob
        self.z = [0.0 for _ in range(l)]
        self.D = [0.0 for _ in range(l)]
        self.C = C

    def fun(self, w):
        f = 0
        y = self.prob.y
        l = self.prob.l
        w_size = len(w)

        # for i in range(len(self.prob.x)):
        #     print([[self.prob.x[i][j].getIndex(), self.prob.x[i][j].getValue()] for j in range(len(self.prob.x[i]))])

        self.Xv(w)

        for i in range(w_size):
            f += w[i] * w[i]
        f /= 2.0

        for i in range(l):
            yz = y[i] * self.z[i]
            if yz >= 0:
                f += self.C[i] * math.log(1 + math.exp(-yz))
            else:
                f += self.C[i] * (-yz + math.log(1 + math.exp(yz)))

        return f


    def grad(self, w, g):
        y = self.prob.y
        l = self.prob.l
        w_size = len(w)

        for i in range(l):
            self.z[i] = 1 / (1 + math.exp(-y[i] * self.z[i]))
            self.D[i] = self.z[i] * (1 - self.z[i])
            self.z[i] = self.C[i] * (self.z[i] - 1) * y[i]

        g = self.XTv(self.z, g)

        for i in range(w_size):
            g[i] = w[i] + g[i]


        return g


    def Xv(self, v):
        l = self.prob.l
        x = self.prob.x

        for i in range(l):
            self.z[i] = self.dot(v, x[i])


    def XTv(self, v, XTv):
        l = self.prob.l
        w_size = self.prob.n
        x = self.prob.x

        for i in range(w_size):
            XTv[i] = 0

        for i in range(l):
            XTv = self.axpy(v[i], x[i], XTv)
        return XTv


    def Hv(self, s, Hs):
        l = self.prob.l
        w_size = self.prob.n
        x = self.prob.x

        for i in range(w_size):
            Hs[i] = 0
        for i in range(l):
            xi = x[i]
            xTs = self.dot(s, xi)
            xTs = self.C[i] * self.D[i] * xTs

            Hs = self.axpy(xTs, xi, Hs)

        for i in range(w_size):
            Hs[i] = s[i] + Hs[i]


    def get_diagH(self, M):
        l = self.prob.l
        w_size = self.prob.n
        x = self.prob.x

        for i in range(w_size):
            M[i] = 1

        for i in range(l):
            for s in x[i]:
                M[s.getIndex() - 1] += s.getValue() * s.getValue() * self.C[i] * self.D[i]

        return M


    def dot(self, s, x):
        ret = 0
        for feature in x:
            ret += s[feature.getIndex() - 1] * feature.getValue()

        return ret


    def axpy(self, a, x, y):
        for feature in x:
            y[feature.getIndex() - 1] += a * feature.getValue()
        return y
