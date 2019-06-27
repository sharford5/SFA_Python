import math

class Tron():

    def __init__(self, fun_obj, eps, max_iter, eps_cg):
        self.fun_obj = fun_obj
        self.eps = eps
        self.max_iter = max_iter
        self.eps_cg = eps_cg


    def tron(self, w):
        # Parameters for updating the iterates.
        eta0 = 1e-4
        eta1 = 0.25
        eta2 = 0.75

        # Parameters for updating the trust region size delta.
        sigma1 = 0.25
        sigma2 = 0.5
        sigma3 = 4

        n =  len(w)  #fun_obj.get_nr_variable()
        delta = 0
        one = 1.0
        search = 1
        iter = 1
        s = [0.0 for _ in range(n)]
        r = [0.0 for _ in range(n)]
        g = [0.0 for _ in range(n)]

        alpha_pcg = 0.01
        M = [0.0 for _ in range(n)]

        # calculate gradient norm at w=0 for stopping condition.
        w0 = [0.0 for _ in range(n)]

        self.fun_obj.fun(w0)
        g = self.fun_obj.grad(w0, g)
        gnorm0 = self.euclideanNorm(g)

        f = self.fun_obj.fun(w)
        g = self.fun_obj.grad(w, g)
        gnorm = self.euclideanNorm(g)

        if gnorm <= self.eps * gnorm0:
            search = 0

        iter = 1
        w_new = [0.0 for _ in range(n)]
        reach_boundary = False
        while (iter <= self.max_iter) and (search != 0):
            M = self.fun_obj.get_diagH(M)
            for i in range(n):
                M[i] = (1 - alpha_pcg) + alpha_pcg * M[i]

            if iter == 1:
                delta = self.uTMv(n, g, M, g)**(1/2)

            cg_iter = self.trpcg(delta, g, M, s, r, reach_boundary)

            w_new = self.arrayCopy(w, 0, w_new, 0, n)
            self.daxpy(one, s, w_new)

            gs = self.dot(g, s)
            prered = -0.5 * (gs - self.dot(s, r))
            fnew = self.fun_obj.fun(w_new)

            # Compute the actual reduction.
            actred = f - fnew
            # On the first iteration, adjust the initial step bound.
            sMnorm = self.uTMv(n, s, M, s)**(1/2)
            if iter == 1:
                delta = min(delta, sMnorm)

            # Compute prediction alpha*sMnorm of the step.
            if fnew - f - gs <= 0:
                alpha = sigma3
            else:
                alpha = max(sigma1, -0.5 * (gs / (fnew - f - gs)))

            # Update the trust region bound according to the ratio of actual to
            # predicted reduction.
            if actred < eta0 * prered:
                delta = min(alpha * sMnorm, sigma2 * delta)
            elif actred < eta1 * prered:
                delta = max(sigma1 * delta, min(alpha * sMnorm, sigma2 * delta));
            elif actred < eta2 * prered:
                delta = max(sigma1 * delta, min(alpha * sMnorm, sigma3 * delta));
            else:
                if reach_boundary:
                    delta = sigma3 * delta
                else:
                    delta = max(delta, min(alpha * sMnorm, sigma3 * delta))

            # print("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d" % (iter, actred, prered, delta, f, gnorm, cg_iter))

            if actred > eta0 * prered:
                iter += 1
                w = self.arrayCopy(w_new, 0, w, 0, n)
                f = fnew
                self.fun_obj.grad(w, g);

                gnorm = self.euclideanNorm(g)
                if gnorm <= self.eps * gnorm0:
                    break

            if f < -1.0e+32:
                print("WARNING: f < -1.0e+32")
                break

            if prered <= 0:
                print("WARNING: prered <= 0")
                break

            if (abs(actred) <= 1.0e-12 * abs(f)) and (abs(prered) <= 1.0e-12 * abs(f)):
                print("WARNING: actred and prered too small")
                break




    def daxpy(self, constant, vector1, vector2):
        if constant == 0:
            return

        for i in range(len(vector1)):
            vector2[i] += constant * vector1[i]


    def dot(self, vector1, vector2):
        product = 0
        if len(vector1) == len(vector2):
            for i in range(len(vector1)):
                product += vector1[i] * vector2[i]
        return product


    def euclideanNorm(self, vector):
        n = len(vector)

        if n < 1:
            return 0

        if n == 1:
            return abs(vector[0])

        # this algorithm is (often) more accurate than just summing up the squares and taking the square-root afterwards

        scale = 0  # scaling factor that is factored out
        sum = 1  # basic sum of squares from which scale has been factored out
        for i in range(n):
            if vector[i] != 0:
                a = abs(vector[i])
                # try to get the best scaling factor
                if scale < a:
                    t = scale / a
                    sum = 1 + sum * (t * t)
                    scale = a
                else:
                    t = a / scale
                    sum += t * t

        return scale * sum**(1/2)


    def trpcg(self, delta, g, M, s, r, reach_boundary):
        n = self.fun_obj.prob.n
        one = 1
        d = [0.0 for _ in range(n)]
        Hd = [0.0 for _ in range(n)]
        z = [0.0 for _ in range(n)]

        reach_boundary = False
        for i in range(n):
            s[i] = 0
            r[i] = -g[i]
            z[i] = r[i] / M[i]
            d[i] = z[i]

        zTr = self.dot(z, r)
        cgtol = self.eps_cg * zTr**(1/2)
        cg_iter = 0

        while True:
            if zTr**(1/2) <= cgtol:
                break

            cg_iter += 1
            self.fun_obj.Hv(d, Hd)

            alpha = zTr / self.dot(d, Hd)
            self.daxpy(alpha, d, s)

            sMnorm = self.uTMv(n, s, M, s)**(1/2)
            if sMnorm > delta:
                # info("cg reaches trust region boundary%n");
                reach_boundary = True
                alpha = -alpha
                self.daxpy(alpha, d, s)

                sTMd = self.uTMv(n, s, M, d)
                sTMs = self.uTMv(n, s, M, s)
                dTMd = self.uTMv(n, d, M, d)
                dsq = delta * delta
                rad = (sTMd * sTMd + dTMd * (dsq - sTMs))**(1/2)
                if sTMd >= 0:
                    alpha = (dsq - sTMs) / (sTMd + rad)
                else:
                    alpha = (rad - sTMd) / dTMd
                self.daxpy(alpha, d, s)
                alpha = -alpha
                self.daxpy(alpha, Hd, r)
                break

            alpha = -alpha
            self.daxpy(alpha, Hd, r)

            for i in range(n):
                z[i] = r[i] / M[i]
            znewTrnew = self.dot(z, r)
            beta = znewTrnew / zTr
            self.scale(beta, d)
            self.daxpy(one, z, d)
            zTr = znewTrnew

        return cg_iter


    def scale(self, constant, vector):
        if constant == 1.0:
            return
        for i in range(len(vector)):
            vector[i] *= constant


    def uTMv(self, n, u, M, v):
        m = n - 4
        res = 0
        i=0
        while i < m:
            res += u[i] * M[i] * v[i] + u[i + 1] * M[i + 1] * v[i + 1] + u[i + 2] * M[i + 2] * v[i + 2] + u[i + 3] * M[i + 3] * v[i + 3] + u[i + 4] * M[i + 4] * v[i + 4]
            i += 5

        while i < n:
            res += u[i] * M[i] * v[i]
            i += 1
        return res


    def arrayCopy(self, src, srcPos, dest, destPos, length):
        for i in range(length):
            dest[i + destPos] = src[i + srcPos]
        return dest
