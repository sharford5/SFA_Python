from src.LibLinear.Model import *
import numpy as np
from src.LibLinear.Problem import *
from src.LibLinear.Tron import *
import random
import math

class Linear():

    def train(self, prob, param):

        var2 = prob.x
        n = len(var2)

        for w_size in range(n):
            nodes = var2[w_size]
            indexBefore = 0
            var7 = nodes
            nr_class = len(nodes)
            for var9 in range(nr_class):
                n = var7[var9]
                if n.getIndex() <= indexBefore:
                    return "feature nodes must be sorted by index in ascending order"
                indexBefore = n.getIndex()

        l = prob.l
        n = prob.n
        w_size = prob.n
        model = Model()
        if prob.bias >= 0.0:
            model.nr_feature = n - 1
        else:
            model.nr_feature = n

        model.solverType = param.solverType
        model.bias = prob.bias


        if param.solverType.isSupportVectorRegression():
            model.w = [None for _ in range(w_size)]
            model.nr_class = 2
            model.label = None
            return "Not Set for Regression"
            # checkProblemSize(n, model.nr_class);
            # train_one(prob, param, model.w, 0.0D, 0.0D);
        else:
            perm = [0 for _ in range(l)]
            perm, rv = self.groupClasses(prob, perm)
            nr_class = rv.nr_class
            label = rv.label
            start = rv.start
            count = rv.count
            # checkProblemSize(n, nr_class);
            model.nr_class = nr_class
            model.label = label
            weighted_C = [param.c for _ in range(nr_class)]

            #Removed part with param weights

            x = [prob.x[perm[j]] for j in range(len(perm))]

            sub_prob = Problem()
            sub_prob.l = l
            sub_prob.n = n
            sub_prob.x = [x[u] for u in range(sub_prob.l)]
            sub_prob.y = [0.0 for _ in range(sub_prob.l)]

            if param.solverType.solvertype == "MCSVM_CS":
                model.w = [0.0 for _ in range(n * nr_class)]
                for i in range(nr_class):
                    i = start[i]
                    while i < start[i] + count[i]:
                        sub_prob.y[i] = i
                        i += 1
                #TODO Not relevant for me now
                # SolverMCSVM_CS solver = new SolverMCSVM_CS(sub_prob, nr_class, weighted_C, param.eps);
                # solver.solve(model.w);
            elif nr_class == 2:
                model.w = [0. for _ in range(w_size)]
                i = start[0] + count[0]

                for i in range(0, i):
                    sub_prob.y[i] = 1.0

                i += 1
                while i < sub_prob.l:
                    sub_prob.y[i] = -1.0
                    i += 1


                w = self.train_one(sub_prob, param, model.w, weighted_C[0], weighted_C[1])
            else:
                model.w = [0. for _ in range(w_size * nr_class)]
                w = [0. for _ in range(w_size)]

                for i in range(nr_class):
                    si = start[i]
                    ei = si + count[i]

                    K = 0
                    for _ in range(si):
                        sub_prob.y[K] = -1.0
                        K += 1

                    while K < ei:
                        sub_prob.y[K] = 1.0
                        K += 1

                    while K < sub_prob.l:
                        sub_prob.y[K] = -1.0
                        K += 1

                    w = self.train_one(sub_prob, param, w, weighted_C[i], param.c)

                    for j in range(n):
                        model.w[j * nr_class + i] = w[j]
        return model


    def train_one(self, prob, param, w, Cp, Cn):
        eps = param.eps
        pos = 0

        for i in range(prob.l):
            if prob.y[i] > 0.0:
                pos += 1

        i = prob.l - pos
        primal_solver_tol = eps * max(min(pos, i), 1) / prob.l
        fun_obj = None
        C = 0.0
        i = 0.0
        prob_col = Problem()
        tron_obj = Tron()

        if param.solverType.solvertype == 'L2R_LR':
            C = [0. for _ in range(prob.l)]

            for i in range(prob.l):
                if prob.y[i] > 0.0:
                    C[i] = Cp
                else:
                    C[i] = Cn

            # fun_obj = L2R_LrFunction(prob, C)
            # tron_obj = new Tron(fun_obj, primal_solver_tol);
            # tron_obj.tron(w);
            # break;
        # elif param.solverType.solvertype == 'L2R_L2LOSS_SVC':
            # C = [0. for _ in range(prob.l)]
            #
            # for(i = 0; i < prob.l; ++i) {
            #     if (prob.y[i] > 0.0D) {
            #         C[i] = Cp;
            #     } else {
            #         C[i] = Cn;
            #     }
            # }
            #
            # Function fun_obj = new L2R_L2_SvcFunction(prob, C);
            # tron_obj = new Tron(fun_obj, primal_solver_tol);
            # tron_obj.tron(w);
        # elif param.solverType.solvertype == 'L2R_L2LOSS_SVC_DUAL':
            # solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L2LOSS_SVC_DUAL);
        # elif param.solverType.solvertype == 'L2R_L1LOSS_SVC_DUAL':
            # solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L1LOSS_SVC_DUAL);
        # elif param.solverType.solvertype == 'L1R_L2LOSS_SVC':
            # prob_col = transpose(prob);
            # solve_l1r_l2_svc(prob_col, w, primal_solver_tol, Cp, Cn);
        # elif param.solverType.solvertype == 'L1R_LR':
            # prob_col = transpose(prob);
            # solve_l1r_lr(prob_col, w, primal_solver_tol, Cp, Cn);
        elif param.solverType.solvertype == 'L2R_LR_DUAL':   #TODO Only one that works
            w2 = self.solve_l2r_lr_dual(prob, w, eps, Cp, Cn)
        # elif param.solverType.solvertype == 'L2R_L2LOSS_SVR':
            # C = new double[prob.l];
            #
            # for(i = 0; i < prob.l; ++i) {
            #     C[i] = param.C;
            # }
            #
            # fun_obj = new L2R_L2_SvrFunction(prob, C, param.p);
            # tron_obj = new Tron(fun_obj, param.eps);
            # tron_obj.tron(w);
        # elif param.solverType.solvertype == 'L2R_L1LOSS_SVR_DUAL':
        # elif param.solverType.solvertype == 'L2R_L2LOSS_SVR_DUAL':
            # solve_l2r_l1l2_svr(prob, w, param);
        else:
            print("unknown solver type: " + param.solverType.solvertype)

        return w2


    def groupClasses(self, prob, perm):
        l = prob.l
        label = []
        for this in prob.y:
            if this in label:
                continue
            else:
                label.append(this)
        count = [0 for _ in range(len(label))]
        data_label = [label.index(lab) for lab in prob.y]
        nr_class = len(label)

        for lab in prob.y:
            count[label.index(lab)] += 1

        if (nr_class == 2) & (label[0] == -1) & (label[1] == 1):
            label = self.swap(label, 0, 1)
            count = self.swap(count, 0, 1)

            for i in range(l):
                if data_label[i] == 0:
                    data_label[i] = 1
                else:
                    data_label[i] = 0

        start = [0 for _ in range(nr_class)]
        start[0] = 0

        for i in range(1, nr_class):
            start[i] = start[i - 1] + count[i - 1]

        for i in range(l):
            perm[start[data_label[i]]] = i
            start[data_label[i]] += 1

        start[0] = 0

        for i in range(1, nr_class):
            start[i] = start[i - 1] + count[i - 1]

        return perm, GroupClassesReturn(nr_class, label, start, count)


    def copyOf(self, original, newLength):
        copy = [None for _ in range(newLength)]
        copy = self.arrayCopy(original, 0, copy, 0, min(original.length, newLength))
        return copy


    def arrayCopy(self, src, srcPos, dest, destPos, length):
        for i in range(length):
            dest[i + destPos] = src[i + srcPos]
        return dest


    def swap(self, array, idxA, idxB):
        temp = array[idxA]
        array[idxA] = array[idxB]
        array[idxB] = temp
        return array


    def solve_l2r_lr_dual(self, prob, w, eps, Cp, Cn):
        l = prob.l
        w_size = prob.n
        iter = 0
        xTx = [0. for _ in range(l)]
        max_iter = 1000
        index = [0 for _ in range(l)]
        alpha = [0. for _ in range(2 * l)]
        y = [0 for _ in range(l)]
        max_inner_iter = 100
        innereps = 0.01
        innereps_min = min(1.0E-8, eps)
        upper_bound = [Cn, 0.0, Cp]

        for i in range(l):
            if prob.y[i] > 0.:
                y[i] = 1
            else:
                y[i] = -1


        for i in range(l):
            alpha[2 * i] = min(0.001 * upper_bound[y[i]+1], 1.0E-8)
            alpha[2 * i + 1] = upper_bound[y[i]+1] - alpha[2 * i]


        for i in range(w_size):
            w[i] = 0.0

        var10001 = 0.
        C = 0.
        i = 0
        for i in range(l):
            xTx[i] = 0.0
            var24 = prob.x[i]
            var25 = len(var24)
            for var26 in range(var25):
                xi = var24[var26]
                C = xi.value
                xTx[i] += C * C
                var10001 = xi.index - 1
                w[var10001] += y[i] * alpha[2 * i] * C
            index[i] += i

        while iter < max_iter:
            newton_iter = 0
            for i in range(l):
                newton_iter = i + random.randint(0,l - i-1)
                index = self.swap(index, i, newton_iter)

            newton_iter = 0
            Gmax = 0.0

            for s in range(l):
                i = index[s]
                yi = y[i]
                C = upper_bound[y[i] + 1]
                ywTx = 0.0
                xisq = xTx[i]
                var34 = prob.x[i]
                var35 = len(var34)

                for var36 in range(var35):
                    xi = var34[var36]
                    ywTx += w[xi.index - 1] * xi.value

                ywTx *= y[i]
                a = xisq
                b = ywTx
                ind1 = 2 * i
                ind2 = 2 * i + 1
                sign = 1
                condition = 0.5 * xisq * (alpha[ind2] - alpha[ind1]) + ywTx
                if condition < 0.0:
                    ind1 = 2 * i + 1
                    ind2 = 2 * i
                    sign = -1

                alpha_old = alpha[ind1]
                z = alpha_old
                if C - alpha_old < 0.5 * C:
                    z = 0.1 * alpha_old

                gp = xisq * (z - alpha_old) + sign * ywTx + math.log(z / (C - z))
                Gmax = max(Gmax, abs(gp))
                eta = 0.1

                inner_iter = 0
                while (inner_iter <= max_inner_iter) & (abs(gp) >= innereps):
                    gpp = a + C / (C - z) / z
                    tmpz = z - gp / gpp
                    if tmpz <= 0.0:
                        z *= 0.1
                    else:
                        z = tmpz

                    gp = a * (z - alpha_old) + sign * b + math.log(z / (C - z))
                    newton_iter += 1
                    inner_iter += 1

                if inner_iter > 0:
                    alpha[ind1] = z
                    alpha[ind2] = C - z
                    var60 = prob.x[i]
                    var51 = len(var60)

                    for var61 in range(var51):
                        xi = var60[var61]
                        var10001 = xi.index - 1
                        w[var10001] += sign * (z - alpha_old) * yi * xi.value

            iter += 1

            if Gmax < eps:
                break

            if newton_iter <= l / 10:
                innereps = max(innereps_min, 0.1 * innereps)

        #Ordering of W is different that JAVA

        v = 0.0

        for i in range(w_size):
            v += w[i] * w[i]

        v *= 0.5

        for i in range(l):
            v += alpha[2 * i] * math.log(alpha[2 * i]) + alpha[2 * i + 1] * math.log(alpha[2 * i + 1]) - upper_bound[y[i]+1] * math.log(upper_bound[y[i] + 1])

        return w


    def predict(self, model, x):
        dec_values = [0. for _  in range(model.nr_class)]
        return self.predictValues(model, x, dec_values)


    def predictValues(self, model, x, dec_values):
        n = 0
        if model.bias >= 0.0:
            n = model.nr_feature + 1
        else:
            n = model.nr_feature

        w = model.w
        nr_w = 0
        if (model.nr_class == 2) & (model.solverType.solvertype != 'MCSVM_CS'):
            nr_w = 1
        else:
            nr_w = model.nr_class

        for dec_max_idx in range(nr_w):
            dec_values[dec_max_idx] = 0.0

        var12 = x
        i = len(x)

        for var8 in range(i):
            lx = var12[var8]
            idx = lx.index
            if idx <= n:
                for i in range(nr_w):
                    dec_values[i] += w[(idx - 1) * nr_w + i] * lx.value

        if model.nr_class == 2:
            if model.solverType.isSupportVectorRegression():
                return dec_values[0]
            else:
                return model.label[0] if dec_values[0] > 0.0 else model.label[1]
        else:
            dec_max_idx = 0

            for i in range(1, model.nr_class):
                if dec_values[i] > dec_values[dec_max_idx]:
                    dec_max_idx = i

            return model.label[dec_max_idx]


class GroupClassesReturn():
    def __init__(self, nr_class, label, start, count):
        self.nr_class = nr_class
        self.label = label
        self.start = start
        self.count = count






