

class SolverType():
    solverdict = {'L2R_LR':[0, True, False],
                  'L2R_L2LOSS_SVC_DUAL':[1, False, False],
                  'L2R_L2LOSS_SVC':[2, False, False],
                  'L2R_L1LOSS_SVC_DUAL':[3, False, False],
                  'MCSVM_CS':[4, False, False],
                  'L1R_L2LOSS_SVC':[5, False, False],
                  'L1R_LR':[6, True, False],
                  'L2R_LR_DUAL':[7, True, False],
                  'L2R_L2LOSS_SVR':[11, False, True],
                  'L2R_L2LOSS_SVR_DUAL':[12, False, True],
                  'L2R_L1LOSS_SVR_DUAL' :[13, False, True]}

    def __init__(self, string, sd = solverdict):
        self.solvertype = string
        self.id, self.logisticRegressionSolver, self.supportVectorRegression = sd[string]

    def isLogisticRegressionSolver(self):
        return self.logisticRegressionSolver

    def isSupportVectorRegression(self):
        return self.supportVectorRegression


