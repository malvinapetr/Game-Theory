import numpy as np
from numpy import genfromtxt
from numpy.linalg import matrix_rank
import time
import random

# rest of your code here

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import linprog

import os

############################### FG COLOR DEFINITIONS ###############################


class bcolors:
    # pure colors...
    GREY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    # color styles...
    HEADER = '\033[95m\033[1m'
    MSG = '\033[95m'
    QUESTION = '\033[93m\033[3m'
    COMMENT = '\033[96m'
    IMPLEMENTED = '\033[92m' + '[IMPLEMENTED] ' + '\033[96m'
    TODO = '\033[94m' + '[TO DO] ' + '\033[96m'
    WARNING = '\033[91m'
    ERROR = '\033[91m\033[1m'
    ENDC = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD = '\033[1m'
    ITALICS = '\033[3m'
    UNDERLINE = '\033[4m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def screen_clear():
    # for mac and linux(here, os.name is 'posix')
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        # for windows platfrom
        _ = os.system('cls')

### A. CONSTRUCTION OF RANDOM GAMES TO SOLVE ###


def generate_random_binary_array(N, K):
    # print(bcolors.IMPLEMENTED + '''
    # # ROUTINE:  generate_random_binary_array ''' + bcolors.ENDC)
    # PRE:      N = length of the 1xN binary array to be constructed
    #           K = number of ones within the 1xN binary array
    # POST:     A randomly constructed numpy array with K 1s and (N-K) zeros''' + bcolors.ENDC)

    if K <= 0:  # construct an ALL-ZEROS array
        randomBinaryArray = np.zeros(N)

    elif K >= N:  # construct an ALL-ONES array
        randomBinaryArray = np.ones(N)

    else:
        randomBinaryArray = np.array([1] * K + [0] * (N-K))
        np.random.shuffle(randomBinaryArray)

    return (randomBinaryArray)


def generate_winlose_game_without_pne(m, n, G01, G10, earliestColFor01, earliestRowFor10):

    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE:   generate_random_binary_array'''
          # PRE:       (m,n) = the dimensions of the game to construct
          #            (G10,G01) = numbers of (1,0)-elements and (0,1) elements in the game
          # POST:      Construct a mxn win-lose game randomly, so that:
          #             * There are G10 (1,0)-elements and G01 (0,1)-elements.
          #             * (1,1)-elements are forbidden
          #             * Each row possesses at least one (0,1)-element
          #             * Each column possesses at least one (1,0)-element
          #             * (0,1)-elements lie in columns from earliestColFor01 to n
          #             * 10-elements lie in rows from earliestRowFor10 to n
          # ERROR HANDLING:
          #   [EXITCODE =  0] SUCCESSFUL CREATION OF RANDOM WIN-LOSE GAME
          #   [EXITCODE = -1] WRONG PARAMETERS
          #   [EXITCODE = -2] INSUFFICIENT 10-ELEMENTS OR 01-ELEMENTS
          #   [EXITCODE = -3] TOO MANY 10-ELEMENTS OR 01-ELEMENTS
          #   [EXITCODE = -4] NOT ENOUGH SPACE TO POSITION 10-ELEMENTS, GIVEN POSITIONS OF 01-ELEMENTS
          #   [EXITCODE = -5] BAD LUCK, SOME COLUMN WITHIN 10-ELIGIBLE AREA IS ALREADY FILLED WITH 01-ELEMENTS'''
        #   + bcolors.ENDC)

    isIntegerFlag = True

    try:
        # try converting to integer
        int(m)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(n)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G10)
    except ValueError:
        isIntegerFlag = False

    try:
        # try converting to integer
        int(earliestColFor01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(earliestRowFor10)
    except ValueError:
        isIntegerFlag = False

    if not isIntegerFlag or np.amin([m, n]) < 2 or np.amax([m, n]) > maxNumberOfActions or m > n or np.amin([earliestRowFor10, earliestColFor01]) < 0 or (earliestRowFor10 > m-1) or (earliestColFor01 > n-1):
        # WRONG INPUT PARAMETERS
        print(bcolors.ERROR +
              "ERROR MESSAGE GEN 1: wrong input parameters" + bcolors.ENDC)
        return (-1, np.zeros([maxNumberOfActions, maxNumberOfActions]), np.zeros([maxNumberOfActions, maxNumberOfActions]))

    # initialization of the two payoff matrices...
    R = np.zeros([m, n])
    C = np.zeros([m, n])

    if (G10 < n or G01 < m):
        print(bcolors.ERROR + "ERROR MESSAGE GEN 2: NOT ENOUGH 10-elements and/or 01-elements: G10 =",
              G10, " < n =", n, "? G01 = ", G01, "< m =", m, "?" + bcolors.ENDC)
        return (-2, R, C)

    if G10 > (m-earliestRowFor10)*n or G01 > m*(n-earliestColFor01) or G01+G10 > m*n - earliestRowFor10*earliestColFor01:
        print(bcolors.ERROR +
              "ERROR MESSAGE GEN 3: TOO MANY 10-elements and/or 01-elements:" + bcolors.ENDC)
        print("\tG10 =", G10, "> (m-earliestRowFor10)*n =",
              (m-earliestRowFor10)*n, "?")
        print("\tG01 =", G01, "> m*(n-earliestColFor01) =",
              m*(n-earliestColFor01), "?")
        print("\tG01+G10 =", G01+G10, "> m*n - earliestRowFor10*earliestColFor01 =",
              m*n - earliestRowFor10*earliestColFor01, "?")
        return (-3, R, C)

    # choose the random positions for 01-elements, within the eligible area of the bimatrix...
    # eligible area for 01-elements: rows = 0,...,m-1, columns = earliestColFor01,...,n-1

    # STEP 1: choose m 01-elements, one per row, within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix.

    # all cells in bimatrix are currently 00-elements
    numEligibleCellsFor01 = m * (n - earliestColFor01)

    ArrayForOne01PerRow = np.zeros(numEligibleCellsFor01)
    for i in range(m):
        random_j = np.random.randint(earliestColFor01, n)
        position = (n-earliestColFor01)*i + random_j - (earliestColFor01)
        ArrayForOne01PerRow[position] = 1

    # STEP 2: choose G01 – m 01-elements within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix
    # differently from those cells chosen in STEP 1.
    binaryArrayFor01s = generate_random_binary_array(
        numEligibleCellsFor01 - m, G01 - m)

    # Position ALL the 01-elements within the eligible area of the bimatrix...
    for i in range(m):
        for j in range(earliestColFor01, n):
            position = (n-earliestColFor01)*i + j - (earliestColFor01)
            if ArrayForOne01PerRow[position] == 1:
                # insert this enforced 10-element in binArrayFor01s

                if position <= 0:  # checking cell (0,earliestColFor01)...
                    binaryArrayFor01sPrefix = np.array([])
                else:
                    binaryArrayFor01sPrefix = binaryArrayFor01s[0:position]

                if position >= numEligibleCellsFor01:  # checking cell (m,n)...
                    binaryArrayFor01sSuffix = np.array([])
                else:
                    binaryArrayFor01sSuffix = binaryArrayFor01s[position:]

                binaryArrayFor01s = np.concatenate(
                    (binaryArrayFor01sPrefix, np.array([1]), binaryArrayFor01sSuffix), axis=None)

            # print("next position to check for 01-element:",position,"related to the cell [",i,j,"].")
            if binaryArrayFor01s[position] == 1:
                C[i, j] = 1

    # STEP 3: choose n 10-elements, one per column, within the eligible area [earliestRowFor10s:m]x[0:n] of the bimatrix. They should be different from those cells chosen in STEPS 1+2

    # all cells in bimatrix are currently 00-elements
    numEligibleCellsFor10 = (m - earliestRowFor10) * n

    # Count only the (0,0)-elements within eligible area of the bimatrix for 10-elements...
    # eligible area for 10-elements: rows = earliestRowFor10,...,m-1, columns = 0,...,n-1
    numFreeEligibleCellsFor10 = 0

    ArrayForOne10PerCol = np.zeros(numEligibleCellsFor10)

    # Count the non-01-elements within the eligible area of the bimatrix for 10-elements
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            if C[i, j] == 0:
                numFreeEligibleCellsFor10 += 1

    # print("Actual number for eligible cells for 10-elements: numEligibleCellsFor10 = ",numFreeEligibleCellsFor10)
    if numFreeEligibleCellsFor10 < G10:
        print(bcolors.ERROR + "ERROR MESSAGE GEN 4: Not enough space to position all the 10-elements within the selected block of the bimatrix and the random position of the 01-elements" + bcolors.ENDC)
        return (-4, np.zeros([m, n]), np.zeros([m, n]))

    # choose the n random positions of 10-elements, one per column, in positions which are NOT already
    # 01-elements, within the 10-eligible area of the bimatrix
    for j in range(n):
        if sum(C[earliestRowFor10:, j:j+1]) == n - earliestRowFor10:
            # the j-th row of the 10-eligible area in the bimatrix is already filled with 01-elements
            print(bcolors.ERROR + "ERROR MESSAGE 5: Bad luck, column", j,
                  "of the bimatrix is already filled with 01-elements." + bcolors.ENDC)
            return (-5, np.zeros([m, n]), np.zeros([m, n]))

        Flag_EmptyCellDiscovered = False
        while not Flag_EmptyCellDiscovered:
            random_i = np.random.randint(earliestRowFor10, m)
            if C[random_i, j] == 0:
                Flag_EmptyCellDiscovered = True
        position = n * (random_i - earliestRowFor10) + j
        ArrayForOne10PerCol[position] = 1

    # choose the remaining G10-n random positions for 10-elements, in positions which are NOT already
    # used by 01-elements or other (the necessary) 10-elements, within the eligible area of the bimatrix
    binaryArrayFor10s = generate_random_binary_array(
        numFreeEligibleCellsFor10-n, G10-n)
    # expand the binaryArrayFor10s to cover the entire eligible area for 10-elements, so that
    # all cells which are already 01-elements get 0-value and all cells with a necessary 10-element
    # get 1-value.

    # print("INITIAL length of binaryArrayFor10s is",len(binaryArrayFor10s))
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            position = n*(i-earliestRowFor10) + j
            if C[i, j] == 1:
                # A 01-element was discovered. Insert a ZERO in binaryArrayFor10s, at POSITION,
                # on behalf of cell (i,j)...

                # print("01-element discovered at position (",i,",",j,"). Inserting an additional ZERO at position ",position)

                if position <= 0:  # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                # checking cell (m,n)...
                if position >= len(binaryArrayFor10s):
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]

                binaryArrayFor10s = np.concatenate(
                    (binaryArrayFor10sPrefix, np.array([0]), binaryArrayFor10sSuffix), axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

            elif ArrayForOne10PerCol[position] == 1:
                # A necessary 10-element discovered. Insert a new ONE in binaryArrayFor10s, at POSITION,
                # on behalf of cell (i,j)...
                # print("A necessary 10-element was discovered at position (",i,",",j,"). Inserting an additional ONE at position ",position)

                if position <= 0:  # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                # checking cell (m,n)...
                if position >= len(binaryArrayFor10s):
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]

                binaryArrayFor10s = np.concatenate(
                    (binaryArrayFor10sPrefix, np.array([1]), binaryArrayFor10sSuffix), axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

    # print("ACTUAL length of binaryArrayFor10s is",len(binaryArrayFor10s))

    # Insert the G10 10-elements in the appropriate positions of the bimatrix...
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            position = n*(i-earliestRowFor10) + j
            # print("next position to check for 10-element:",position,"related to the cell [",i,j,"], with C-value = ",C[i,j],"and binaryArrayFor10s-value = ",binaryArrayFor10s[position])
            if binaryArrayFor10s[position] == 1:
                R[i, j] = 1

    return (0, R, C)

### B. MANAGEMENT OF BIMATRICES ###


def drawLine(lineLength, lineCharacter):

    LINE = '\t'
    consecutiveLineCharacters = lineCharacter
    for i in range(lineLength):
        consecutiveLineCharacters = consecutiveLineCharacters + lineCharacter
    LINE = '\t' + consecutiveLineCharacters
    return (LINE)


def drawBimatrix(m, n, R, C):

    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE:    drawBimatrix ''' + bcolors.ENDC)
    # PRE:        Dimensions and payoff matrices of a win-lose bimatrix game
    # POST:       The bimatrix game, with RED for 10-elements, GREEN for 01-elements, and BLUE for 11-elements
    # ''' + bcolors.ENDC)

    for i in range(m):
        # PRINTING ROW i...
        if i == 0:
            print(EQLINE)
        else:
            print(MINUSLINE)

        printRowString = ''

        for j in range(n):
            # PRINTING CELL (i,j)...
            if R[i, j] == 1:
                if C[i, j] == 1:
                    CellString = bcolors.CYAN + "("
                else:
                    CellString = bcolors.RED + "("
            elif C[i, j] == 1:
                CellString = bcolors.GREEN + "("
            else:
                CellString = "("

            CellString += str(int(R[i, j])) + "," + \
                str(int(C[i, j])) + ")" + bcolors.ENDC
            if printRowString == '':
                printRowString = '\t[ ' + CellString
            else:
                printRowString = printRowString + ' | ' + CellString

        printRowString = printRowString + ' ]'
        print(printRowString)

    print(EQLINE)

# ALGORITHMS FOR SOLVING BIMATRIX GAMES

# ALG0: Solver for ZERO-SUM games...


def checkForPNE(m, n, R, C):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: checkForPNE '''
    #       + bcolors.ENDC)
    # PRE:    Two mxn payoff matrices R,C, with real values (not necessarily in [0,1])
    # METHOD: Checking for the existence of a pure Nash equilibrium (PNE).
    # POST:   (0,0) if no pure NE exists for (R, C), or else a pair of actions (i, j) that constitute a pure NE.
    # ''' + bcolors.ENDC)

    poss_row = []
    poss_col = []
    max_of_each_row = np.max(R, axis=0)  # finds the max value of each  row
    max_of_each_col = np.max(C, axis=1)  # finds the max value of each  column

    # finds the indexes of all cells with the max row value
    for i in range(m):
        for j in range(n):
            if R[i][j] == max_of_each_row[i]:
                poss_row.append((i, j))

    # finds the indexes of all cells with the max column value
    for j in range(n):
        for i in range(m):
            if C[i][j] == max_of_each_col[i]:
                poss_col.append((i, j))

    # checks if any cells have max value for both row and column, if yes it returns the first it finds as a PNE
    for i in range(0, len(poss_row)):
        for j in range(0, len(poss_col)):
            if poss_row[i] == poss_col[j]:
                return (i, j)

    return (0, 0)


def solveZeroSumGame(m, n, A):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: solveZeroSumGame ''' + bcolors.ENDC)
    # PRE:  An arbirary payoff matrix A, with real values (not necessarily in [0,1])
    # METHOD:
    #    Construct the LP describing the MAGNASARIAN-STONE formulation for the 0_SUM case: R = A, C = -A
    #    [0SUMLP]
    #    minmize          1*r           + 1*c +  np.zeros(m).reshape([1,m]@x + np.zeros(n).reshape([1,n]@y
    #      s.t.
    #           -np.ones(m)*r + np.zeros(m)*c +            np.zeros([m,m])@x +                          R@y <= np.zeros(m),
    #           np.zeros(n)*r -  np.ones(n)*c +                          C'x +            np.zeros([n,n])@y <= np.zeros(n),
    #                     0*r             0*c +  np.ones(m).reshape([1,m])@x + np.zeros(n).reshape([1,n])@y = 1,
    #                     0*r             0*c + np.zeros(m).reshape([1,m])@x +  np.ones(n).reshape([1,n])@y = 1,
    #                                                                   np.zeros(m) <= x,              np.zeros(n) <= y
    #
    # vector of unknowns is a (1+1+m+n)x1 array: chi = [ r, c, x^T , y^T ],
    # where r is ROW's payoff and c is col's payoff, wrt the profile (x,y).
    #

    c = np.block([np.ones(2), np.zeros(m+n)])

    # 1x(m+n+2) array...
    Coefficients_a = np.block([(-1)*np.ones(m), np.zeros(n), np.array([0, 0])])
    # 1x(m+n+2) array...
    Coefficients_b = np.block([np.zeros(m), (-1)*np.ones(n), np.array([0, 0])])
    # mx(m+n+2) array...
    Coefficients_x = (np.block([np.zeros(
        [m, m]), (-1)*A, np.ones(m).reshape([m, 1]), np.zeros(m).reshape([m, 1])])).transpose()
    Coefficients_y = (np.block([A.transpose(), np.zeros([n, n]), np.zeros(n).reshape(
        [n, 1]), np.ones(n).reshape([n, 1])])).transpose()  # nx(m+n+2) array...

    SIGMA0 = (np.block([Coefficients_a.reshape(
        [m+n+2, 1]), Coefficients_b.reshape([m+n+2, 1]), Coefficients_x, Coefficients_y]))

    SIGMA0_ub = SIGMA0[0:m+n, :]
    Constants_vector_ub = np.zeros(m+n)

    SIGMA0_eq = SIGMA0[m+n:m+n+2, :]
    Constants_vector_eq = np.ones(2)

    # variable bounds
    Var_bounds = [(None, None), (None, None)]
    for i in range(m+n):
        Var_bounds.append((0, None))  # type: ignore

    zero_sum_res = linprog(c,
                           A_ub=SIGMA0_ub,
                           b_ub=Constants_vector_ub,
                           A_eq=SIGMA0_eq,
                           b_eq=Constants_vector_eq,
                           bounds=Var_bounds,
                           method='highs', callback=None, options=None, x0=None)

    chi = zero_sum_res.x

    x = chi[2:m+2]
    y = chi[m+2:m+n+2]

    return (x, y)


def removeStrictlyDominatedStrategies(m, n, R, C):

    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: removeStrictlyDominatedStrategies ''' + bcolors.ENDC)
    # PRE:    A win-lose bimatrix game, described by the two payoff matrices, with payoff values in {0,1}.
    # POST:   The subgame constructed by having all strictly dominated actions removed.
    #          ''' + bcolors.ENDC)

    # Remove strictly dominated rows
    check = 0
    dominated_rows = set()
    dominated_cols = set()
    while check == 0:
        num_removals = 0

        for i in range(m):
            if i not in dominated_rows and all(R[i][k] <= R[j][k] for k in range(n) for j in range(m) if j != i):
                if not all(R[i][k] == R[j][k] for k in range(n) for j in range(m) if j != i):
                    # if the two rows aren't exactly the same (to avoid erasing weakly dominated rows)
                    dominated_rows.add(i)
                    num_removals += 1
                    for z in range(n):
                        R[i][z] = 0
                        C[i][z] = 0

        # Remove strictly dominated columns
        for j in range(n):
            if j not in dominated_cols and all(C[i][j] <= C[i][k] for k in range(n) for i in range(m)):
                if not all(C[i][j] == C[i][k] for k in range(n) for i in range(m)):
                   # if the two columns aren't exactly the same (to avoid erasing weakly dominated columns)
                    dominated_cols.add(j)
                    num_removals += 1
                    for z in range(m):
                        R[z][j] = 0
                        C[z][j] = 0

        # Check if any row or column was deleted in this iteration, if not then reduction is complete
        if num_removals == 0:
            check = 1

    # delete strictly dominated rows
    reduced_R = np.delete(R, list(dominated_rows), axis=0)
    reduced_C = np.delete(C, list(dominated_rows), axis=0)
    reduced_m = reduced_R.shape[0]

    # delete strictly dominated columns
    reduced_C = np.delete(reduced_C, list(dominated_cols), axis=1)
    reduced_R = np.delete(reduced_R, list(dominated_cols), axis=1)
    reduced_n = reduced_C.shape[1]

    return reduced_m, reduced_n, reduced_R, reduced_C, dominated_rows, dominated_cols


def interpretReducedStrategiesForOriginalGame(reduced_x, reduced_y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE:    interpretReducedStrategiesForOriginalGame ''' + bcolors.ENDC)
    # PRE:        A profile of strategies (reduced_x,reduced_y) for the reduced
    #             game (reduced_R,reduced_C), without (0,*)-rows or (*,0)-columns.
    # POST:       The corresponding profile for the original game (R,C).
    # ''' + bcolors.ENDC)

    x = np.zeros(R.shape[0])
    y = np.zeros(C.shape[1])
    i = 0
    j = 0
    check = 0
    if (reduced_R.shape[0] == R.shape[0] and reduced_R.shape[1] == R.shape[1] and reduced_C.shape[0] == C.shape[0] and reduced_C.shape[1] == C.shape[1]):

        x = reduced_x
        y = reduced_y

    else:
        if (len(reduced_x) == R.shape[0]):
            x = reduced_x
        else:
            for i in range(len(x)):
                if i not in dominated_rows:

                    if check != 0:
                        x[i] = reduced_x[i - check]
                    else:
                        x[i] = reduced_x[i]
                else:
                    x[i] = 0
                    check += 1

        if (len(reduced_y) == C.shape[1]):
            y = reduced_y
        else:
            check = 0
            for j in range(len(y)):
                if j not in dominated_cols:

                    if check != 0:
                        y[j] = reduced_y[j - check]
                    else:
                        y[j] = reduced_y[j]
                else:
                    y[j] = 0
                    check += 1
    return x, y


def computeApproximationGuarantees(m, n, R, C, x, y):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: computeApproximationGuarantees''' + bcolors.ENDC)
    # PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    #         A profile (x,y) of strategies for the two players.
    # POST:   The two NASH approximation guarantees, epsAPPROX and eps in [0,1].''' + bcolors.ENDC)

    # Compute the payoffs for player 1 and player 2
    payoff_1 = np.dot(x, np.dot(R, y))
    payoff_2 = np.dot(x, np.dot(C, y))

    # Compute the approximate Nash equilibrium (APPROX-NE) payoffs for player 1 and player 2
    approx_ne_payoff_1 = np.max(np.dot(R, y))
    approx_ne_payoff_2 = np.max(np.dot(C.T, x))

    epsAPPROXrow = approx_ne_payoff_1-payoff_1
    epsAPPROXcol = approx_ne_payoff_2-payoff_2

    # Compute the well-supported Nash equilibrium (WSNE) payoffs for player 1 and player 2
    wsne_payoff_1 = np.min(np.dot(R, y))
    wsne_payoff_2 = np.min(np.dot(C.T, x))

    approx_wsne_payoff1 = np.max(np.dot(R, y))
    approx_wsne_payoff2 = np.max(np.dot(C.T, x))

    wsepsAPPROXrow = approx_wsne_payoff1-wsne_payoff_1
    wsepsAPPROXcol = approx_wsne_payoff2-wsne_payoff_2

    # Compute the approximation guarantees
    epsAPPROX = max(epsAPPROXrow, epsAPPROXcol)
    epsWSNE = max(wsepsAPPROXrow, wsepsAPPROXcol)

    return epsAPPROX, epsWSNE


def approxNEConstructionDMP(m, n, R, C):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: approxNEConstructionDMP ''' + bcolors.ENDC)
    # PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    # POST:   A profile of strategies (x,y) produced by the DMP algorithm.''' + bcolors.ENDC)

    # Initialize strategies x and y
    x = np.full(R.shape[0], 0.)
    y = np.full(C.shape[1], 0.)
    initial_move = np.random.randint(m)

    x[initial_move] = 1.

    # Best response for column player
    ymax = np.argmax(np.dot(C.T, x))
    y[ymax] = 1.

    # Best response for row player
    xmax = np.argmax(np.dot(R, y))
    xnew = np.full(R.shape[0], 0.)
    xnew[xmax] = 1.
    x = (x + xnew)/2

    # Check if the game is zero-sum
    count = 0
    for i in range(m):
        for j in range(n):
            if R[i][j] == - C[i][j]:
                count = count+1
    if count == m*n:  # if the above condition is true for every cell
        x, y = solveZeroSumGame(m, n, R)

    # Compute the approximation guarantees
    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x, y)

    # Return the profile of strategies and the approximation guarantees
    return x, y, epsAPPROX, epsWSNE


def approxNEConstructionFPPBR(m, n, R, C):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: approxNEConstructionFP ''' + bcolors.ENDC)
    # PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    # POST:   A profile of strategies (x,y) produced by the FICTITIOUS PLAY algorithm (Pure Best Response version).''' + bcolors.ENDC)

    # checking if the game is zero-sum
    count = 0
    for i in range(m):
        for j in range(n):
            if R[i][j] == - C[i][j]:
                count = count+1
    if count == m*n:  # if the above condition is true for every cell
        χ, ψ = solveZeroSumGame(m, n, R)
        epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, χ, ψ)
        return (χ, ψ, epsAPPROX, epsWSNE)
    # if not zero sum then the algorithm starts
    # steps 1,2 , initializing the variables
    T = 10
    x = np.full(R.shape[0], 0.)
    x[0] = 1.
    y = np.full(C.shape[1], 0.)
    y[0] = 1.
    χ, ψ = x, y

    # step 3, calculating pure best responses
    for t in range(1, T-1, 1):
        # row player
        xmax = np.argmax(np.dot(R, ψ))
        x = np.full(R.shape[0], 0.)
        x[xmax] = 1.
        χ = 1/(t+1) * (t*χ + x)

        # column player
        ymax = np.argmax(np.dot(C.T, χ))
        y = np.full(C.shape[1], 0.)
        y[ymax] = 1.
        ψ = 1/(t+1) * (t*ψ + y)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, χ, ψ)

    return (χ, ψ, epsAPPROX, epsWSNE)


def approxNEConstructionFPUNI(m, n, R, C):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: approxNEConstructionFP ''' + bcolors.ENDC)
    # PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    # POST:   A profile of strategies (x,y) produced by the FICTITIOUS PLAY algorithm (Uniform version).''' + bcolors.ENDC)

    count = 0
    for i in range(m):
        for j in range(n):
            if R[i][j] == - C[i][j]:
                count = count+1
    if count == m*n:  # if the above condition is true for every cell
        x, y = solveZeroSumGame(m, n, R)
        epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x, y)
        return (x, y, epsAPPROX, epsWSNE)

    # steps 1,2 , initializing the variables
    T = 10
    x = np.full(R.shape[0], 1/m)
    y = np.full(C.shape[1], 1/n)
    χ, ψ = x, y

    # step 3, calculating pure best responses
    for t in range(1, T-1, 1):
        pbr_list = set()

        # row player
        xmax = np.max(np.dot(R, ψ))
        for i in range(m):
            if (np.dot(R, ψ))[i] == xmax:
                pbr_list.add(i)

        random_move = random.choice(list(pbr_list))
        x = np.full(R.shape[0], 0)
        x[random_move] = 1
        χ = 1/(t+1) * (t*χ + x)

        pbr_list = set()
        # column player
        ymax = np.max(np.dot(C.T, χ))
        for i in range(n):
            if (np.dot(C.T, χ))[i] == ymax:
                pbr_list.add(i)

        random_move = random.choice(list(pbr_list))
        y = np.full(C.shape[1], 0)
        y[random_move] = 1
        ψ = 1/(t+1) * (t*ψ + y)

    # Check if the game is zero-sum
    count = 0
    for i in range(m):
        for j in range(n):
            if R[i][j] == - C[i][j]:
                count = count+1
    if count == m*n:  # if the above condition is true for every cell
        x, y = solveZeroSumGame(m, n, R)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, χ, ψ)

    return (χ, ψ, epsAPPROX, epsWSNE)


def approxNEConstructionDEL(m, n, R, C):
    # print(bcolors.IMPLEMENTED + '''
    # ROUTINE: approxNEConstructionDEL '''+bcolors.ENDC)
    # PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    # POST:   A profile of strategies (x,y) produced by the DEL algorithm.''' + bcolors.ENDC)

    # Check if the game is zero-sum
    count = 0
    for i in range(m):
        for j in range(n):
            if R[i][j] == - C[i][j]:
                count = count+1
    if count == m*n:  # if the above condition is true for every cell
        x, y = solveZeroSumGame(m, n, R)
        epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x, y)
        return (x, y, epsAPPROX, epsWSNE)

    # if not zero sum then the algorithm starts
    # Calculation of MAXMIN strategies and guaranteed payoffs for both players
    xrow, yrow = solveZeroSumGame(m, n, R)  # MAXMIN strategy for row player
    # Guaranteed payoff for row player
    Vrow = np.dot(np.transpose(xrow), np.dot(R, yrow))

    # MAXMIN strategy for column player
    xcol, ycol = solveZeroSumGame(m, n, -C)

    # Guaranteed payoff for column player
    Vcol = np.dot(np.transpose(xcol), np.dot(C, ycol))

    # Compare the guaranteed payoffs and transpose matrices if necessary
    if Vrow < Vcol:
        temp_xrow, temp_yrow, temp_Vrow = xrow, yrow, Vrow
        xrow, yrow, Vrow = xcol, ycol, Vcol
        xcol, ycol, Vcol = temp_xrow, temp_yrow,  temp_Vrow

    # SCENARIO 1: No player has a guaranteed payoff greater than 2/3
    if Vrow <= 2/3:
        x = xcol
        y = yrow

    # SCENARIO 2: Guaranteed utility > 2/3 for row player, but column player finds no utility > 2/3 against xrow
    elif np.max(np.dot(xrow.T, C)) <= 2/3:
        x = xrow
        y = yrow

    else:
        # SCENARIO 3: Row player is guaranteed utility > 2/3 and column player finds utility > 2/3 against xrow
        j = np.argmax(np.dot(xrow.T, C))
        i = np.argmax(np.logical_and(R[:, j] > 1/3, C[:, j] > 1/3))
        x = np.zeros(m)
        x[i] = 1
        y = np.zeros(n)
        y[j] = 1

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x, y)

    return (x, y, epsAPPROX, epsWSNE)

### C. GET INPUT PARAMETERS ###


def determineGameDimensions():

    m = 0
    while m < 2 or m > maxNumberOfActions:
        RowActionsString = input(bcolors.QUESTION + 'Determine the size 2 =< m =< ' + str(
            maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if RowActionsString.isdigit():
            m = int(RowActionsString)
            print(bcolors.MSG + "You provided the value m =" +
                  str(m) + bcolors.ENDC)
            if m < 2 or m > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 1: Only positive integers between 2 and ' + str(
                    maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            m = 0
            print(bcolors.ERROR + 'ERROR INPUT 2: Only integer values between 2 and ' +
                  str(maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)

    n = 0
    while n < 2 or n > maxNumberOfActions:
        ColActionsString = input(bcolors.QUESTION + 'Determine the size 1 =< n =< ' + str(
            maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if ColActionsString.isdigit():
            n = int(ColActionsString)
            print(bcolors.MSG + "You provided the value n =" +
                  str(n) + bcolors.ENDC)
            if n < 2 or n > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 3: Only positive integers between 2 and ' + str(
                    maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            n = 0
            print(bcolors.ERROR + 'ERROR INPUT 4: Only integer values between 2 and ' +
                  str(maxNumberOfActions) + ' are allowable values for n. Try again...' + bcolors.ENDC)

    return (m, n)


def determineNumRandomGamesToSolve():

    numOfRandomGamesToSolve = 0
    while numOfRandomGamesToSolve < 1 or numOfRandomGamesToSolve > 10000:
        numOfRandomGamesToSolveString = input(
            bcolors.QUESTION + 'Determine the number of random games to solve: ' + bcolors.ENDC)
        if numOfRandomGamesToSolveString.isdigit():
            numOfRandomGamesToSolve = int(numOfRandomGamesToSolveString)
            print(bcolors.MSG + "You requested to construct and solve " +
                  str(numOfRandomGamesToSolve) + " random games to solve." + bcolors.ENDC)
            if n < 2 or m > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 5: Only positive integers between 1 and ' + str(
                    maxNumOfRandomGamesToSolve) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            numOfRandomGamesToSolve = 0
            print(bcolors.ERROR + 'ERROR INPUT 6: Only integer values between 2 and ' +
                  str(maxNumOfRandomGamesToSolve) + ' are allowable values for n. Try again...' + bcolors.ENDC)

    return (numOfRandomGamesToSolve)


def determineNumGoodCellsForPlayers(m, n):

    G10 = 0
    G01 = 0

    while G10 < 1 or G10 > m*n:
        G10String = input(
            bcolors.QUESTION + 'Determine the number of (1,0)-elements in the bimatrix: ' + bcolors.ENDC)
        if G10String.isdigit():
            G10 = int(G10String)
            print(bcolors.MSG + "You provided the value G10 =" +
                  str(G10) + bcolors.ENDC)
            if G10 < 0 or G10 > m*n:
                print(bcolors.ERROR + 'ERROR INPUT 7: Only non-negative integers up to ' +
                      str(m*n) + ' are allowable values for G10. Try again...' + bcolors.ENDC)
        else:
            G10 = 0
            print(bcolors.ERROR + 'ERROR INPUT 8: Only integer values up to ' +
                  str(m*n) + ' are allowable values for G10. Try again...' + bcolors.ENDC)

    while G01 < 1 or G01 > m*n:
        G01String = input(
            bcolors.QUESTION + 'Determine the number of (0,1)-elements in the bimatrix: ' + bcolors.ENDC)
        if G01String.isdigit():
            G01 = int(G01String)
            print(bcolors.MSG + "You provided the value G01 =" +
                  str(G01) + bcolors.ENDC)
            if G01 < 0 or G01 > m*n:
                print(bcolors.ERROR + 'ERROR INPUT 9: Only non-negative integers up to ' +
                      str(m*n) + ' are allowable values for G01. Try again...' + bcolors.ENDC)
        else:
            G01 = 0
            print(bcolors.ERROR + 'ERROR INPUT 10: Only integer values up to ' +
                  str(m*n) + ' are allowable values for G01. Try again...' + bcolors.ENDC)

    return (G10, G01)

### D. PREAMBLE FOR LAB-2 ###


def print_LAB2_preamble():
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
                        CEID-NE509 (2022-3) / LAB-2""")
    print(MINUSLINE + """
        STUDENT NAME:            Ion Bournakas 
        STUDENT AM:              1075475 
        JOINT WORK WITH:         Maria-Vasiliki Petropoulou 1072540""")
    print(MINUSLINE + bcolors.ENDC)

    input("Press ENTER to continue...")
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
        LAB-2 OBJECTIVE: EXPERIMENTATION WITH WIN-LOSE BIMATRIX GAMES\n""" + MINUSLINE + """  
        1.      GENERATOR OF INSTANCES: Construct rando win-lose games 
        with given densities for non-(0,0)-elements, and without pure 
        Nash equilibria.                          (PROVIDED IN TEMPLATE)

        2.      BIMATRIX CLEANUP: Remove all STRICTLY DOMINATED actions 
        for the players, ie, all (0,*)-rows, and all (*,0)-columns from 
        the bimatrix.                                (TO BE IMPLEMENTED) 

        3.      Implementation of elementary algorithms for constructing
        strategy profiles that are then tested for their quality as 
        ApproxNE, or WSNE points.                    (TO BE IMPLEMENTED)

        4.      EXPERIMENTAL EVALUATION: Construct P random games, for
        some user-determined input parameter P, and solve each of them 
        with each of the elementary algorithms. Record the observed 
        approximation guarantees (both epsAPPROXNE and epsWSNE) for the 
        provided strategy profiles.                  (TO BE IMPLEMENTED)
        
        5.      VISUALIZATION OF RESULTS: Show the performances of the 
        algorithms (as approxNE or WSNE constructors), by constructin 
        the appropriate histograms (bucketing the observewd approximation 
        guarantees at one-decimal-point precision).  (TO BE IMPLEMENTED)
    """ + MINUSLINE + bcolors.ENDC)

    input("Press ENTER to continue...")

### MAIN PROGRAM FOR LAB-2 ###


LINELENGTH = 80
EQLINE = drawLine(LINELENGTH, '=')
MINUSLINE = drawLine(LINELENGTH, '-')
PLUSLINE = drawLine(LINELENGTH, '+')

print_LAB2_preamble()

screen_clear()
cnt=1
numOfRandomGamesToSolve = 0
print("Would you like to load the R and C matrices from a file?")

while True:
    choice = input("Enter 'y' for yes or 'n' for no: ")
    
   
    if choice == "y":
        # detect if there are already R and C files in the directory
        if os.path.isfile("R.out") and os.path.isfile("C.out"):
            print("R.out and C.out files detected in path. Would you like to load them?")
            choice = input("Enter 'y' for yes or 'n' for no: ")
            if choice == "y":
                R = np.genfromtxt("R.out", delimiter=",", skip_header=0)
                C = np.genfromtxt("C.out", delimiter=",", skip_header=0)
                m = R.shape[0]
                n = R.shape[1]
                if(m != C.shape[0] or n != C.shape[1]):
                    print("The dimensions of the R and C matrices do not match. Please select another pair of files.")
                else:
                    break
            else:
                print("No or incorrect input detected. So negative answer assumed.")
                
        
        # ask the user what is the file name for the R matrix
        while True:
            Rfilename = input(
                "Enter the name of the file containing the R matrix: ")
            # ask the user what is the file name for the C matrix
            Cfilename = input(
                "Enter the name of the file containing the C matrix: ")
            # add the .out extension to the file names if not already present
            if not Rfilename.endswith('.out'):
                Rfilename += '.out'
            if not Cfilename.endswith('.out'):
                Cfilename += '.out'
            # check if the files exist
            if os.path.isfile(Rfilename) and os.path.isfile(Cfilename):
                break
            else:
                print("The files you provided do not exist. Try again...")
        # load the R and C matrices from the files
        R = np.genfromtxt(Rfilename, delimiter=",", skip_header=0)
        C = np.genfromtxt(Cfilename, delimiter=",", skip_header=0)
        m = R.shape[0]
        n = R.shape[1]
        if(m != C.shape[0] or n != C.shape[1]):
            print("The dimensions of the R and C matrices do not match. Please select another pair of files.")
        else:
            break
    elif choice == "n":
        print("Generating random R and C matrices based on your specifications:")
        maxNumOfRandomGamesToSolve = 10000

        maxNumberOfActions = 20

        m, n = determineGameDimensions()

        G10, G01 = determineNumGoodCellsForPlayers(m, n)

        numOfRandomGamesToSolve = determineNumRandomGamesToSolve()

        earliestColFor01 = 0
        earliestRowFor10 = 0
        break
    else:

        print("Invalid input. Please enter 'y' or 'n'.")



games = 0
flag = 0
flag1 = 0
choice1 = 0
DMPtotalTime = 0
DELtotalTime = 0
FPPBRtotalTime = 0
FPUNtotalTime = 0


if (numOfRandomGamesToSolve > 1):
    print("Mass experiment detected. Skipping save to path question and limiting outputs for easier understanding"
          " of results.")
    registerE_DMP= {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerW_DMP = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerE_DEL = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerW_DEL = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerE_FPPBR = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerW_FPPBR = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerE_FPUN = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    registerW_FPUN = {
        "[0,0.1)": 0,
        "[0.1,0.2)": 0,
        "[0.2,0.3)": 0,
        "[0.3,0.4)": 0,
        "[0.4,0.5)": 0,
        "[0.5,0.6)": 0,
        "[0.6,0.7)": 0,
        "[0.7,0.8)": 0,
        "[0.8,0.9)": 0,
        "[0.9,1.0]": 0
    }
    DMPepsAPPROX_list = []
    DELepsAPPROX_list = []
    FPPBRepsAPPROX_list = []
    FPUNepsAPPROX_list = []
    R_list = []
    C_list = []
    flag = 1
if (numOfRandomGamesToSolve == 0):
    flag1 = 1
    numOfRandomGamesToSolve = 1

while games < numOfRandomGamesToSolve:
    games += 1
    if (flag1 == 0):
        EXITCODE = -5
        numOfAttempts = 0
        # TRY GETTING A NEW RANDOM GAME
        # REPEAT UNTIL EXITCODE = 0, ie, a valid game was constructed.
        # NOTE: EXITCODE in {-1,-2,-3} indicates invalid parameters and exits the program)
        while EXITCODE < 0:
            # EXIT CODE = -4 ==> No problem with parameters, only BAD LUCK, TOO MANY 01-elements within 10-eligible area
            # EXIT CODE = -5 ==> No problem with parameters, only BAD LUCK, ALL-01 column exists within 10-eligible area
            numOfAttempts += 1
            if (flag == 0):
                print("Attempt #" + str(numOfAttempts) +
                  " to construct a random game...")
            EXITCODE, R, C = generate_winlose_game_without_pne(
                m, n, G01, G10, earliestColFor01, earliestRowFor10)
            if EXITCODE in [-1, -2, -3]:
                print(bcolors.ERROR + "ERROR MESSAGE MAIN 1: Invalid parameters were provided for the construction of the random game." + bcolors.ENDC)
                exit()
    if (flag == 0 and flag1 == 0):
        print("Would you like to save the R and C matrices to a file?")
        choice = input("Enter 'y' for yes or 'n' for no: ")
        if choice == "y":
            while True:
                # ask the user what is the file name for the R matrix
                Rfilename = input(
                    "Enter the name of the file to save the R matrix: ")
                # ask the user what is the file name for the C matrix
                Cfilename = input(
                    "Enter the name of the file to save the C matrix: ")
                # check if the user put valid file names
                if Rfilename != "" and Cfilename != "" and Rfilename != Cfilename:
                    break

                else:
                    print("Invalid file names. Try again...")
            # add the .out extension to the file names if not already present
            if not Rfilename.endswith('.out'):
                Rfilename += '.out'
            if not Cfilename.endswith('.out'):
                Cfilename += '.out'
            # save the R and C matrices to the files
            np.savetxt(Rfilename, R, delimiter=',', fmt='%1.4e')
            np.savetxt(Cfilename, C, delimiter=',', fmt='%1.4e')
        else:
            print("No or incorrect input detected. So negative answer assumed.")

        drawBimatrix(m, n, R, C)
    if (flag1 == 1):
        drawBimatrix(m, n, R, C)

    # SEEKING FOR PNE IN THE GAME (R,C)...
    (i, j) = checkForPNE(m, n, R, C)
    if (i, j) != (0, 0):
        if(flag==0):
            print(bcolors.MSG + "A pure NE (", i, ",", j,
              ") was discovered for (R,C)." + bcolors.ENDC)
        exit()
    else:
        if(flag==0):
            print(bcolors.MSG + "No pure NE exists for (R,C). Looking for an approximate NE point..." + bcolors.ENDC)
    reduced_m, reduced_n, reduced_R, reduced_C, dominated_rows, dominated_cols = removeStrictlyDominatedStrategies(
        m, n, R, C)
    if (flag == 0 or flag1 == 1):
        print(bcolors.MSG +
              "Reduced bimatrix, after removal of strictly dominated actions:")
        drawBimatrix(reduced_m, reduced_n, reduced_R, reduced_C)
    if (choice1 == 0):
        while True:
            # Prompt the user to choose which algorithm to run
            print("Which algorithm would you like to run?")
            print("1. DMP Algorithm")
            print("2. DEL Algorithm")
            print("3. FPPBR Algorithm")
            print("4. FPUNIFORM Algorithm")
            print("5. All of the above")
            choice1 = input("Enter your choice (1/2/3/4/5): ")
            if choice1 in ["1", "2", "3", "4", "5"]:
                break
            else:
                print("Invalid input. Try again...")

    # Run the chosen algorithm(s)
    if choice1 == "1":
        # EXECUTING DMP ALGORITHM...
        start_time = time.time()
        x, y, DMPepsAPPROX, DMPepsWSNE = approxNEConstructionDMP(
            reduced_m, reduced_n, reduced_R, reduced_C)
        DMPx, DMPy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)

        end_time = time.time()
        elapsed_time = end_time - start_time
        DMPtotalTime += elapsed_time
        # if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DMP:")
        print(MINUSLINE)
        print("\tDMPx =", DMPx, "\n\tDMPy =", DMPy)
        print("\tDMPepsAPPROX =", DMPepsAPPROX,
            ".\tDMPepsWSNE =", DMPepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
        if(flag==1):
            DMPepsAPPROX_list.append(DMPepsAPPROX)
            R_list.append(R)
            C_list.append(C)

            if 0 <= DMPepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= DMPepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= DMPepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= DMPepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= DMPepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= DMPepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= DMPepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= DMPepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= DMPepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= DMPepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_DMP[interval] += 1
            if 0 <= DMPepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= DMPepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= DMPepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= DMPepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= DMPepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= DMPepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= DMPepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= DMPepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= DMPepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= DMPepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_DMP[interval1] += 1
            if games == numOfRandomGamesToSolve:
                print("\tTotal Time for DMP =", DMPtotalTime, "seconds",
                    ".\tAvg Time for DMP =", DMPtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                
                with open("PkDMPApproxNEHistogram.out", "w") as f:
                    for key in registerE_DMP.keys():
                        f.write("%s,%s\n" % (key, registerE_DMP[key]))
                with open("PkDMPWSNEHistogram.out", "w") as f:
                    for key in registerW_DMP.keys():
                        f.write("%s,%s\n" % (key, registerW_DMP[key]))
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(DMPepsAPPROX_list)
                max1_index = DMPepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                DMPepsAPPROX_list.remove(max1)
                max2 = max(DMPepsAPPROX_list)
                max2_index = DMPepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkDMPApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDMPApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkDMPApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDMPApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_DMP.keys(), registerE_DMP.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDMPApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_DMP.keys(), registerW_DMP.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDMPWSNEHistogram.jpg')
                plt.clf()

    elif choice1 == "2":
        # EXECUTING DEL ALGORITHM...
        start_time = time.time()
        x, y, DELepsAPPROX, DELepsWSNE = approxNEConstructionDEL(
            reduced_m, reduced_n, reduced_R, reduced_C)
        DELx, DELy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        DELtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DEL:")
        print(MINUSLINE)
        print("\tDELx =", DELx, "\n\tDELy =", DELy)
        print("\tDELepsAPPROX =", DELepsAPPROX,
            ".\tDELepsWSNE =", DELepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)

        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            DELepsAPPROX_list.append(DELepsAPPROX)
            R_list.append(R)
            C_list.append(C)

            if 0 <= DELepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= DELepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= DELepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= DELepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= DELepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= DELepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= DELepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= DELepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= DELepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= DELepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_DEL[interval] += 1

            if 0 <= DELepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= DELepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= DELepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= DELepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= DELepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= DELepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= DELepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= DELepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= DELepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= DELepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_DEL[interval1] += 1
            if games == numOfRandomGamesToSolve:
                print("\tTotal Time for DEL =", DELtotalTime, "seconds",
                    ".\tAvg Time for DEL =", DELtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)

                with open("PkDELApproxNEHistogram.out", "w") as f:
                    for key in registerE_DEL.keys():
                        f.write("%s,%s\n" % (key, registerE_DEL[key]))
                with open("PkDELWSNEHistogram.out", "w") as f:
                    for key in registerW_DEL.keys():
                        f.write("%s,%s\n" % (key, registerW_DEL[key]))


                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(DELepsAPPROX_list)
                max1_index = DELepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                DELepsAPPROX_list.remove(max1)
                max2 = max(DELepsAPPROX_list)
                max2_index = DELepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkDELApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDELApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkDELApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDELApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')
                
                #plot the histogram of epsAPPROX
                plt.bar(registerE_DEL.keys(), registerE_DEL.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDELApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_DEL.keys(), registerW_DEL.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDELWSNEHistogram.jpg')
                plt.clf()

    elif choice1 == "3":
        # EXECUTING FICTITIOUS PLAY algorithm (Pure Best Response version) ...
        start_time = time.time()
        x, y, FPPBRepsAPPROX, FPPBRepsWSNE = approxNEConstructionFPPBR(
            reduced_m, reduced_n, reduced_R, reduced_C)
        FPPBRx, FPPBRy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        FPPBRtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for FICTITIOUS PLAY (Pure Best Response version):")
        print(MINUSLINE)
        print("\tFPPBRx =", FPPBRx, "\n\tFPPBRy =", FPPBRy)
        print("\tFPPBRepsAPPROX =", FPPBRepsAPPROX,
            ".\tFPPBRepsWSNE =", FPPBRepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
        
        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            FPPBRepsAPPROX_list.append(FPPBRepsAPPROX)
            R_list.append(R)
            C_list.append(C)

            if 0 <= FPPBRepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= FPPBRepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= FPPBRepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= FPPBRepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= FPPBRepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= FPPBRepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= FPPBRepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= FPPBRepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= FPPBRepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= FPPBRepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_FPPBR[interval] += 1

            if 0 <= FPPBRepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= FPPBRepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= FPPBRepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= FPPBRepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= FPPBRepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= FPPBRepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= FPPBRepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= FPPBRepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= FPPBRepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= FPPBRepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_FPPBR[interval1] += 1

            if games == numOfRandomGamesToSolve:
                print("\tTotal Time for FPPBR =", FPPBRtotalTime, "seconds",
                    ".\tAvg Time for FPPBR =", FPPBRtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)


                with open("PkFPPBRApproxNEHistogram.out", "w") as f:
                    for key in registerE_FPPBR.keys():
                        f.write("%s,%s\n" % (key, registerE_FPPBR[key]))
                with open("PkFPPBRWSNEHistogram.out", "w") as f:
                    for key in registerW_FPPBR.keys():
                        f.write("%s,%s\n" % (key, registerW_FPPBR[key]))
                
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(FPPBRepsAPPROX_list)
                max1_index = FPPBRepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                FPPBRepsAPPROX_list.remove(max1)
                max2 = max(FPPBRepsAPPROX_list)
                max2_index = FPPBRepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkFPPBRRApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPPBRApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkFPPBRApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPPBRApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_FPPBR.keys(), registerE_FPPBR.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPPBRApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_FPPBR.keys(), registerW_FPPBR.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPPBRWSNEHistogram.jpg')
                plt.clf()


    elif choice1 == "4":
        # EXECUTING FICTITIOUS PLAY  UNIFORM ALGORITHM...
        start_time = time.time()
        x, y, FPUNepsAPPROX, FPUNepsWSNE = approxNEConstructionFPUNI(
            reduced_m, reduced_n, reduced_R, reduced_C)
        FPUNx, FPUNy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        FPUNtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for FICTITIOUS PLAY (UNIFORM):")
        print(MINUSLINE)
        print("\tFPUNx =", FPUNx, "\n\tFPUNy =", FPUNy)
        print("\tFPUNepsAPPROX =", FPUNepsAPPROX,
            ".\tFPUNepsWSNE =", FPUNepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
        
        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            FPUNepsAPPROX_list.append(FPUNepsAPPROX)
            R_list.append(R)
            C_list.append(C)


            if 0 <= FPUNepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= FPUNepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= FPUNepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= FPUNepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= FPUNepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= FPUNepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= FPUNepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= FPUNepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= FPUNepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= FPUNepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_FPUN[interval] += 1

            if 0 <= FPUNepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= FPUNepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= FPUNepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= FPUNepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= FPUNepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= FPUNepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= FPUNepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= FPUNepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= FPUNepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= FPUNepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_FPUN[interval1] += 1
            if games == numOfRandomGamesToSolve:

                print("\tTotal Time for FPUN =", FPUNtotalTime, "seconds",
                    ".\tAvg Time for FPUN =", FPUNtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                with open("PkFPUNIFORMApproxNEHistogram.out", "w") as f:
                    for key in registerE_FPUN.keys():
                        f.write("%s,%s\n" % (key, registerE_FPUN[key]))
                with open("PkFPUNIFORMWSNEHistogram.out", "w") as f:
                    for key in registerW_FPUN.keys():
                        f.write("%s,%s\n" % (key, registerW_FPUN[key]))
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(FPUNepsAPPROX_list)
                max1_index = FPUNepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                FPUNepsAPPROX_list.remove(max1)
                max2 = max(FPUNepsAPPROX_list)
                max2_index = FPUNepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkFPUNIFORMApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPUNIFORMApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkFPUNIFORMApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPUNIFORMApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_FPUN.keys(), registerE_FPUN.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPUNIFORMApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_FPUN.keys(), registerW_FPUN.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPUNIFORMWSNEHistogram.jpg')
                plt.clf()


    elif choice1 == "5":
        # EXECUTING DMP ALGORITHM...
        start_time = time.time()
        x, y, DMPepsAPPROX, DMPepsWSNE = approxNEConstructionDMP(
            reduced_m, reduced_n, reduced_R, reduced_C)
        DMPx, DMPy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        DMPtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DMP:")
        print(MINUSLINE)
        print("\tDMPx =", DMPx, "\n\tDMPy =", DMPy)
        print("\tDMPepsAPPROX =", DMPepsAPPROX,
            ".\tDMPepsWSNE =", DMPepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
        if(flag == 1):
            DMPepsAPPROX_list.append(DMPepsAPPROX)
            R_list.append(R)
            C_list.append(C)

            if 0 <= DMPepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= DMPepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= DMPepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= DMPepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= DMPepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= DMPepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= DMPepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= DMPepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= DMPepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= DMPepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_DMP[interval] += 1
            if 0 <= DMPepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= DMPepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= DMPepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= DMPepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= DMPepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= DMPepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= DMPepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= DMPepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= DMPepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= DMPepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_DMP[interval1] += 1
            if games == numOfRandomGamesToSolve:
                
                with open("PkDMPApproxNEHistogram.out", "w") as f:
                    for key in registerE_DMP.keys():
                        f.write("%s,%s\n" % (key, registerE_DMP[key]))
                with open("PkDMPWSNEHistogram.out", "w") as f:
                    for key in registerW_DMP.keys():
                        f.write("%s,%s\n" % (key, registerW_DMP[key]))
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(DMPepsAPPROX_list)
                max1_index = DMPepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                DMPepsAPPROX_list.remove(max1)
                max2 = max(DMPepsAPPROX_list)
                max2_index = DMPepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkDMPApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDMPApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkDMPApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDMPApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_DMP.keys(), registerE_DMP.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDMPApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_DMP.keys(), registerW_DMP.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDMPWSNEHistogram.jpg')
                plt.clf()


        # EXECUTING DEL ALGORITHM...
        start_time = time.time()
        x, y, DELepsAPPROX, DELepsWSNE = approxNEConstructionDEL(
            reduced_m, reduced_n, reduced_R, reduced_C)
        DELx, DELy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        DELtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DEL:")
        print(MINUSLINE)
        print("\tDELx =", DELx, "\n\tDELy =", DELy)
        print("\tDELepsAPPROX =", DELepsAPPROX,
            ".\tDELepsWSNE =", DELepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)

        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            DELepsAPPROX_list.append(DELepsAPPROX)

            if 0 <= DELepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= DELepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= DELepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= DELepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= DELepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= DELepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= DELepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= DELepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= DELepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= DELepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_DEL[interval] += 1

            if 0 <= DELepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= DELepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= DELepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= DELepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= DELepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= DELepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= DELepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= DELepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= DELepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= DELepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_DEL[interval1] += 1
            if games == numOfRandomGamesToSolve:
                
                with open("PkDELApproxNEHistogram.out", "w") as f:
                    for key in registerE_DEL.keys():
                        f.write("%s,%s\n" % (key, registerE_DEL[key]))
                with open("PkDELWSNEHistogram.out", "w") as f:
                    for key in registerW_DEL.keys():
                        f.write("%s,%s\n" % (key, registerW_DEL[key]))


                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(DELepsAPPROX_list)
                max1_index = DELepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                DELepsAPPROX_list.remove(max1)
                max2 = max(DELepsAPPROX_list)
                max2_index = DELepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkDELApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDELApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkDELApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkDELApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')
                
                #plot the histogram of epsAPPROX
                plt.bar(registerE_DEL.keys(), registerE_DEL.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDELApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_DEL.keys(), registerW_DEL.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkDELWSNEHistogram.jpg')
                plt.clf()



        # EXECUTING FICTITIOUS PLAY algorithm (Pure Best Response version) ...
        start_time = time.time()
        x, y, FPPBRepsAPPROX, FPPBRepsWSNE = approxNEConstructionFPPBR(
            reduced_m, reduced_n, reduced_R, reduced_C)
        FPPBRx, FPPBRy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        FPPBRtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for FICTITIOUS PLAY (Pure Best Response version):")
        print(MINUSLINE)
        print("\tFPPBRx =", FPPBRx, "\n\tFPPBRy =", FPPBRy)
        print("\tFPPBRepsAPPROX =", FPPBRepsAPPROX,
            ".\tFPPBRepsWSNE =", FPPBRepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            FPPBRepsAPPROX_list.append(FPPBRepsAPPROX)

            if 0 <= FPPBRepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= FPPBRepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= FPPBRepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= FPPBRepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= FPPBRepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= FPPBRepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= FPPBRepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= FPPBRepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= FPPBRepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= FPPBRepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_FPPBR[interval] += 1

            if 0 <= FPPBRepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= FPPBRepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= FPPBRepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= FPPBRepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= FPPBRepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= FPPBRepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= FPPBRepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= FPPBRepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= FPPBRepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= FPPBRepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_FPPBR[interval1] += 1

            if games == numOfRandomGamesToSolve:
                
                with open("PkFPPBRApproxNEHistogram.out", "w") as f:
                    for key in registerE_FPPBR.keys():
                        f.write("%s,%s\n" % (key, registerE_FPPBR[key]))
                with open("PkFPPBRWSNEHistogram.out", "w") as f:
                    for key in registerW_FPPBR.keys():
                        f.write("%s,%s\n" % (key, registerW_FPPBR[key]))
                
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(FPPBRepsAPPROX_list)
                max1_index = FPPBRepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                FPPBRepsAPPROX_list.remove(max1)
                max2 = max(FPPBRepsAPPROX_list)
                max2_index = FPPBRepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkFPPBRRApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPPBRApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkFPPBRApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPPBRApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_FPPBR.keys(), registerE_FPPBR.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPPBRApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_FPPBR.keys(), registerW_FPPBR.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPPBRWSNEHistogram.jpg')
                plt.clf()

        # EXECUTING FICTITIOUS PLAY  UNIFORM ALGORITHM...
        start_time = time.time()
        x, y, FPUNepsAPPROX, FPUNepsWSNE = approxNEConstructionFPUNI(
            reduced_m, reduced_n, reduced_R, reduced_C)
        FPUNx, FPUNy = interpretReducedStrategiesForOriginalGame(
            x, y, R, C, reduced_R, reduced_C, dominated_rows, dominated_cols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        FPUNtotalTime += elapsed_time
        #if (flag == 0 or flag1 == 1):
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for FICTITIOUS PLAY (UNIFORM):")
        print(MINUSLINE)
        print("\tFPUNx =", FPUNx, "\n\tFPUNy =", FPUNy)
        print("\tFPUNepsAPPROX =", FPUNepsAPPROX,
            ".\tFPUNepsWSNE =", FPUNepsWSNE,
            ".\tRuntime =", elapsed_time, "seconds", "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)

        if(flag == 1):
            # save all the epsAPPROX of this algorithm's iterations values in a list
            FPUNepsAPPROX_list.append(FPUNepsAPPROX)


            if 0 <= FPUNepsAPPROX < 0.1:
                interval = "[0,0.1)"
            elif 0.1 <= FPUNepsAPPROX < 0.2:
                interval = "[0.1,0.2)"
            elif 0.2 <= FPUNepsAPPROX < 0.3:
                interval = "[0.2,0.3)"
            elif 0.3 <= FPUNepsAPPROX < 0.4:
                interval = "[0.3,0.4)"
            elif 0.4 <= FPUNepsAPPROX < 0.5:
                interval = "[0.4,0.5)"
            elif 0.5 <= FPUNepsAPPROX < 0.6:
                interval = "[0.5,0.6)"
            elif 0.6 <= FPUNepsAPPROX < 0.7:
                interval = "[0.6,0.7)"
            elif 0.7 <= FPUNepsAPPROX < 0.8:
                interval = "[0.7,0.8)"
            elif 0.8 <= FPUNepsAPPROX < 0.9:
                interval = "[0.8,0.9)"
            elif 0.9 <= FPUNepsAPPROX <= 1:
                interval = "[0.9,1.0]"
            registerE_FPUN[interval] += 1

            if 0 <= FPUNepsWSNE < 0.1:
                interval1 = "[0,0.1)"
            elif 0.1 <= FPUNepsWSNE < 0.2:
                interval1 = "[0.1,0.2)"
            elif 0.2 <= FPUNepsWSNE < 0.3:
                interval1 = "[0.2,0.3)"
            elif 0.3 <= FPUNepsWSNE < 0.4:
                interval1 = "[0.3,0.4)"
            elif 0.4 <= FPUNepsWSNE < 0.5:
                interval1 = "[0.4,0.5)"
            elif 0.5 <= FPUNepsWSNE < 0.6:
                interval1 = "[0.5,0.6)"
            elif 0.6 <= FPUNepsWSNE < 0.7:
                interval1 = "[0.6,0.7)"
            elif 0.7 <= FPUNepsWSNE < 0.8:
                interval1 = "[0.7,0.8)"
            elif 0.8 <= FPUNepsWSNE < 0.9:
                interval1 = "[0.8,0.9)"
            elif 0.9 <= FPUNepsWSNE <= 1:
                interval1 = "[0.9,1.0]"
            registerW_FPUN[interval1] += 1
            if games == numOfRandomGamesToSolve:
                print("\tTotal Time for DMP =", DMPtotalTime, "seconds",
                    ".\tAvg Time for DMP =", DMPtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                print("\tTotal Time for DEL =", DELtotalTime, "seconds",
                    ".\tAvg Time for DEL =", DELtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                print("\tTotal Time for FPPBR =", FPPBRtotalTime, "seconds",
                    ".\tAvg Time for FPPBR =", FPPBRtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                print("\tTotal Time for FPUN =", FPUNtotalTime, "seconds",
                    ".\tAvg Time for FPUN =", FPUNtotalTime/numOfRandomGamesToSolve, "seconds", "." + bcolors.ENDC)
                with open("PkFPUNIFORMApproxNEHistogram.out", "w") as f:
                    for key in registerE_FPUN.keys():
                        f.write("%s,%s\n" % (key, registerE_FPUN[key]))
                with open("PkFPUNIFORMWSNEHistogram.out", "w") as f:
                    for key in registerW_FPUN.keys():
                        f.write("%s,%s\n" % (key, registerW_FPUN[key]))
                
                #find the 2 largest epsAPPROX in the list and their corresponding R and C
                max1 = max(FPUNepsAPPROX_list)
                max1_index = FPUNepsAPPROX_list.index(max1)
                R1 = R_list[max1_index]
                C1 = C_list[max1_index]
                FPUNepsAPPROX_list.remove(max1)
                max2 = max(FPUNepsAPPROX_list)
                max2_index = FPUNepsAPPROX_list.index(max2)
                R2 = R_list[max2_index]
                C2 = C_list[max2_index]
                
                #save the R1 and C1 in a .out file with a name PkALGApproxNEWorstGame1ROW and PkALGApproxNEWorstGame1COL
                np.savetxt("PkFPUNIFORMApproxNEWorstGame1ROW.out", R1, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPUNIFORMApproxNEWorstGame1COL.out", C1, delimiter=',', fmt='%1.4e')

                #save the R2 and C2 in a .out file with a name PkALGApproxNEWorstGame2ROW and PkALGApproxNEWorstGame2COL
                np.savetxt("PkFPUNIFORMApproxNEWorstGame2ROW.out", R2, delimiter=',', fmt='%1.4e')
                np.savetxt("PkFPUNIFORMApproxNEWorstGame2COL.out", C2, delimiter=',', fmt='%1.4e')

                #plot the histogram of epsAPPROX
                plt.bar(registerE_FPUN.keys(), registerE_FPUN.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Approximate Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPUNIFORMApproxNEHistogram.jpg')
                plt.clf()

                #plot the histogram of epsWSNE
                plt.bar(registerW_FPUN.keys(), registerW_FPUN.values(), color='g')
                plt.xlabel('Epsilon')
                plt.ylabel('Frequency')
                plt.title('Histogram of Epsilon Well Supported Nash Equilibrium')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
                plt.savefig('PkFPUNIFORMWSNEHistogram.jpg')
                plt.clf()

