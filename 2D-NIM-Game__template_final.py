# 2-D NIM Game
import os
import random
import math
import re

############################### FG COLOR DEFINITIONS ###############################


class bcolors:
    # pure colors...
    GREY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    # color styles...
    HEADER = '\033[95m'
    QUESTION = '\033[93m\033[3m'
    MSG = '\033[96m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
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

############################### GAME FUNCTIONS ###############################


def screen_clear():
    # for mac and linux(here, os.name is 'posix')
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        # for windows platfrom
        _ = os.system('cls')


def initializeBoard(N):
    board = ['']*(N*N+1)

    # this is the COUNTER of cells in the board already filled with R or G
    board[0] = 0

    # each EMPTY cell in the board contains its cardinal number
    for i in range(N*N):
        if i < 9:
            board[i+1] = ' ' + str(i+1)
        else:
            board[i+1] = str(i+1)
    return board


def drawNimPalette(board, N):

    EQLINE = '\t'
    MINUSLINE = '\t'
    CONSECEQUALS = ''
    CONSECMINUS = ''
    for i in range(5):
        CONSECEQUALS = CONSECEQUALS + '='
        CONSECMINUS = CONSECMINUS + '-'

    for i in range(10):
        EQLINE = EQLINE + CONSECEQUALS
        MINUSLINE = MINUSLINE + CONSECMINUS

    for i in range(N):
        # PRINTING ROW i...
        if i == 0:
            print(EQLINE)
        else:
            print(MINUSLINE)

        printRowString = ''

        for j in range(N):
            # PRINTING CELL (i,j)...
            CellString = str(board[N*i+j+1])
            if CellString == 'R':
                CellString = ' ' + bcolors.RED + CellString + bcolors.ENDC

            if CellString == 'G':
                CellString = ' ' + bcolors.GREEN + CellString + bcolors.ENDC

            if printRowString == '':
                printRowString = '\t[ ' + CellString
            else:
                printRowString = printRowString + ' | ' + CellString
        printRowString = printRowString + ' ]'
        print(printRowString)
    print(EQLINE)
    cnt = 0
    for i in range(1, N*N + 1):
        if board[i] == 'R' or board[i] == 'G':
            cnt = cnt+1

    print(bcolors.PURPLE +
          '\t\t\tCOUNTER = [ ' + str(cnt) + ' ]' + bcolors.ENDC)
    print(EQLINE)


def inputPlayerLetter():
    # The player chooses which label (letter) will fill the cells
    letter = ''
    while not (letter == 'G' or letter == 'R'):
        print(bcolors.QUESTION +
              '[Q1] What letter do you choose to play? [ G(reen) | R(ed) ]' + bcolors.ENDC)
        letter = input().upper()
        # The first letter corresponds to the HUMAN and the second element corresponds to the COMPUTER
        if letter == 'G':
            return ['G', 'R']
        else:
            if letter == 'R':
                return ['R', 'G']
            else:
                print(
                    bcolors.ERROR + 'ERROR1: You provided an invalid choice. Please try again...' + bcolors.ENDC)


def whoGoesFirst():
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'


def howComputerPlays():

    while True:
        print(bcolors.QUESTION +
              '[Q5] How will the computer play? [ R (randomly) | F (first Free) | C (copycat)]' + bcolors.ENDC)
        strategyLetter = input().upper()

        if strategyLetter == 'R':
            return 'random'
        else:
            if strategyLetter == 'F':
                return 'first free'
            else:
                if strategyLetter == 'C':
                    return 'copycat'
                else:
                    print(
                        bcolors.ERROR + 'ERROR 3: Incomprehensible strategy was provided. Try again...' + bcolors.ENDC)


def getBoardSize():

    BoardSize = 0
    while BoardSize < 1 or BoardSize > 10:
        GameSizeString = input(
            'Determine the size 1 =< N =< 10, for the NxN board to play: ')
        if GameSizeString.isdigit():
            BoardSize = int(GameSizeString)
            if BoardSize < 1 or BoardSize > 10:
                print(bcolors.ERROR + 'ERROR 4: Only positive integers between 1 and 10 are allowable values for N. Try again...' + bcolors.ENDC)
        else:
            print(bcolors.ERROR + 'ERROR 5: Only positive integers between 1 and 10 are allowable values for N. Try again...' + bcolors.ENDC)
    return (BoardSize)


def startNewGame():
    # Function for starting a new game
    print(bcolors.QUESTION +
          '[Q0] Would you like to start a new game? (yes or no)' + bcolors.ENDC)
    return input().lower().startswith('y')


def continuePlayingGame():
    # Function for starting a new game
    print(bcolors.QUESTION +
          '[Q2] Would you like to continue playing this game? (yes or no)' + bcolors.ENDC)
    return input().lower().startswith('y')


def playAgain():
    # Function for replay (when the player wants to play again)
    print(bcolors.QUESTION +
          '[Q3] Would you like to continue playing this game? (yes or no)' + bcolors.ENDC)
    return input().lower().startswith('y')


def isBoardFull(board, N):
    # Function for checking if the board is full
    for i in range(1, N*N + 1):
        # if the cell does not contain G or R then it is empty
        if board[i] != 'G' and board[i] != 'R':
            return False  # return False if there is an empty cell
    return True


def getPlayerMove(nimBoard, diagonalCells, initialNimBoard):
    # This function returns the player's move.
    # the player will choose a tile that is empty and will fill it with his label.
    # Then if he chooses to continue he will choose another tile and so on until he has picked 3 consecutive tiles in the same row or column
    possibleMoves = getAvailableCells(nimBoard)
    choice = []
    while (1):
        print(bcolors.QUESTION +
              '[Q4] Which is the first tile you choose:' + bcolors.ENDC)
        num = input()
        if num.isdigit():
            num = int(num)
            if num in possibleMoves:        # if the selected tile is available
                choice.append(num)
                if num in diagonalCells:    # if a main diagonal cell is chosen as the first move, the player can choose no more tiles
                    return choice
                while (1):
                    print(bcolors.QUESTION +
                          '[Q5] Which is the second tile you choose:' + bcolors.ENDC)
                    num1 = input()
                    if num1.isdigit():
                        num1 = int(num1)
                        if num1 in possibleMoves and checkValidMove(initialNimBoard, choice + [num1]):
                            choice.append(num1)
                            while (1):
                                print(bcolors.QUESTION +
                                      '[Q6] Which is the third tile you choose:' + bcolors.ENDC)
                                num2 = input()
                                if num2.isdigit():
                                    num2 = int(num2)

                                    if num2 in possibleMoves and checkValidMove(initialNimBoard, choice + [num2]):
                                        choice.append(num2)
                                        if checkValidMove2(choice):
                                            return choice
                                        else:
                                            print(bcolors.ERROR +
                                                  'ERROR 8: The tiles you chose were not consecutive. Try again from the beginning...' + bcolors.ENDC)
                                            return False
                                    else:
                                        print(bcolors.ERROR +
                                              'ERROR 6: The tile you chose is not available or invalid. Try again...' + bcolors.ENDC)
                                else:
                                    print(bcolors.MSG +
                                          'Your input was not an integer so your turn was terminated.' + bcolors.ENDC)
                                    if checkValidMove2(choice):
                                        print(bcolors.GREEN +
                                              'Move executed succesfully!' + bcolors.ENDC)
                                        return choice
                                    else:
                                        print(bcolors.ERROR +
                                              'ERROR 8: The tiles you chose were not consecutive. Try again from the beginning...' + bcolors.ENDC)
                                        return False
                        else:
                            print(bcolors.ERROR +
                                  'ERROR 5: The tile you chose is not available or invalid. Try again...' + bcolors.ENDC)
                    else:
                        print(bcolors.MSG +
                              'Your input was not an integer so your turn was terminated.' + bcolors.ENDC)
                        print(bcolors.GREEN +
                                  'Move executed succesfully!' + bcolors.ENDC)
                        return choice
            else:
                print(bcolors.ERROR +
                      'ERROR 6: The tile you chose is not available. Try again...' + bcolors.ENDC)

        else:
            print(bcolors.ERROR +
                  'ERROR 7: Your input was not an integer. Please enter a valid choice!' + bcolors.ENDC)


def getComputerMove(nimBoard, computerStrategy, initialNimBoard, playermoves):
    # This function returns the computer's move.
    # The computer will choose a random move from the list of
    # available moves.
    if computerStrategy == 'random':
        return getComputerMove_random(nimBoard, initialNimBoard)
    else:
        if computerStrategy == 'first free':
            return getComputerMove_firstfit(nimBoard, initialNimBoard)
        else:
            if computerStrategy == 'copycat':
                return getComputerMove_copycat(nimBoard, initialNimBoard, playermoves)
            else:
                print(
                    bcolors.ERROR + 'ERROR 2: Incomprehensible strategy was provided. Try again...' + bcolors.ENDC)


def getComputerMove_random(nimBoard, initialNimBoard):
    # This function returns a random move from the list of
    # available moves.
    # return a random available tile to start the move
    possibleMoves = getAvailableCells(nimBoard)
    choice = []

    # if there's more than five available cells left (case of 0 or 1 cell left are considered trivial and also included)
    if len(possibleMoves) > 5 or len(possibleMoves) < 2:
        choice = randomMoreThanFiveLeft(nimBoard, initialNimBoard)

    # if there's five or less available cells left (not including 0 or 1 cell left)
    elif len(possibleMoves) == 2:
        choice = twoCellsLeft(
            'random', nimBoard, initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 3:
        choice = threeCellsLeft(
            'random', nimBoard, initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 4:
        choice = fourCellsLeft(
            'random', nimBoard, initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 5:
        choice = fiveCellsLeft(
            'random', nimBoard, initialNimBoard, possibleMoves, [])
    return choice


def randomMoreThanFiveLeft(nimBoard, initialNimBoard):
    possibleMoves = getAvailableCells(nimBoard)
    choice = []

    tile = random.choice(possibleMoves)
    choice.append(tile)

    # randomly select a number between 1 and 3
    num = random.randint(1, 3)

    #  select if the next tile will be in the same row or column
    if random.randint(0, 1) == 0:
        where = 'row'
    else:
        where = 'column'

    if random.randint(0, 1) == 0:
        to_where = 'up'
    else:
        to_where = 'down'
    i = 1
    if (num > 1 and tile not in getDiagonalCells(initialNimBoard)):
        while i < num:
            flag = 0

            # check the where and to_where and then check the board if the next tile move is valid.
            # If it is valid then append it to the choice list if not then break
            if where == 'row' and to_where == 'up':
                next_tile = tile - 1
                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    # if the move is not valid, change the to_where variable and try again
                    to_where = 'down'
                    if num == 3 and i == 2:
                        tile = choice[0]
                    flag = 1
            if where == 'row' and to_where == 'down':
                next_tile = tile + 1
                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    # if the move is not valid, change the to_where variable and try again
                    to_where = 'up'
                    i = i-1
                    if num == 3 and i == 2:
                        tile = choice[0]
                    # if the value of flag is 1 then break the loop (to_where is already changed)
                    if flag == 1:
                        break
            if where == 'column' and to_where == 'up':
                next_tile = tile - int(math.sqrt(len(nimBoard)))

                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    # if the move is not valid, change the to_where variable and try again
                    to_where = 'down'
                    if num == 3 and i == 2:
                        tile = choice[0]
                    flag = 1
            if where == 'column' and to_where == 'down':
                next_tile = tile + int(math.sqrt(len(nimBoard)))

                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    # if the move is not valid, change the to_where variable and try again
                    to_where = 'up'
                    i = i-1
                    if num == 3 and i == 2:
                        tile = choice[0]
                    # if the value of flag is 1 then break the loop (to_where is already changed)
                    if flag == 1:
                        break
            i = i+1
    return choice


def getComputerMove_firstfit(nimBoard, initialNimBoard):
    # This function returns the first free move from the list of
    # available moves.
    # return the first available tile to start the move
    possibleMoves = getAvailableCells(nimBoard)
    choice = []

    # if there's more than five available cells left (case of 0 or 1 cell left are considered trivial and also included)
    if len(possibleMoves) > 5 or len(possibleMoves) < 2:
        choice = firstfreeMoreThanFiveLeft(nimBoard, initialNimBoard)

    # if there's five or less available cells left (not including 0 or 1 cell left)
    elif len(possibleMoves) == 2:
        choice = twoCellsLeft('first free', nimBoard,
                              initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 3:
        choice = threeCellsLeft(
            'first free', nimBoard, initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 4:
        choice = fourCellsLeft(
            'first free', nimBoard, initialNimBoard, possibleMoves, [])

    elif len(possibleMoves) == 5:
        choice = fiveCellsLeft(
            'first free', nimBoard, initialNimBoard, possibleMoves, [])
    return choice


def firstfreeMoreThanFiveLeft(nimBoard, initialNimBoard):
    # This function returns the first free move from the list of
    # available moves.
    # return the first available tile to start the move
    possibleMoves = getAvailableCells(nimBoard)
    choice = []
    tile = possibleMoves[0]
    choice.append(tile)

    # randomly select a number between 1 and 3
    num = random.randint(1, 3)
    #  select if the next tile will be in the same row or column
    if random.randint(0, 1) == 0:
        where = 'row'
    else:
        where = 'column'

    to_where = 'down'

    if (num > 1 and tile not in getDiagonalCells(initialNimBoard)):
        for i in range(1, num):
            # check if the next move is valid
            if where == 'row' and to_where == 'down':
                next_tile = tile + 1
                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    break
            elif where == 'column' and to_where == 'down':
                next_tile = tile + int(math.sqrt(len(nimBoard)))
                if next_tile in possibleMoves and checkValidMove(initialNimBoard, choice + [next_tile]):
                    tile = next_tile
                    choice.append(tile)
                else:
                    break
    return choice


def getComputerMove_copycat(nimBoard, initialNimBoard, playermoves):
    # This function returns the copycat move from the list of
    # available moves.
    # return the first available tile to start the move
    possibleMoves = getAvailableCells(nimBoard)
    choice = []
    # if there's more than five available cells left (case of 0 or 1 cell left are considered trivial and also included)
    if len(possibleMoves) > 5 or len(possibleMoves) < 2:
        choice = copycatMoreThanFiveLeft(
            nimBoard, initialNimBoard, playermoves)
    # if there's five or less available cells left (not including 0 or 1 cell left)
    elif len(possibleMoves) == 2:
        choice = twoCellsLeft(
            'copycat', nimBoard, initialNimBoard, possibleMoves, playermoves)
    elif len(possibleMoves) == 3:
        choice = threeCellsLeft(
            'copycat', nimBoard, initialNimBoard, possibleMoves, playermoves)
    elif len(possibleMoves) == 4:
        choice = fourCellsLeft(
            'copycat', nimBoard, initialNimBoard, possibleMoves, playermoves)
    elif len(possibleMoves) == 5:
        choice = fiveCellsLeft(
            'copycat', nimBoard, initialNimBoard, possibleMoves, playermoves)
    return choice


def copycatMoreThanFiveLeft(nimBoard, initialNimBoard, playermoves):
    # This function returns the copycat move from what the player played. If the computer goes first or the player move can be copied then the computer chooses either a random move with getComputerMove_random or a first free move with getComputerMove_firstfit. 
    # If the players move cant be copied then the move randomly chosen must contain as many tiles as the player move or less, but not more.
    # return the first available tile to start the move
    possibleMoves = getAvailableCells(nimBoard)
    init_possibleMoves = getAvailableCells(initialNimBoard)
    choice = []
    # if the computer goes first
    if len(possibleMoves) == len(init_possibleMoves):
        if random.randint(0, 1) == 0:
            choice = getComputerMove_random(nimBoard, initialNimBoard)
        else:
            choice = getComputerMove_firstfit(nimBoard, initialNimBoard)
    # if the player goes first
    else:
        # find the move the player performed last and copy it by doing the same move symmetrically by the diagonal, The players last move is in the playermoves list
        # find the tile that is symmetric to the player move
        # if the players move is on the diagonal then the computer chooses the tile in the diagonal that is symmetric in the center of the diagonal.
        # If the cell is not available then the computer chooses a random cell in the diagonal.
        if playermoves[0] in getDiagonalCells(initialNimBoard):
            diagon = getDiagonalCells(initialNimBoard)
            for i in range(len(diagon)):
                if playermoves[0] == diagon[i]:
                    if diagon[len(diagon)-1-i] in possibleMoves:
                        choice.append(diagon[len(diagon)-1-i])
                    else:
                        # check first if any of the elements of the diagonal are possible moves
                        # if not then choose a random move
                        if not any(x in possibleMoves for x in diagon):
                            choice.append(possibleMoves[0])
                        else:
                            # choice is equal to a random empty cell in the diagonal
                            while True:
                                choice = []
                                choice.append(random.choice(diagon))
                                if choice[0] in possibleMoves:
                                    break
                # if choice is not empty then break
                if choice:
                    break
        # if the players move is not on the diagonal then the computer chooses the tile that is symmetric to the player move,the symmetry is determined by the diagonal
        else:
            flag = 0
            for i in range(len(playermoves)):
                row = (playermoves[i] - 1) // N  # 0-indexed row
                col = (playermoves[i] - 1) % N   # 0-indexed column
                # find the tile that is symmetric to the player move
                symmetric = col * N + row + 1
                choice.append(symmetric)
                if choice[i] not in possibleMoves:
                    flag = 1
            if flag == 1:
                # even if one tile of move is not valid, choose a random move
                if random.randint(0, 1) == 0:
                    choice = getComputerMove_random(nimBoard, initialNimBoard)
                    while len(choice) > len(playermoves):
                        choice = getComputerMove_random(
                            nimBoard, initialNimBoard)
                else:
                    choice = getComputerMove_firstfit(
                        nimBoard, initialNimBoard)
                    while len(choice) > len(playermoves):
                        choice = getComputerMove_firstfit(
                            nimBoard, initialNimBoard)
    return choice


# a function that handles the case of exactly 2 available cells left
def twoCellsLeft(mode, nimBoard, initialNimBoard, possibleMoves, playermoves):
    global result_type
    result_type = 'undefined'
    choice2 = []
    # if the two cells are consecutive and none of them is on the diagonal then the computer chooses these cells and wins
    if possibleMoves[1] in getCellNeighbours(possibleMoves[0], initialNimBoard) and possibleMoves[0] not in getDiagonalCells(initialNimBoard) and possibleMoves[1] not in getDiagonalCells(initialNimBoard):
        choice2.extend([possibleMoves[0], possibleMoves[1]])
        result_type = 'specific'
    # else if the cells can't both be chosen (because one of them is on the diagonal or they are not consecutive)
    # then the computer continues playing according to its previous strategy
    else:
        if mode == 'random':
            t_choice = randomMoreThanFiveLeft(nimBoard, initialNimBoard)
            result_type = 'random'
        elif mode == 'first free':
            t_choice = firstfreeMoreThanFiveLeft(nimBoard, initialNimBoard)
            result_type = 'first free'
        elif mode == 'copycat':
            t_choice = copycatMoreThanFiveLeft(
                nimBoard, initialNimBoard, playermoves)
            result_type = 'copycat'
        choice2.extend(t_choice)
    return choice2


# a function that handles the case of exactly 3 available cells left
def threeCellsLeft(mode, nimBoard, initialNimBoard, possibleMoves, playermoves):
    global result_type
    result_type = 'undefined'
    choice3 = []
    # if the three cells are not consecutive and one of them has the other 2 as adjacent (with none them on the diagonal) then the computer picks the one in the middle
    if getCellwithNumNeighbours(2, initialNimBoard, possibleMoves) == False:
        # if thats not the case the computer plays the selected strategy
        if getCellwithNumNeighbours(1, initialNimBoard, possibleMoves) == False:
            if mode == 'random':
                t_choice = randomMoreThanFiveLeft(nimBoard, initialNimBoard)
            elif mode == 'first free':
                t_choice = firstfreeMoreThanFiveLeft(nimBoard, initialNimBoard)
            elif mode == 'copycat':
                t_choice = copycatMoreThanFiveLeft(
                    nimBoard, initialNimBoard, playermoves)
            choice3.extend(t_choice)
        # if there is a cell with 1 neighbour (neither of them in the diagonal), then that's the computer's next move
        else:
            t_choice = getCellwithNumNeighbours(
                1, initialNimBoard, possibleMoves)
            choice3.append(t_choice)
    # if there is a cell with 2 neighbours
    else:
        # if the three cells are consecutive and none of them is on the diagonal then the computer chooses these cells and wins
        t_choice = getCellwithNumNeighbours(2, initialNimBoard, possibleMoves)
        # check if the three cells are consecutive and the move is legal and not on the diagonal if yes then the computer chooses the three cells and wins
        temp = []
        temp.extend(possibleMoves)
        if (abs(possibleMoves[0] - possibleMoves[1]) == 1 or abs(possibleMoves[0] - possibleMoves[1]) == N) and (abs(possibleMoves[1] - possibleMoves[2]) == 1 or abs(possibleMoves[1] - possibleMoves[2]) == N) and possibleMoves[0] not in getDiagonalCells(initialNimBoard) and possibleMoves[1] not in getDiagonalCells(initialNimBoard) and possibleMoves[2] not in getDiagonalCells(initialNimBoard) and checkValidMove(initialNimBoard, temp):
            choice3.extend(temp)
            result_type = 'consecutive'
        else:
            choice3.append(t_choice)
    return choice3


# a function that handles the case of exactly 4 available cells left
def fourCellsLeft(mode, nimBoard, initialNimBoard, possibleMoves, playermoves):
    choice4 = []
    global result_type4
    result_type4 = 'undefined'

    # picks every combination of 2 available cells and, if they are consecutive and non-diagonal,
    # checks if they give a winning move
    for i in range(0, 4):
        t_posmoves = []
        t_posmoves.append(possibleMoves[i])
        for j in range(0, 4):
            checklist = []
            if i != j:
                t_posmoves.append(possibleMoves[j])
                twoCellsLeft(mode, nimBoard, initialNimBoard,
                             t_posmoves, playermoves)
                # checks if the 2 cells are consecutive and not on the diagonal
                if result_type == 'specific':
                    for k in range(0, 4):
                        if possibleMoves[k] not in t_posmoves:
                            checklist.append(possibleMoves[k])
                    twoCellsLeft(mode, nimBoard,
                                 initialNimBoard, checklist, playermoves)
                    # checks if the 2 cells that were previously chosen give a winning move
                    if result_type == 'random' or result_type == 'first free' or result_type == 'copycat':
                        choice4.extend(t_posmoves)
                        return choice4
                del t_posmoves[1]

    # if no winning move found, then computer plays according to its previous strategy
    if mode == 'random':
        t_choice = randomMoreThanFiveLeft(nimBoard, initialNimBoard)
        result_type4 = 'random'
    elif mode == 'first free':
        t_choice = firstfreeMoreThanFiveLeft(nimBoard, initialNimBoard)
        result_type4 = 'first free'
    elif mode == 'copycat':
        t_choice = copycatMoreThanFiveLeft(
            nimBoard, initialNimBoard, playermoves)
        result_type4 = 'copycat'
    choice4.extend(t_choice)

    return choice4


# a function that handles the case of exactly 5 available cells left
def fiveCellsLeft(mode, nimBoard, initialNimBoard, possibleMoves, playermoves):
    choice5 = []

    # picks every combination of 3 available cells and, if they are consecutive and non-diagonal,
    # checks if they give a winning move
    for i in range(0, 5):
        t_posmoves = []
        t_posmoves.append(possibleMoves[i])
        for j in range(0, 5):
            if i != j:
                t_posmoves.append(possibleMoves[j])
                for m in range(0, 5):
                    checklist = []
                    if m != j and m != i:
                        t_posmoves.append(possibleMoves[m])
                        threeCellsLeft(
                            mode, nimBoard, initialNimBoard, t_posmoves, playermoves)
                        # checks if the 3 cells are consecutive and not on the diagonal
                        if result_type == 'consecutive':
                            for k in range(0, 5):
                                if possibleMoves[k] not in t_posmoves:
                                    checklist.append(possibleMoves[k])
                            twoCellsLeft(mode, nimBoard,
                                         initialNimBoard, checklist, playermoves)
                            # checks if the 3 cells that were previously chosen give a winning move
                            if result_type == 'random' or result_type == 'first free' or result_type == 'copycat':
                                choice5.extend(t_posmoves)
                                return choice5
                        del t_posmoves[2]
                del t_posmoves[1]

    # picks every cell individually, and examines if playing it will give a winning move. If yes, the selected cell is played
    for i in range(0, 5):
        t_posmoves = []
        t_posmoves.append(possibleMoves[i])
        checklist = []
        for j in range(0, 5):
            if possibleMoves[j] not in t_posmoves:
                checklist.append(possibleMoves[j])
        fourCellsLeft(mode, nimBoard, initialNimBoard, checklist, playermoves)
        if result_type4 == 'random' or result_type4 == 'first free' or result_type4 == 'copycat':
            choice5.append(t_posmoves[0])
            return choice5


# a function that returns, if it exists, a cell with exactly num neighbours
def getCellwithNumNeighbours(num, initialNimBoard, possibleMoves):
    for i in range(len(possibleMoves)):
        count = getNumberofNeighbours(
            possibleMoves[i], initialNimBoard, possibleMoves)
        if count == num:
            return possibleMoves[i]
    return False


# a function that returns the number of neighbours of a given cell
def getNumberofNeighbours(cell, initialNimBoard, possibleMoves):
    count = 0
    neighbours = getCellNeighbours(cell,initialNimBoard)

    for i in range (0,len(neighbours)):
        if neighbours[i] in possibleMoves:
            count = count + 1
    return count


# a function that returns the neighbours of a cell
def getCellNeighbours(cell, initialNimBoard):
    neighbours = []
    row1 = getRowCells(initialNimBoard, 1)
    col1 = getColumnCells(initialNimBoard, 1)
    rowN = getRowCells(initialNimBoard, N*N - N + 1)
    colN = getColumnCells(initialNimBoard, N*N)

   # cells of first row
    if (cell in row1) and cell != N:
        neighbours.append(cell+1)
        neighbours.append(cell+N)
        if cell != 1:
            neighbours.append(cell-1)
    # cells of first column
    elif cell in col1:
        neighbours.append(cell+1)
        neighbours.append(cell-N)
        if cell != (N*N - N + 1):
            neighbours.append(cell+N)
    # cells of last row
    elif cell in rowN:
        neighbours.append(cell-1)
        neighbours.append(cell-N)
        if cell != N*N:
            neighbours.append(cell+1)
    # cells of last column
    elif cell in colN:
        neighbours.append(cell-1)
        neighbours.append(cell+N)
        if cell != N:
            neighbours.append(cell-N)
    # other cells
    else:
        neighbours.append(cell-1)
        neighbours.append(cell+1)
        neighbours.append(cell-N)
        neighbours.append(cell+N)
    return neighbours


# define a function that, given a vector of tiles, checks if the move is valid (if all tiles are in the same row or column,
# and not on the diagonal)
def checkValidMove(nimBoard, move):
    # get the length of the move
    # print(move)
    length = len(move)
    # check if the move is valid
    if length == 3:
        if move[length-1] != move[length-2]:
            # check if the move is in the same row
            if move[length-1] in getRowCells(nimBoard, move[length-2]):
                if move[length-1] in getRowCells(nimBoard, move[0]):
                    return True
            # check if the move is in the same column
            elif move[length-1] in getColumnCells(nimBoard, move[length-2]):
                if move[length-1] in getColumnCells(nimBoard, move[0]):
                    return True
            else:
                return False
        else:
            if move[length-1] in getRowCells(nimBoard, move[0]):
                return True
            # check if the move is in the same column
            if move[length-1] in getColumnCells(nimBoard, move[0]):
                return True
            else:
                return False
    elif length == 2:
        if move[length-1] not in getDiagonalCells(nimBoard):
            if move[length-1] in getRowCells(nimBoard, move[0]):
                return True
            # check if the move is in the same column
            elif move[length-1] in getColumnCells(nimBoard, move[0]):
                return True
            else:
                return False


# define a function that, given a vector of tiles, checks if the move is valid (if all tiles are consecutive)
def checkValidMove2(move):
    length = len(move)
    move.sort(reverse=False)    # arranges the move list into ascending order
    # check if the move is valid
    # check if the move is consecutive
    if length == 3:
        # if some of the tiles have been chosen more than once:
        if move[0] == move[1] == move[2]:
            return True
        elif (move[0] == move[1]) and (move[0] == move[2] - 1 or move[0] == move[2] - N):
            return True
        elif (move[1] == move[2]) and (move[0] == move[1] - 1 or move[0] == move[1] - N):
            return True
        # if all the tiles have been chosen exactly once:
        # check if the tiles are consecutive in the same row
        elif (move[0] == move[1] - 1) and (move[0] == move[2] - 2):
            return True
        # check if the tiles are consecutive in the same column
        elif (move[0] == move[1] - N) and (move[0] == move[2] - 2*N):
            return True
        else:
            return False
    elif length == 2:
        # if the player chose the same tile twice
        if move[0] == move[1]:
            return True
        # if the two chosen tiles are different
        else:
            # check if the tiles are consecutive in the same row
            if move[0] == move[1] - 1:
                return True
            # check if the tiles are consecutive in the same column
            elif move[0] == move[1] - N:
                return True
            else:
                return False


# define a function that returns the name of the cells in the same row of the nimboard as the cell with the given name
def getRowCells(nimBoard, cell_name):
    # create a new variable to hold the modified cell name
    new_cell_name = cell_name - 1
    # get the length of the nimboard
    length = len(nimBoard)
    # get the number of rows
    rows = int(math.sqrt(length))
    # extract the number from the cell name (if it's a string)
    if isinstance(new_cell_name, str):
        # extract the number from the cell name
        number_str = re.findall(r'\d+', new_cell_name)[0]
        number = int(number_str)
        # check if the number is a single-digit number with a space
        if len(number_str) == 1 and number_str[0] == ' ':
            # adjust the row number accordingly
            if number == 1:  # special case for first row
                row = 0
            else:
                row = (number - 1) // rows
            # adjust the column number accordingly
            col = (number - 1) % rows
            # convert to the actual cell name
            new_cell_name = nimBoard[row*rows + col]
    else:
        number = new_cell_name
    # get the row of the cell
    row = number // rows
    # create a list of the cells in the same row
    rowCells = []
    for i in range(rows):
        # search nimBoard for the cell with the same row and current column
        if isinstance(new_cell_name, str):
            search_num = i + 1
            cell = [c for c in nimBoard if str(search_num) in c][0]
        else:
            cell = nimBoard[row*rows + i]
            if isinstance(cell, int):
                cell = str(cell)
        number = int(re.findall(r'\d+', cell)[0])
        # add +1 to all the contents of the rowCells list
        rowCells.append(number + 1)
    return rowCells


# define a function getColumnCells that is like getRowCells but for columns
def getColumnCells(nimBoard, cell_name):
    # create a new variable to hold the modified cell name
    new_cell_name = cell_name - 1
    # get the length of the nimboard
    length = len(nimBoard)
    # get the number of rows
    rows = int(math.sqrt(length))
    # extract the number from the cell name (if it's a string)
    if isinstance(new_cell_name, str):
        # extract the number from the cell name
        number_str = re.findall(r'\d+', new_cell_name)[0]
        number = int(number_str)
        # check if the number is a single-digit number with a space
        if len(number_str) == 1 and number_str[0] == ' ':
            # adjust the row number accordingly
            if number == 1:  # special case for first row
                row = 0
            else:
                row = (number - 1) // rows
        # adjust the column number accordingly
        col = (number - 1) % rows
        # convert to the actual cell name
        new_cell_name = nimBoard[row*rows + col]
    else:
        number = new_cell_name
    # get the column of the cell
    col = number % rows
    # create a list of the cells in the same column
    columnCells = []
    for i in range(rows):
        # search nimBoard for the cell with the same column and current row
        if isinstance(new_cell_name, str):
            search_num = i*rows + 1
            cell = [c for c in nimBoard if str(search_num) in c][0]
        else:
            cell = nimBoard[i*rows + col]
            if isinstance(cell, int):
                cell = str(cell)
        number = int(re.findall(r'\d+', cell)[0])
        # add +1 to all the contents of the columnCells list
        columnCells.append(number + 1)
    return columnCells


# define a function that returns the name of the cells in the diagonal of the nimboard
def getDiagonalCells(nimBoard):
    # get the length of the nimboard
    length = len(nimBoard)
    # get the number of rows and columns
    rows = int(math.sqrt(length))
    # create a list of the diagonal cells
    diagonalCells = []
    # create a list of the diagonal cells
    for i in range(0, rows):
        cell_name = nimBoard[i*rows + i + 1]
        number = re.findall(r'\d+', cell_name)[0]
        diagonalCells.append(int(number))
    return diagonalCells


def makeMove(nimBoard, letter, move):
    # This function places the letter on the board at the move
    # (i.e. it changes the board).
    if isinstance(move, list):
        for i in range(len(move)):
            nimBoard[move[i]] = letter
    nimBoard[0] = nimBoard[0] + 1


def getAvailableCells(nimBoard):
    # This function returns a list of all the empty cells.
    # The list is empty if there are no empty cells.
    availableMoves = []
    for i in range(1, len(nimBoard)):
        # check if the nimboard[i] is a digit or (' ' + a digit)
        if nimBoard[i].isdigit() or (len(nimBoard[i]) == 2 and nimBoard[i][1].isdigit()):
            availableMoves.append(i)
    return availableMoves


######### MAIN PROGRAM BEGINS #########
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     CEID NE509 / LAB-1  
---------------------------------------------------------------------
STUDENT NAME:           < Ion Bournakas >
STUDENT AM:             < 1075475 >
JOINT WORK WITH:        < Maria - Vasilikh Petropoulou 1072540 >
---------------------------------------------------------------------
""" + bcolors.ENDC)

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (I)
---------------------------------------------------------------------
    1.      A human PLAYER plays against the COMPUTER.
    2.      The starting position is an empty NxN board.
    3.      One player (the green) writes G, the other player 
                (the red) writes R, in empty cells.
""" + bcolors.ENDC)

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (II) 
---------------------------------------------------------------------
    4.      The cells within the NxN board are indicated as 
            consecutive numbers, from 1 to N^2, starting from the 
            upper-left cell. E.g. for N=4, the starting position 
            and some intermediate position of the game would be 
            like those:
                    INITIAL POSITION        INTERMEDIATE POSITION
                    =====================   =====================
                    [  1 |  2 |  3 |  4 ]   [  1 |  2 |  3 |  4 ]
                    ---------------------   ---------------------
                    [  5 |  6 |  7 |  8 ]   [  5 |  R |  7 |  8 ]    
                    ---------------------   ---------------------
                    [  9 | 10 | 11 | 12 ]   [  9 |  R | 11 | 12 ] 
                    ---------------------   ---------------------
                    [ 13 | 14 | 15 | 16 ]   [  G |  G | 15 |  G ] 
                    =====================   =====================
                       COUNTER = [ 0 ]         COUNTER = [ 5 ]
                    =====================   =====================
""" + bcolors.ENDC)

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (III) 
---------------------------------------------------------------------
    5.      In each round the current player's turn is to fill with 
            his/her own letter (G or R) at least one 1 and at most 
            3 CONSECUTIVE, currently empty cells of the board, all 
            of them lying in the SAME ROW, or in the SAME COLUMN 
            of the board. Alternatively, the player may choose ONLY
            ONE empty diagonal cell to play.
    6.      The player who fills the last cell in the board WINS.
    7.      ENJOY!!!
---------------------------------------------------------------------
""" + bcolors.ENDC)

maxNumMoves = 3

playNewGameFlag = True

while playNewGameFlag:

    if not startNewGame():
        break

    N = getBoardSize()

    nimBoard = initializeBoard(N)
    initialNimBoard = nimBoard.copy()

    playerLetter, computerLetter = inputPlayerLetter()

    turn = whoGoesFirst()

    diagonalCells = getDiagonalCells(nimBoard)

    computerStrategy = howComputerPlays()

    print(bcolors.MSG + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + bcolors.ENDC)
    print(bcolors.MSG + 'A new ' + str(N) + 'x' + str(N) +
          ' game is about to start. The ' + turn + ' makes the first move.' + bcolors.ENDC)
    print(bcolors.MSG + ' * The computer will play according to the ' +
          bcolors.HEADER + computerStrategy + bcolors.MSG + ' strategy.' + bcolors.ENDC)
    print(bcolors.MSG + ' * The player will use the letter ' + playerLetter +
          ' and the computer will use the ' + computerLetter + '.' + bcolors.ENDC)
    print(bcolors.MSG + ' * The first move will be done by the ' +
          turn + '.' + bcolors.ENDC)
    print(bcolors.MSG + '---------------------------------------------------------------------' + bcolors.ENDC)
    drawNimPalette(nimBoard, N)
    playermoves = []
    # provided this code make the computer play with the random strategy
    while not isBoardFull(nimBoard, N):
        if turn == 'player':
            playermoves = []
            # Player's turn.
            move = getPlayerMove(nimBoard, diagonalCells, initialNimBoard)
            while (move == False):
                move = getPlayerMove(
                    nimBoard, diagonalCells, initialNimBoard)
            playermoves.extend(move)
            makeMove(nimBoard, playerLetter, move)
            drawNimPalette(nimBoard, N)
            if (isBoardFull(nimBoard, N) is True):
                drawNimPalette(nimBoard, N)
                print(
                    bcolors.HEADER + 'Congrats you have beaten the computer!' + bcolors.ENDC)
                break
            else:
                turn = 'computer'
        else:
            # Computer's turn.
            move = getComputerMove(
                nimBoard, computerStrategy, initialNimBoard, playermoves)
            print(move)
            makeMove(nimBoard, computerLetter, move)
            drawNimPalette(nimBoard, N)
            if (isBoardFull(nimBoard, N) is True):
                drawNimPalette(nimBoard, N)
                print(
                    bcolors.HEADER + 'The computer has beaten you! You lose.' + bcolors.ENDC)
                break
            else:
                turn = 'player'
