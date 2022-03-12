#!/usr/bin/env python3

import numpy as np
import random
from math import inf 
import pickle

# generate data of input and desired output pairs for neural net to train with
def main():
    # MAIN DATA GENERATING SCRIPT
    # INPUTS TO CHANGE
    numGames = 10000

    # Main function
    games = []
    serialList = []
    idx = 0
    serialHashMap = {}
    while idx < numGames:
        currentGame = GameData()
        currentGame.randomGame()
        if currentGame.serialString not in serialHashMap.values():
            games.append(currentGame)
            serialHashMap[idx] = currentGame.serialString
            serialList.append(currentGame.serialString)
            idx += 1

    dataNNformat = []
    idxer = numGames // 10
    for i in range(10):
        print("...Running " + str(i*10) + "%/ done...")
        dataNNformat.append(DataFormatNeuralNet(games[i*idxer:(i+1)*idxer]))
        dataNNformat[i].formatToNeuralNet()
        dataNNformat[i].getIOPairs()

        fileName = 'data' + str(i) + '.pkl'
        with open(fileName, 'wb') as f: 
            pickle.dump(dataNNformat[i], f) 

    # testing memoization
    # testBool = True
    # numContinues = 0
    # for i in range(len(dataNNformat.IOmap) // 2):
    #     addIdx = (len(dataNNformat.IOmap) // 2)
    #     # print(sum(dataNNformat.IOmap[i]['input'][9:18]))
    #     # print(dataNNformat.IOmap[i]['input'])
    #     # print(dataNNformat.IOmap[i + addIdx]['input'])
    #     if np.allclose(dataNNformat.IOmap[i]['input'], dataNNformat.IOmap[i + addIdx]['input']) and sum(dataNNformat.IOmap[i]['input'][9:18]) < 8:

    #         if not np.allclose(dataNNformat.IOmap[i]['output'], dataNNformat.IOmap[i + addIdx]['output']):
    #             testBool = False
    #             # print(dataNNformat.IOmap[i]['output'])
    #             # print(dataNNformat.IOmap[i + addIdx]['output'])
    #             idxBad = i + addIdx
    #             break

       
    

# Classes
class Translators:
      
    # Map moves square to board matrix
    boardMap = {}
    boardMap[1] = {'row': 0, 'col': 0}
    boardMap[2] = {'row': 1, 'col': 0}
    boardMap[3] = {'row': 2, 'col': 0}
    boardMap[4] = {'row': 0, 'col': 1}
    boardMap[5] = {'row': 1, 'col': 1}
    boardMap[6] = {'row': 2, 'col': 1}
    boardMap[7] = {'row': 0, 'col': 2}
    boardMap[8] = {'row': 1, 'col': 2}
    boardMap[9] = {'row': 2, 'col': 2}

    # Map NN format to board matrix
    NNtoBoardMap = {}
    NNtoBoardMap[0] = {'row': 0, 'col': 0}
    NNtoBoardMap[1] = {'row': 1, 'col': 0}
    NNtoBoardMap[2] = {'row': 2, 'col': 0}
    NNtoBoardMap[3] = {'row': 0, 'col': 1}
    NNtoBoardMap[4] = {'row': 1, 'col': 1}
    NNtoBoardMap[5] = {'row': 2, 'col': 1}
    NNtoBoardMap[6] = {'row': 0, 'col': 2}
    NNtoBoardMap[7] = {'row': 1, 'col': 2}
    NNtoBoardMap[8] = {'row': 2, 'col': 2} 


class GameData:
    def __init__(self):
        pass
    
    # play a random game
    def randomGame(self):
        self.gameBoard = np.zeros((3, 3))
        self.possibleSquares = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        winFlag = False

        self.moves = []
        numMoves = 9
        for i in range(numMoves):
            # add next move to game board if game is not complete
            if not winFlag:
                lastIdx = i
                if i == 0:
                    self.moves.append({'square': self.getSquare(), 'value': 1, 'winner': None})
                    self.fillGameBoard(i)

                else:
                    if self.moves[i-1]['value'] == 1:
                        self.moves.append({'square': self.getSquare(), 'value': -1, 'winner': None})
                    else:
                        self.moves.append({'square': self.getSquare(), 'value': 1, 'winner': None})

                self.fillGameBoard(i)

                if i > 3:
                    result = self.checkWin()
                    if result is not None:
                        self.attachSerialString()
                        winFlag = True
                        self.moves.pop()        # we don't need data for the last move
        
        # give each game a serial number so as to not store repeated games
        if not winFlag:
            self.attachSerialString()
            self.moves.pop()            # we dont care about first move for data


    def getSquare(self):
        # get a random square of possible choices
        possibleSquares = self.possibleSquares
        square = random.choice(possibleSquares)
        possibleSquares.remove(square)
        return square

    def fillGameBoard(self, moveNum):
        # add move to game board
        gameBoard = self.gameBoard
        moves = self.moves
        boardMap = Translators.boardMap

        square = boardMap[moves[moveNum]['square']]
        gameBoard[square['row']][square['col']] = moves[moveNum]['value']

    def checkWin(self):
        gameBoard = self.gameBoard
        # check rows for win
        for i in range(3):
            total = gameBoard[i][0] + gameBoard[i][1] + gameBoard[i][2]
            if total == 3:
                self.populateWinner('X')
                return 'X'
            elif total == -3:
                self.populateWinner('O')
                return 'O'

        # check columns for win
        for i in range(3):
            total = gameBoard[0][i] + gameBoard[1][i] + gameBoard[2][i]
            if total == 3:
                self.populateWinner('X')
                return 'X'
            elif total == -3:
                self.populateWinner('O')
                return 'O'

        # check diags for win
        total = gameBoard[0][0] + gameBoard[1][1] + gameBoard[2][2]
        if total == 3:
            self.populateWinner('X')
            return 'X'
        elif total == -3:
            self.populateWinner('O')
            return 'O'

        total = gameBoard[0][2] + gameBoard[1][1] + gameBoard[2][0]
        if total == 3:
            self.populateWinner('X')
            return 'X'
        elif total == -3:
            self.populateWinner('O')
            return 'O'

        return None

    # add winner to moves sructure
    def populateWinner(self, winner):
        for i in range(len(self.moves)):
            self.moves[i]['winner'] = winner
        return

    # attatch unique serial number to moves structures
    def attachSerialString(self):
        self.serialString = ''
        for i in range(len(self.moves)):
            if self.moves[i]['square'] != 0:
                if self.moves[i]['value'] == 1:
                    mover = 'X'
                elif self.moves[i]['value'] == -1:
                    mover = 'O'
                else:
                    raise Exception("Something wrong here")
                
                self.serialString = self.serialString + str(self.moves[i]['square']) + mover
            else:
                break

        return

# format data for neural net
class DataFormatNeuralNet:
    def __init__(self, gamedata):
        if isinstance(gamedata, list):
            self.gamedata = gamedata
        else:
            pass

    # positional data vs full game data
    def formatToNeuralNet(self):
        self.inputData = []
        gamedata = self.gamedata
        for i in range(len(gamedata)):            
            currentPos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for j in range(len(gamedata[i].moves)):
                if gamedata[i].moves[j]['square'] == 0:
                    break
                currentPos = self.translateToNN(gamedata[i].moves[j], currentPos)
                self.inputData.append(currentPos.copy())

    # translate move structure to neural net structure
    def translateToNN(self, move, lastPos):
        currentPos = lastPos
        if move['value'] == 1:
            currentPos[move['square']-1] = 1
            currentPos[9 + move['square']-1] = 0
        elif move['value'] == -1:
            currentPos[18 + move['square']-1] = 1
            currentPos[9 + move['square']-1] = 0
        else:
            raise Exception('this shouldnt happen')

        return currentPos

    # get input/desired output pairs hashmap
    def getIOPairs(self):

        inputData = self.inputData
        inputData.extend(inputData)

        outputData = self.getDesiredOutput(inputData)
        self.IOmap = {}

        for i in range(len(inputData)):
            self.IOmap[i] = {'input': inputData[i], 'output': outputData[i]} 

        # self.gameBoardIOMap = {}
        # for i in range(len(inputData)):
        #     self.gameBoardIOMap[i] = {'input': self.formatNNtoBoard(inputData[i]), 'output': self.formatNNtoBoard(outputData[i])}


    # desired output is created using minimax recursive algorithm with memoization
    def getDesiredOutput(self, inputData):
        self.idxX = 0
        self.idx_ = 9
        self.idxO = 18
        self.NNnum = 1
        self.other = -1

        outputData = []
        intermediateData = []

        self.positionCache = {}

        # get optimal output given position
        for i in range(len(inputData)):
            if sum(inputData[i][self.idx_:self.idxO]) < 9:         # first move
                # minimax
                gameBoard = self.formatNNtoBoard(inputData[i])
                depth = sum(inputData[i][self.idx_:self.idxO])
                if depth % 2 != 0:
                    self.maxChoice = 'X'
                    self.opponentChoice = 'O'
                else:
                    self.maxChoice = 'O'
                    self.opponentChoice = 'X'

                if depth > 0:
                    NN = 1
                    # run minimax algorithm to get next move
                    nextMove = self.minimax(depth, gameBoard.copy(), NN, self.maxChoice, self.opponentChoice, self.NNnum, self.positionCache)

                    if self.maxChoice == 'X':
                        gameBoard[nextMove[0], nextMove[1]] = 1
                    else:
                        gameBoard[nextMove[0], nextMove[1]] = -1

                interData = (self.formatBoardToNN(gameBoard) - inputData[i])
                if sum(interData[self.idxX:self.idx_] > 0):
                    outputData.append(interData[self.idxX:self.idx_])
                elif sum(interData[self.idxO:self.idxO+9] > 0):
                    outputData.append(interData[self.idxO:self.idxO+9])
                else:
                    raise Exception('Uh oh.')
                # outputData.append(self.formatBoardToNN(gameBoard))

        return outputData

    # add next move
    def makeMove(self, inputData, square, xoDex, _Dex):
        outputData = inputData.copy()
        outputData[xoDex + square - 1] = 1
        outputData[_Dex + square - 1] = 0
        
        return outputData

    # format NN input to gameboard
    @classmethod
    def formatNNtoBoard(self, data):
        # X = 1
        # O = -1
        # Blank = 0
        gameBoard = np.zeros((3, 3))
        idxX = self.idxX
        idx_ = self.idx_
        idxO = self.idxO

        NNtoBoardMap = Translators.NNtoBoardMap

        for i in range(idx_):
            if data[i] == 1:
                square = NNtoBoardMap[i]
                gameBoard[square['row'], square['col']] = 1

        for i in range(idxO, idxO + 9):
            if data[i] == 1:
                square = NNtoBoardMap[i - idxO]
                gameBoard[square['row'], square['col']] = -1

        return gameBoard

    # format NN output to gameboard
    @classmethod
    def formatOutputToBoard(self, dataOutput, dataInput, mover):
        gameBoard = self.formatNNtoBoard(dataInput)

        NNtoBoardMap = Translators.NNtoBoardMap

        for i in range(len(dataOutput)):
            if dataOutput[i] == 1:
                square = NNtoBoardMap[i]
                if mover =='X':
                    gameBoard[square['row'], square['col']] = 1
                else:
                    gameBoard[square['row'], square['col']] = -1

        return gameBoard
        

    ## format gameboard to NN input
    def formatBoardToNN(self, gameBoard):
        data = np.zeros(27)
        for i in range(len(gameBoard)):
            for j in range(len(gameBoard[i])):
                if i == 0 and j == 0:
                    if gameBoard[i][j] == 1:
                        data[0] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 0] = 1
                    else:
                        data[self.idx_ + 0] = 1
                elif i == 1 and j == 0:
                    if gameBoard[i][j] == 1:
                        data[1] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 1] = 1
                    else:
                        data[self.idx_ + 1] = 1
                elif i == 2 and j == 0:
                    if gameBoard[i][j] == 1:
                        data[2] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 2] = 1
                    else:
                        data[self.idx_ + 2] = 1
                elif i == 0 and j == 1:
                    if gameBoard[i][j] == 1:
                        data[3] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 3] = 1
                    else:
                        data[self.idx_ + 3] = 1
                elif i == 1 and j == 1:
                    if gameBoard[i][j] == 1:
                        data[4] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 4] = 1
                    else:
                        data[self.idx_ + 4] = 1
                elif i == 2 and j == 1:
                    if gameBoard[i][j] == 1:
                        data[5] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 5] = 1
                    else:
                        data[self.idx_ + 5] = 1
                elif i == 0 and j == 2:
                    if gameBoard[i][j] == 1:
                        data[6] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 6] = 1
                    else:
                        data[self.idx_ + 6] = 1
                elif i == 1 and j == 2:
                    if gameBoard[i][j] == 1:
                        data[7] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 7] = 1
                    else:
                        data[self.idx_ + 7] = 1
                elif i == 2 and j == 2:
                    if gameBoard[i][j] == 1:
                        data[8] = 1
                    elif gameBoard[i][j] == -1:
                        data[self.idxO + 8] = 1
                    else:
                        data[self.idx_ + 8] = 1
                else:
                    raise Exception('wtf')

        return data

    # minimax recursive algorithm
    @classmethod
    def minimax(self, depth, gameBoard, player, maxChoice, opponentChoice, NNnum, positionCache):

        # check if position is already memoized in hashmap
        hashableGameBoard = self.toHashable(gameBoard.copy(), player)
        if hashableGameBoard in positionCache:
            return positionCache.get(hashableGameBoard)

        # starting metrics
        if player == NNnum:
            best = [-1, -1, -inf]
        else:
            best = [-1, -1, +inf]

        # check if winner, reward earlier wins
        result = self.checkWin(gameBoard) 
        if depth == 0 or result is not None:
            if result == maxChoice:
                score = 10 + depth
            elif result == opponentChoice:
                score = -10 - depth
            else:
                score = 0

            returnVal = [-1, -1, score]

            # add to hashmap
            hashableGameBoard = self.toHashable(gameBoard.copy(), player)
            positionCache.update({hashableGameBoard: returnVal})

            return returnVal

        possibleSquares = self.checkLegal(gameBoard)

        # depth first search on possible squares
        for i in range(len(possibleSquares)):
            row, col = possibleSquares[i][0], possibleSquares[i][1]
            if depth % 2 != 0:
                gameBoard[row][col] = 1
            else:
                gameBoard[row][col] = -1

            # recurse
            score = self.minimax(depth - 1, gameBoard, -player, maxChoice, opponentChoice, NNnum, positionCache)
            gameBoard[row][col] = 0
            score[0], score[1] = row, col

            if player == NNnum:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        # store result in hashmap
        hashableGameBoard = self.toHashable(gameBoard.copy(), -player)
        positionCache.update({hashableGameBoard: best})

        return best

    # check if move is legal
    @classmethod
    def checkLegal(self, gameBoard):
        possibleSquares = []
        for i in range(len(gameBoard)):
            for j in range(len(gameBoard[i])):
                if gameBoard[i][j] == 0:
                    possibleSquares.append([i, j])

        return possibleSquares

    # check if there is a winner
    @classmethod
    def checkWin(self, gameBoard):
        # check rows for win
        for i in range(3):
            total = gameBoard[i][0] + gameBoard[i][1] + gameBoard[i][2]
            if total == 3:
                return 'X'
            elif total == -3:
                return 'O'

        # check columns for win
        for i in range(3):
            total = gameBoard[0][i] + gameBoard[1][i] + gameBoard[2][i]
            if total == 3:
                return 'X'
            elif total == -3:
                return 'O'

        # check diags for win
        total = gameBoard[0][0] + gameBoard[1][1] + gameBoard[2][2]
        if total == 3:
            return 'X'
        elif total == -3:
            return 'O'

        total = gameBoard[0][2] + gameBoard[1][1] + gameBoard[2][0]
        if total == 3:
            return 'X'
        elif total == -3:
            return 'O'

        return None

    # hashable input
    @classmethod
    def toHashable(self, gameBoard, player):
        hashableInput = ''
        for i in range(len(gameBoard)):
            for j in range(len(gameBoard[i])):
                hashableInput = hashableInput + str(int(gameBoard[i][j]))
        return hashableInput + '_' + str(player)

# Run code
if __name__ == "__main__":
    main()
