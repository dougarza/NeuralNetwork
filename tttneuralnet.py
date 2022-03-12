import numpy as np
import pickle
from math import exp
import random
from generatedata import DataFormatNeuralNet
from generatedata import GameData

# Create and trian tictactoe neural net
def main():
    # Inputs
    loadData = True
    tttNet = NeuralNetwork()    # create tictactoe neural net
    # fileName = 'NN9.pkl'
    # tttNet = loadNet(fileName)

    # test and train loop for all data files
    for i in range(10):
        # Load data
        fileName = 'data' + str(i) + '.pkl'
        if loadData:
            games = loadGames(fileName)

        # results variables (stores wins, losses, draws, reasons for loss (legal or illegal))
        resultsInter = []
        resultsTotal = []
        totalOut = []
        wins = []
        losses = []
        draws = []
        numTests = 100
        iter = 0
        while True:
            resultsInter = tttNet.performance(numTests)     # test tttnet performance

            # store results
            resultsTotal.append(resultsInter)
            totalOut.append(resultsInter.shape[0])
            wins.append(0)
            losses.append(0)
            draws.append(0)
            for j in range(resultsInter.shape[0]):
                if resultsInter[j, 0] == 'M':
                    losses[iter] += 1
                elif resultsInter[j, 0] == 'D':
                    draws[iter] += 1
                elif resultsInter[j, 0] == 'N':
                    wins[iter] += 1
                else:
                    raise Exception('Invalid Result')

            idx1 = iter * 5000

            idx2 = (iter + 1) * 5000
            if idx2 > len(games.IOmap):
                idx2 = len(games.IOmap)

            # create input/ouput map for neural net training
            inputGames = DataFormatNeuralNet(False)
            inputGames.IOmap = {}
            for j in range(idx1, idx2):
                inputGames.IOmap[j] = {'input': games.IOmap[j]['input'], 'output': games.IOmap[j]['output']}
            
            firstDex = idx1
            tttNet.train(inputGames, firstDex)      # train neural net

            if idx2 == len(games.IOmap):
                break

            iter += 1

        # save this iteration of neural net
        netName = 'NN' + str(i) + '.pkl'
        with open(netName, 'wb') as f: 
            pickle.dump(tttNet, f) 
    
# load data for neural net
def loadGames(fileName):
    # Data File
    with open(fileName, 'rb') as f:  # Python 3: open(..., 'rb')
        games = pickle.load(f)
    return games

# load saved iteration of neural net
def loadNet(fileName):
    # Data File
    with open(fileName, 'rb') as f:  # Python 3: open(..., 'rb')
        myNN = pickle.load(f)
    return myNN

# neural net class
class NeuralNetwork(object):
    def __init__(self):
        self.batchSize = 16         # training batch size          
        self.r = 0.075              # learning rate

        self.inputSize = 27         # input layer
        self.layer2Size = 81        # hidden layer
        # self.layer3Size = 81
        self.outputSize = 9         # outputs

        # initialize random weights
        self.W1 = np.random.randn(self.inputSize, self.layer2Size)      
        self.W2 = np.random.randn(self.layer2Size, self.outputSize)
        # self.W3 = np.random.randn(self.layer3Size, self.outputSize)

        # initialize zeros biases
        self.B1 = np.zeros((1, self.layer2Size))
        # self.B2 = np.zeros((1, self.layer3Size))
        self.B2 = np.zeros((1, self.outputSize))

    def forward(self, X):
        # forward propagation
        self.z2 = np.dot(X, self.W1) + self.B1      # input x weights + biases
        self.a2 = self.ReLU(self.z2)                # relu activation function

        self.z3 = np.dot(self.a2, self.W2) + self.B2
        self.a3 = self.softmax(self.z3)             # softmax activation function

        # self.z4 = np.dot(self.a3, self.W3) + self.B3
        # self.a4 = self.softmax(self.z4)

        return self.a3

    def backward(self, X, y, a3):
        # back propagation to train

        # unbatch layers
        # unbatchA4 = (1 / a4.shape[0]) * np.sum(a4.copy(), axis=0, dtype='float64')
        unbatchA3 = (1 / self.a3.shape[0]) * np.sum(self.a3.copy(), axis=0, dtype='float64').reshape(1, self.a3.shape[1])
        unbatchA2 = (1 / self.a2.shape[0]) * np.sum(self.a2.copy(), axis=0, dtype='float64').reshape(1, self.a2.shape[1])
        unbatchX = (1 / X.shape[0]) * np.sum(X.copy(), axis=0, dtype='float64').reshape(1, X.shape[1])

        # error and gradient change
        self.errorOutput = (1 / a3.shape[0]) * np.sum(y - a3, axis=0, dtype='float64').reshape(1, self.a3.shape[1])
        self.deltaOutput = self.errorOutput.dot(self.dsoftmax(unbatchA3))

        # self.errorA3 = self.deltaOutput.dot(self.W3.T)
        # self.deltaA3 = self.errorA3 * self.dReLU(unbatchA3)

        # error and gradient change of next layer
        self.errorA2 = self.deltaOutput.dot(self.W2.T)
        self.deltaA2 = self.errorA2 * self.dReLU(unbatchA2)

        # new weights
        self.W1 -= self.r * unbatchX.T.dot(self.deltaA2)
        self.W2 -= self.r * unbatchA2.T.dot(self.deltaOutput)
        # self.W3 -= self.r * unbatchA3.T.dot(self.deltaOutput)

        # new biases
        self.B1 -= self.r * self.deltaA2
        self.B2 -= self.r * self.deltaOutput
        # self.B3 -= self.r * self.deltaOutput

    def ReLU(self, s):
        # ReLU activation function
        output = np.maximum(0, s)
        return output

    def dReLU(self, s):
        # derivative of ReLU activation function
        s[s<=0] = 0
        s[s>0] = 1
        return s

    def softmax(self, s):
        # softmax activation function
        output = np.exp(s) / np.sum(np.exp(s))
        return output

    def dsoftmax(self, s):
        # derivative of softmax activation function
        v = np.sum(s, axis=0, dtype='float64').reshape(-1, 1)
        output = np.diagflat(v) - np.dot(v, v.T)
        return output

    def train(self, games, firstDex):
        # training method
        idx = firstDex

        # loop through all input/desired output pairs in hashmap
        while idx < len(games.IOmap):

            # input data
            X = []          # input batch
            y = []          # target batch
            for i in range(self.batchSize):
                X.append(games.IOmap[idx]['input'])
                y.append(games.IOmap[idx]['output'])
                idx += 1
                if idx >= len(games.IOmap):
                    return

            # forward prop
            X = np.array(X)
            y = np.array(y)
            output = self.forward(X)

            # backprop comparing output and desired output
            self.backward(X, y, output)

    def performance(self, numGames):
        # test performance of NN against minimax
        iter = 0

        # store results here
        resultArray = np.empty([numGames, 2], dtype = str)

        DataFormatNeuralNet(False)

        idxX = 0
        idx_ = 9
        idxO = 18

        # grab index pointers
        DataFormatNeuralNet.idxX = idxX
        DataFormatNeuralNet.idx_ = idx_
        DataFormatNeuralNet.idxO = idxO
        DataFormatNeuralNet.positionCache = {}
        DataFormatNeuralNet.NNnum = 1
        player = -1
        while iter < numGames:
            XorO = ['X', 'O']
            playAs = random.choice(XorO)            # play as random x or o

            # first move
            possibleSquares = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            startSquare = random.choice(possibleSquares)

            X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            X[idxX + startSquare - 1] = 1
            X[idx_ + startSquare - 1] = 0

            X = np.array(X)

            if playAs == 'X':
                # minimax's move 
                DataFormatNeuralNet.maxChoice = 'O'
                DataFormatNeuralNet.opponentChoice = 'X'
                depth = sum(X[idx_:idxO])

                # gameboard from input
                gameBoard = DataFormatNeuralNet.formatNNtoBoard(X.copy().tolist())
                while True:
                    # minimax moved, redraw gameboard
                    nextMove = DataFormatNeuralNet.minimax(depth, gameBoard.copy(), player, DataFormatNeuralNet.maxChoice, DataFormatNeuralNet.opponentChoice, DataFormatNeuralNet.NNnum, DataFormatNeuralNet.positionCache)
                    gameBoard[nextMove[0], nextMove[1]] = -1
                    X = DataFormatNeuralNet.formatBoardToNN(DataFormatNeuralNet, gameBoard)
                    depth -= 1

                    # check if game is complete (winner, draw)
                    if DataFormatNeuralNet.checkWin(gameBoard) is not None:
                        if DataFormatNeuralNet.checkWin(gameBoard) == playAs:
                            winner = "NN"
                        else:
                            winner = "Minimax"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

                    if depth <= 0:
                        winner = "Draw"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

                    # NN's move
                    output = self.forward(X)
                    outputAdjusted = (np.round(output[0])).astype(dtype='float64')          # WCAFTL

                    dummyInput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    outputMove = DataFormatNeuralNet.formatOutputToBoard(outputAdjusted.copy().tolist(), dummyInput, 'X')

                    possibleSquares = DataFormatNeuralNet.checkLegal(gameBoard)
                    legalFlag = False

                    # check if NN made a legal move
                    for i in range(len(possibleSquares)):
                        row, col = possibleSquares[i][0], possibleSquares[i][1]
                        if outputMove[row][col] != 0:
                            legalFlag = True
                            break
                    
                    # end game if NN made an illegal move
                    if not legalFlag:
                        winner = "Minimax"
                        reason = "Illegal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break
                    
                    # redraw gameboard
                    gameBoard = DataFormatNeuralNet.formatOutputToBoard(outputAdjusted.copy().tolist(), X.copy().tolist(), 'X')

                    # check if win
                    if DataFormatNeuralNet.checkWin(gameBoard) is not None:
                        if DataFormatNeuralNet.checkWin(gameBoard) == playAs:
                            winner = "NN"
                        else:
                            winner = "Minimax"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break
                        
                    if depth <= 0:
                        winner = "Draw"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break
                        
            elif playAs == 'O':
                # minimax's move
                DataFormatNeuralNet.maxChoice = 'X'
                DataFormatNeuralNet.opponentChoice = 'O'
                depth = sum(X[idx_:idxO])
                while True:
                    # NN's move
                    output = self.forward(X)
                    outputAdjusted = (np.round(output[0])).astype(dtype='float64')          # WCAFTL

                    dummyInput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    outputMove = DataFormatNeuralNet.formatOutputToBoard(outputAdjusted.copy().tolist(), dummyInput, 'O')

                    gameBoard = DataFormatNeuralNet.formatNNtoBoard(X.copy().tolist())
                    possibleSquares = DataFormatNeuralNet.checkLegal(gameBoard)
                    legalFlag = False
                    for i in range(len(possibleSquares)):
                        row, col = possibleSquares[i][0], possibleSquares[i][1]
                        if outputMove[row][col] != 0:
                            legalFlag = True
                            break

                    if not legalFlag:
                        winner = "Minimax"
                        reason = "Illegal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

                    gameBoard = DataFormatNeuralNet.formatOutputToBoard(outputAdjusted.copy().tolist(), X.copy().tolist(), 'O')

                    if DataFormatNeuralNet.checkWin(gameBoard) is not None:
                        if DataFormatNeuralNet.checkWin(gameBoard) == playAs:
                            winner = "NN"
                        else:
                            winner = "Minimax"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break
                        
                    if depth <= 0:
                        winner = "Draw"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

                    # Minimax's move
                    # gameBoard = DataFormatNeuralNet.formatNNtoBoard(X.copy().tolist())
                    nextMove = DataFormatNeuralNet.minimax(depth, gameBoard.copy(), player, DataFormatNeuralNet.maxChoice, DataFormatNeuralNet.opponentChoice, DataFormatNeuralNet.NNnum, DataFormatNeuralNet.positionCache)
                    gameBoard[nextMove[0], nextMove[1]] = 1
                    X = DataFormatNeuralNet.formatBoardToNN(DataFormatNeuralNet, gameBoard)
                    depth -= 1
                    if DataFormatNeuralNet.checkWin(gameBoard) is not None:
                        if DataFormatNeuralNet.checkWin(gameBoard) == playAs:
                            winner = "NN"
                        else:
                            winner = "Minimax"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

                    if depth <= 0:
                        winner = "Draw"
                        reason = "Legal"
                        resultArray[iter, 0] = winner
                        resultArray[iter, 1] = reason
                        break

            else:
                raise Exception("Choose a side.")

            iter += 1

        return resultArray

if __name__ == "__main__":
    main()
