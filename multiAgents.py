# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
# Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        capsules=successorGameState.getCapsules()
        numCapsules=len(capsules)
        if numCapsules>0:
          CdistMin=min([util.manhattanDistance(newPos,xy2) for xy2 in capsules])
          #CdistMin=min([searchAgents.mazeDistance(newPos,xy2,successorGameState) for xy2 in capsules])
        else:
          CdistMin=99999
        if CdistMin==0:
          CdistMin=10000

        ghostPositions=[newGhostStates[i].getPosition() for i in range(len(newGhostStates))]
        foodCount=newFood.count()
        GdistsFromP=[util.manhattanDistance(newPos,xy2) for xy2 in ghostPositions]
        ClosestGdist=min(GdistsFromP)
        scaredtimeSum=sum(newScaredTimes)
        foodPositions=newFood.asList()
        FdistsFromP=[util.manhattanDistance(newPos,xy2) for xy2 in foodPositions]
        if not len(FdistsFromP)==0:
          FdistMin=min(FdistsFromP)
          #FdistMin=searchAgents.closestFoodDistance(successorGameState)
          FdistMax=max(FdistsFromP)
        else:
          FdistMin=0
          FdistMax=0

        if not scaredtimeSum==0:
          evalue=(20000./float(foodCount+1))+(20./float(ClosestGdist+1))+(10./(float(FdistMin)+1))+(1./(float(FdistMax)+1))+(10./(float(CdistMin)+1))+(20./(numCapsules+1))
        else:
          evalue=(20000./float(foodCount+1))-(20./float(ClosestGdist+1))+(10./float((FdistMin)+1))+(1./(float(FdistMax)+1))+(10./float((CdistMin)+1))+(20./(numCapsules+1))
        return evalue


        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
      """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      "*** YOUR CODE HERE ***"
      return self.value(gameState, 0, self.depth-1)[1]

    def value(self, gameState, agentIndex, depth):
      #terminate when it is a leaf node, i.e. when the game ends
      if gameState.isWin() or gameState.isLose():
        return (self.evaluationFunction(gameState), 'stop')
      #last ghost reached, time to decrease a depth
      elif agentIndex == gameState.getNumAgents():
        return self.value(gameState, 0, depth - 1)
      elif agentIndex > 0: #agent is a ghost
        return self.minvalue(gameState,agentIndex, depth)
      elif agentIndex == 0: #agent is pacman
        return self.maxvalue(gameState,agentIndex, depth)
      else:
        print "ERROR"
        return 0

    def maxvalue(self, gameState, agentIndex, depth):
      v = float("-inf")
      bestAction = 'stop'
      legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
      for action in legalMoves:
        score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)
        if score[0] > v:
          v = score[0]
          bestAction = action
      return (v, bestAction)

    def minvalue(self, gameState, agentIndex, depth):
      v = float("inf")
      bestAction = 'stop'
      #terminate when agent is the final ghost at depth 0
      if agentIndex == (gameState.getNumAgents() - 1) and depth == 0:
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        for action in legalMoves:
          score = (self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), action)
          if score[0] < v:
            bestAction = action
            v = score[0]
        return (v, bestAction)
      else: # keep on recursing
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        for action in legalMoves:
          score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)
          if score[0] < v:
            v = score[0]
            bestAction = action
        return (v, bestAction)  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
      """
        Returns the minimax action using self.depth and self.evaluationFunction
      """
      "*** YOUR CODE HERE ***"
      return self.value(gameState, 0, float("-inf"), float("inf"), self.depth-1)[1]

    def value(self, gameState, agentIndex, alpha, beta, depth):
      #terminate when it is a leaf node, i.e. when the game ends
      if gameState.isWin() or gameState.isLose():
        return (self.evaluationFunction(gameState), 'stop')
      #last ghost reached, time to decrease a depth
      elif agentIndex == gameState.getNumAgents():
        return self.value(gameState, 0, alpha, beta, depth - 1)
      elif agentIndex > 0: #agent is a ghost
        return self.minvalue(gameState,agentIndex, alpha, beta, depth)
      elif agentIndex == 0: #agent is pacman
        return self.maxvalue(gameState,agentIndex, alpha, beta, depth)
      else:
        print "ERROR"
        return 0

    def maxvalue(self, gameState, agentIndex, alpha, beta, depth):
      v = float("-inf")
      bestAction = 'stop'
      legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
      for action in legalMoves:
        score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, alpha, beta, depth)
        if score[0] > v:
          v = score[0]
          bestAction = action
          if v > beta:
            return (v, bestAction)
          alpha = max(v,alpha)
      return (v, bestAction)

    def minvalue(self, gameState, agentIndex, alpha, beta, depth):
      v = float("inf")
      bestAction = 'stop'
       #terminate when agent is the final ghost at depth 0
      if agentIndex == (gameState.getNumAgents() - 1) and depth == 0:
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        for action in legalMoves:
          score = (self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), action)
          if score[0] < v:
            bestAction = action
            v = score[0]
            if v < alpha:
              return (v, bestAction)
            beta = min(beta, v)
        return (v, bestAction)
      else: # keep on recursing
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        for action in legalMoves:
          score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, alpha, beta, depth)
          if score[0] < v:
            v = score[0]
            bestAction = action
            if v < alpha:
              return (v, bestAction)
            beta = min(beta, v)
        return (v, bestAction) 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
      """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
      """
      "*** YOUR CODE HERE ***"
      return self.value(gameState, 0, self.depth-1)[1]

    def value(self, gameState, agentIndex, depth):
      #terminate when it is a leaf node, i.e. when the game ends
      if gameState.isWin() or gameState.isLose():
        return (self.evaluationFunction(gameState), 'stop')
      #last ghost reached, time to decrease a depth
      elif agentIndex == gameState.getNumAgents():
        return self.value(gameState, 0, depth - 1)
      elif agentIndex > 0: #agent is a ghost
        return self.expvalue(gameState,agentIndex, depth)
      elif agentIndex == 0: #agent is pacman
        return self.maxvalue(gameState,agentIndex, depth)
      else:
        print "ERROR"
        return 0

    def maxvalue(self, gameState, agentIndex, depth):
      v = float("-inf")
      bestAction = 'stop'
      legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
      for action in legalMoves:
        score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)
        if score > v:
          v = score
          bestAction = action
      return (v,bestAction)

    def expvalue(self, gameState, agentIndex, depth):
      expvalue = 0
      bestAction = 'stop'
      #terminate when agent is the final ghost at depth 0
      if agentIndex == (gameState.getNumAgents() - 1) and depth == 0:
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        numMoves=len(legalMoves)
        for action in legalMoves:
          score=(self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), action)[0]
          expvalue+=(float(score))*((1./float(numMoves)))
        return (expvalue,'L')
      else: # keep on recursing
        legalMoves = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
        numMoves=len(legalMoves)
        for action in legalMoves:
          score = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)[0]
          if type(score)==tuple:
            score=score[0]
          expvalue+=(float(score))*((1./float(numMoves)))
        return (expvalue,'stop')

        #Returns the expectimax action using self.depth and self.evaluationFunction

        #All ghosts should be modeled as choosing uniformly at random from their
        #legal moves.

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function rewards for the number of food pellets left, penalizes for the manhattan distance to the closest ghost to Pacman, 
    penalizes for the mazedistance to closest food, penalizes for the maximum manhattan distance to food, penalizes for the manhattan distance to the closest capsule, penalizes for the number of capsules,
    and rewards for a higher game score.
  """
  "*** YOUR CODE HERE ***"
  successorGameState = currentGameState
  newPos = successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  capsules=successorGameState.getCapsules()
  numCapsules=len(capsules)
  if numCapsules>0:
    CdistMin=min([util.manhattanDistance(newPos,xy2) for xy2 in capsules])
    #CdistMin=min([searchAgents.mazeDistance(newPos,xy2,successorGameState) for xy2 in capsules])
  else:
    CdistMin=99999
  if CdistMin==0:
    CdistMin=10000

  ghostPositions=[newGhostStates[i].getPosition() for i in range(len(newGhostStates))]
  foodCount=newFood.count()
  GdistsFromP=[util.manhattanDistance(newPos,xy2) for xy2 in ghostPositions]
  ClosestGdist=min(GdistsFromP)
  scaredtimeSum=sum(newScaredTimes)
  foodPositions=newFood.asList()
  FdistsFromP=[util.manhattanDistance(newPos,xy2) for xy2 in foodPositions]
  if not len(FdistsFromP)==0:
    #FdistMin=searchAgents.closestFoodDistance(successorGameState)
    FdistMin=min(FdistsFromP)
    FdistMax=max(FdistsFromP)
  else:
    FdistMin=0
    FdistMax=0

  gameScore=successorGameState.getScore()
  if gameScore>0:
    gameScore=successorGameState.getScore()
  else:
    gameScore=0

  if not scaredtimeSum==0:
    evalue=(20000./float(foodCount+1))+(20./float(ClosestGdist+1))+(1./(float(FdistMin)+1))+(1./(float(FdistMax)+1))+(1./(float(CdistMin)+1))+(20./(numCapsules+1))-(float(10000)/(float(gameScore)+1))
  else:
    evalue=(20000./float(foodCount+1))-(30./float(ClosestGdist+1))+(1./float((FdistMin)+1))+(1./(float(FdistMax)+1))+(1./float((CdistMin)+1))+(20./(numCapsules+1))-(float(10000)/(float(gameScore)+1))

  return evalue

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

