# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import Directions
from game import Agent
from game import Actions
import util
import time
#import search
import random

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
          #CdistMin=min([util.manhattanDistance(newPos,xy2) for xy2 in capsules])
          CdistMin=min([mazeDistance(newPos,xy2,successorGameState) for xy2 in capsules])
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
          FdistMin=closestFoodDistance(successorGameState)
          #FdistMin=min(FdistsFromP)
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
  from game import Agent

  successorGameState = currentGameState
  newPos = successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  capsules=successorGameState.getCapsules()
  numCapsules=len(capsules)
  if numCapsules>0:
    #CdistMin=min([util.manhattanDistance(newPos,xy2) for xy2 in capsules])
    CdistMin=min([mazeDistance(newPos,xy2,successorGameState) for xy2 in capsules])
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
    #agent=Agent()
    #searchAgent=SearchAgent(agent)
    #CDagent=ClosestDotSearchAgent(searchAgent)
    FdistMin=closestFoodDistance(successorGameState)
    #FdistMin=min(FdistsFromP)
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
    evalue=(20000./float(foodCount+1))-(1./float(ClosestGdist+1))+(1./float((FdistMin)+1))+(1./(float(FdistMax)+1))+(1./float((CdistMin)+1))+(20./(numCapsules+1))-(float(10000)/(float(gameScore)+1))

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
      return self.value(gameState, 0, self.depth-1)[1]

    def value(self, gameState, agentIndex, depth):
      #terminate when it is a leaf node, i.e. when the game ends
      if gameState.isWin() or gameState.isLose():
        return (betterEvaluationFunction(gameState), 'stop')
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
          score=(betterEvaluationFunction(gameState.generateSuccessor(agentIndex, action)), action)[0]
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




class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    

    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    '''
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    '''
    from game import Directions
    
    fringe=util.Stack()
    state=problem.getStartState()
    
    plans={}
    closed=set()
    successors=problem.getSuccessors(state)
    #plans dict gets some keys (state tuples) and values (directions)
    for i in successors:
        fringe.push(i)
        plans[i]=i[1]
    closed.add(state)
    
    
    while not problem.isGoalState(state):
        if not fringe:
            print "FAILURE"
            return none
        stateFull=fringe.pop()
        state=stateFull[0]
        #if this is the goal state, then return the value of the current state in the dict (winning plan)
        if problem.isGoalState(state):
            movedir=stateFull[1]
            plan=plans[stateFull]
            return list(plan) 
        if state not in closed:
            closed.add(state)
            successors=problem.getSuccessors(state)
            #add successors to the fringe, and also create dict entries for them
            #that depend on the plan of their parent node (the current state)
            #and their own direction from the current state
            for i in successors:
                if type(plans[stateFull])==tuple:
                    plans[i]=plans[stateFull]+(i[1],)
                else: #single case, plans[state] is a string
                    plans[i]=plans[stateFull],i[1]
                fringe.push(i)
                                    
    return []
    

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    "*** YOUR CODE HERE ***"
    
    from game import Directions
    
    fringe=util.Queue()
    state=problem.getStartState()
    
    plans={}
    closed=set()
    successors=problem.getSuccessors(state)
    #plans dict gets some keys (state tuples) and values (directions)
    for i in successors:
        fringe.push(i)
        plans[i]=i[1]
    closed.add(state)
    
    while not problem.isGoalState(state):
        if not fringe:
            print "FAILURE"
            return none
        stateFull=fringe.pop()
        state=stateFull[0]
        #if this is the goal state, then return the value of the current state in the dict (winning plan)
        if problem.isGoalState(state):
            movedir=stateFull[1]
            plan=plans[stateFull]
            if type(plan)==str:
                plan=[plan]
            else:
                plan=list(plan)
            return plan #changed from list(plan) to just plan
        if state not in closed:
            closed.add(state)
            successors=problem.getSuccessors(state)
            #add successors to the fringe, and also create dict entries for them
            #that depend on the plan of their parent node (the current state)
            #and their own direction from the current state
            for i in successors:
                if type(plans[stateFull])==tuple:
                    plans[i]=plans[stateFull]+(i[1],)
                else: #single case, plans[state] is a string
                    plans[i]=plans[stateFull],i[1]
                fringe.push(i)

    return []

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    
    from game import Directions
    
    fringe=util.PriorityQueue()
    state=problem.getStartState()
    
    plans={}
    closed=set()
    successors=problem.getSuccessors(state)
    #plans dict gets some keys (state tuples) and values (directions)
    for i in successors:
        fringe.push(i,problem.getCostOfActions([i[1]]))
        plans[i]=i[1]
    closed.add(state)
    
    
    while not problem.isGoalState(state):
        if not fringe:
            print "FAILURE"
            return none
        stateFull=fringe.pop()
        state=stateFull[0]
        #if this is the goal state, then return the value of the current state in the dict (winning plan)
        if problem.isGoalState(state):
            movedir=stateFull[1]
            plan=plans[stateFull]
            return list(plan)
        if state not in closed:
            closed.add(state)
            successors=problem.getSuccessors(state)
            #add successors to the fringe, and also create dict entries for them
            #that depend on the plan of their parent node (the current state)
            #and their own direction from the current state
            for i in successors:
                if type(plans[stateFull])==tuple:
                    plans[i]=plans[stateFull]+(i[1],)
                else: #single case, plans[state] is a string
                    plans[i]=plans[stateFull],i[1]
                fringe.push(i,problem.getCostOfActions(list(plans[i])))

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    from game import Directions
    
    fringe=util.PriorityQueue()
    state=problem.getStartState()
    
    plans={}
    closed=set()
    successors=problem.getSuccessors(state)
    #plans dict gets some keys (state tuples) and values (directions)
    for i in successors:
        fringe.push(i,problem.getCostOfActions([i[1]])+(heuristic(i[0],problem)))
        plans[i]=i[1]
    closed.add(state)
    
    
    while not problem.isGoalState(state):
        if not fringe:
            print "FAILURE"
            return none
        stateFull=fringe.pop()
        state=stateFull[0]
        #if this is the goal state, then return the value of the current state in the dict (winning plan)
        if problem.isGoalState(state):
            movedir=stateFull[1]
            plan=plans[stateFull]
            return list(plan)
        if state not in closed:
            closed.add(state)
            successors=problem.getSuccessors(state)
            #add successors to the fringe, and also create dict entries for them
            #that depend on the plan of their parent node (the current state)
            #and their own direction from the current state
            for i in successors:
                if type(plans[stateFull])==tuple:
                    plans[i]=plans[stateFull]+(i[1],)
                else: #single case, plans[state] is a string
                    plans[i]=plans[stateFull],i[1]
                fringe.push(i,problem.getCostOfActions(list(plans[i]))+(heuristic(i[0],problem)))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        print self.actions
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded

        "*** YOUR CODE HERE ***"
        #the starting state is a tuple of tuples
        #the first tuple is a tuple containing all the corners, which will be subtracted out
        #with each one visited. the second is the position of the pacman
        self.startState = (self.corners, self.startingPosition) 

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** YOUR CODE HERE ***"
        return self.startState


    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** YOUR CODE HERE ***"
        #when there is no more corners left in the corners "checklist" we can say goal reached
        return len(state[0]) == 0


    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x, y = state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextMetaState = [] #nextMetaState contains the corners that still need to be visited
                for corner in state[0]:
                    if not (nextx,nexty) == corner: #a corner has not been reached so retain it
                        nextMetaState.append(corner)

                cost = 1
                successors.append( ( (tuple(nextMetaState),(nextx,nexty)), action, cost) )

        #self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    #the corner heuristic is the manhattan distance to the nearest corner plus the manhattan distance of the rest of the corners
    metaState, pacposition = state
    if len(metaState) == 0: #goal reached
        return 0
    listmetaState = list(metaState) #set of all the corners left to visit by the pacman
    #find the nearest corner left to visit by manhattan distance
    #then find the nearest corner left to visit from that corner by manhattan distance
    #continue until all corners have been reached and sum the manhattan distances
    returnheuristic = 0 #heuristic to be returned
    positionOfInterest = pacposition #initial position is pacman's location
    while not len(listmetaState) == 0: #continune loop until the set of corners to visit becomes empty
        #initial condition is set to be the first corner
        shortestdistance = util.manhattanDistance(positionOfInterest,listmetaState[0])
        closestcorner = listmetaState[0] 
        for cornerleft in listmetaState:
            cornerdistance = util.manhattanDistance(positionOfInterest,cornerleft)
            if cornerdistance < shortestdistance:
                shortestdistance = cornerdistance
                closetcorner = cornerleft
        returnheuristic += shortestdistance #add this manhattan distance on
        listmetaState.remove(closestcorner) #since we just "visited" the corner, remove it from our set to visit
        positionOfInterest = closestcorner #now we want to look from the perspective of the closest corner next
        
    return returnheuristic

    #return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
     For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
     if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
     """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    #the food heuristic is the manhattan distance to the nearest food plus the manhattan distance of the rest of the foods
    listmetaState = foodGrid.asList() #list of all the food left to visit by the pacman
    if len(listmetaState) == 0: #goal reached
        return 0

    if len(listmetaState) > 1:
        #find the foods with the furthest maze distance from each other
        longestdistance = 0
        twofurthestfoods = [listmetaState[0],listmetaState[0]]
        for i in listmetaState:
            for j in listmetaState:
                fooddistance = mazeDistance(i,j,problem.startingGameState)
                if fooddistance > longestdistance:
                    longestdistance = fooddistance
                    twofurthestfoods = [i,j]
        #find the pacman's distance to the closer of the two foods found and add it to the longest distance
        minDist = min(mazeDistance(twofurthestfoods[0],position,problem.startingGameState),mazeDistance(twofurthestfoods[1],position,problem.startingGameState))
        return max(longestdistance + minDist, len(listmetaState))
    else: #only one food left
        return util.manhattanDistance(position, listmetaState[0])

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        from search import breadthFirstSearch
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        path = breadthFirstSearch(problem)
        return path

class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state
        isGoal=self.food[x][y]

        '''
        foodTuples=self.food.asList()
        dists=util.Counter()
        for food in foodTuples:
            distance=util.manhattanDistance(state,food)
            dists[food]=distance
        shortestDistance=min(dists.values())
        closestFood=dists[shortestDistance]
        isGoal=state==closestFood
        '''

        return isGoal

        #util.raiseNotDefined()

##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        path = breadthFirstSearch(problem)
        return path

    def getAction(self, state):
        """
        From game.py:
        The Agent will receive a GameState and must return an action from
        Directions.{North, South, East, West, Stop}
        """
        "*** YOUR CODE HERE ***"
        print "hello"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(bfs(prob))

def closestFoodDistance(gameState):
    """
    Returns the the distance to the closest food
    """
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)
    path = breadthFirstSearch(problem)
    return len(path)