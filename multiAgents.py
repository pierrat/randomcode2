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
import search
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
          CdistMin=min([searchAgents.mazeDistance(newPos,xy2,successorGameState) for xy2 in capsules])
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
          FdistMin=searchAgents.closestFoodDistance(successorGameState)
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
    agent=Agent()
    searchAgent=SearchAgent(agent)
    CDagent=ClosestDotSearchAgent(searchAgent)
    FdistMin=CDagent.findPathToClosestDot(successorGameState)
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

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs BFS on a PositionSearchProblem to find location (1,1)

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
            #print successors
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

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """
    from game import Actions

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
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

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
            dx, dy = game.Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

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
    return len(breadthFirstSearch(prob))


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
        """
      Your expectimax agent (question 4)
    """
    """
    Returns the expectimax action using self.depth and self.evaluationFunction

    All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
    """
    "*** YOUR CODE HERE ***"
