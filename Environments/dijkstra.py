import numpy as np
from queue import PriorityQueue


def in_bounds(x, y, grid):
    return x >= 0 and y >=0 and x < grid.shape[0] and y < grid.shape[1]

def get_valid_neighbors(x, y, grid, visited):
    """
    Args:
       x : the x position on the grid
       y : the y position
       grid : an array representing the environment, 0 is explored,
             1 is unexplored, and -1 is obstacle
       visited: an array representing whether we have visited a cell
                or not, 1 is visited, 0 is not visited

    Returns:
       A list of valid neighbors and their coordinates
    """
    neighbors = []

    if in_bounds(x+1, y, grid) and visited[x+1][y] == 0 and grid[x+1][y] != -1:
        neighbors.append((x+1,y))

    if in_bounds(x-1, y, grid) and visited[x-1][y] == 0 and grid[x-1][y] != -1:
        neighbors.append((x-1,y))

    if in_bounds(x, y+1, grid) and visited[x][y+1] == 0 and grid[x][y+1] != -1:
        neighbors.append((x,y+1))

    if in_bounds(x, y-1, grid) and visited[x][y-1] == 0 and grid[x][y-1] != -1:
        neighbors.append((x,y-1))

    return neighbors


def dijkstra(grid):
    """
    Args:
       grid : an array representing the environment, 1 is explored,
             0 is unexplored, and -1 is obstacle

    Returns:
       a cost array representing the cost from the closest unexplored node

    Notes:
    we will assume that the goals points are unexplored cells
    """
    open_set = PriorityQueue()
    visited = np.zeros(grid.shape)
    cost = -1*np.ones(grid.shape)

    #adding all unexplored cells to the open set
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            #checking if cell is unexplored
            if grid[i][j] == 0:
                #adding unexplored cell to open set
                open_set.put((0, (i,j)))

    #main dijkstra loop
    while not open_set.empty():
        cell = open_set.get()

        #check if cell has already been visited
        if visited[cell[1][0]][cell[1][1]] == 1:
            continue

        #mark as visited, finalize cost
        visited[cell[1][0]][cell[1][1]] = 1
        cost[cell[1][0]][cell[1][1]] = cell[0]

        #looping over all neighbors and updating their costs
        neighbors = get_valid_neighbors(cell[1][0], cell[1][1], grid, visited)

        for neighbor in neighbors:
            open_set.put((cell[0] + 1, neighbor))

    return cost


if __name__ == "__main__":
    grid = np.ones((10,10))
    grid = np.pad(grid,((1,1),(1,1)), constant_values = ((0,0),(0,0)))
    print(dijkstra(grid))

