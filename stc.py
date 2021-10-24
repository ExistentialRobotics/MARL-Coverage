import numpy as np
import math

class SpanningTreeCoveragePolicy(object):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''
    def __init__(self, internal_grid_rad, startpos="bl"):
        super().__init__()

        #describes what subcell we are currently in
        self._curr_pos = startpos
        self._internal_grid_rad = internal_grid_rad

        #reset policy (creates visited array and curr_x, curr_y)

    def pi(self, obs):
        """
        Args:
            obs : an egocentric observation of radius 2 on the map of subcells,
                  obstacles take value 1, free takes zero
        Returns:
            Returns the controls based on the given observation.
        """
        assert obs.shape == (5,5), "wrong observation dummy"
        print(obs)

        # controls cheatsheet
        # 0 - right, 1 - up, 2 - left, 3 - down

        #checking whether the cell is fully explored
        if self.isCellVisited(self._curr_x, self._curr_y):
            #means we got to leave the cell
            if self._curr_pos == "bl":
                u = 3
            elif self._curr_pos == "br":
                u = 0
            elif self._curr_pos == "ur":
                u = 1
            elif self._curr_pos == "ul":
                u = 2

        #exploring further if we can
        else:
            #different checks depending on which sub-cell we are in
            if self._curr_pos == "bl":
                #check if cell below has any obstacles
                obs = obs[2:4][0:2]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y - 1):
                    u = 3
                else:
                    u = 0
            elif self._curr_pos == "br":
                #check if cell right has any obstacles
                obs = obs[3:5][2:4]
                free = not np.any(obs == 1)

                if free == 0 and not self.isAnySubcellVisited(self._curr_x + 1, self._curr_y):
                    u = 0
                else:
                    u = 1
            elif self._curr_pos == "ur":
                #check if cell above has any obstacles
                obs = obs[1:3][3:5]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y + 1):
                    u = 1
                else:
                    u = 2
            elif self._curr_pos == "ul":
                #check if cell above has any obstacles
                obs = obs[0:2][1:3]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x - 1, self._curr_y):
                    u = 2
                else:
                    u = 3

        return u

    def isCellVisited(self,x,y):
        '''
        returns true if all subcells in the enclosing cell of (x,y) are visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2][y:y+2]
        print(cell.shape)
        return np.all(cell == 1)

    def isAnySubcellVisited(self, x, y):
        '''
        returns true if any subcell in the enclosing cell of (x,y) is visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2][y:y+2]
        print(cell.shape)
        return np.any(cell == 1)

    def reset(self):
        '''
        resets the policy to run again on a different environment
        '''
        #internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        #setting the starting x,y
        index = (internal_grid_rad + internal_grid_rad % 2)/2 - 1
        if self._curr_pos == "br":
            xinc = 1
            yinc = 0
        elif self._curr_pos == "ur":
            xinc = 1
            yinc = 1
        elif self._curr_pos == "ul":
            xinc = 0
            yinc = 1
        self._curr_x = 2*index + xinc
        self._curr_y = 2*index + yinc

if __name__ == "__main__":
    #testing spanning tree coverage on dec_grid_rl environment
    pass
