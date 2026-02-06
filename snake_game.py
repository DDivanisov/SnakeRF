import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20

class SnakeGame:
    
    def __init__(self,ren=True, w=640, h=480, speed=0, test=False):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.speed = speed
        self.test = test

        self.reset()
        self.rendering = ren

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.moves_since_last_food = 0

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.moves = 0
        self._place_food()


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def get_state(self, view_type):
        if view_type == 'directional':
            state = self.get_directional_state()
        return state
    
    def get_directional_state(self):
        """
        Returns the a 1D numpy array.
        Values:
        0 or 1 indicating:
        - Danger straight
        - Danger right
        - Danger left
        - Move direction (left, right, up, down)
        - Food location (food left, food right, food up, food down)
        -Body length??? Should i add this? 
        """
        point_l = Point(self.head.x - BLOCK_SIZE, self.head.y)
        point_r = Point(self.head.x + BLOCK_SIZE, self.head.y)
        point_u = Point(self.head.x, self.head.y - BLOCK_SIZE)
        point_d = Point(self.head.x, self.head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # NEW: Body immediately to the left (relative to direction)
            (dir_u and self._is_body(point_l)) or 
            (dir_d and self._is_body(point_r)) or 
            (dir_r and self._is_body(point_u)) or 
            (dir_l and self._is_body(point_d)),
            
            # NEW: Body immediately to the right (relative to direction)
            (dir_u and self._is_body(point_r)) or 
            (dir_d and self._is_body(point_l)) or 
            (dir_l and self._is_body(point_u)) or 
            (dir_r and self._is_body(point_d)),

            #body length
            len(self.snake)/768,  # Max length is 884 in a 640x480 field

            # Free space in each direction (normalized)
            self._space_in_direction(point_l) / 15,
            self._space_in_direction(point_r) / 15,
            self._space_in_direction(point_u) / 15,
            self._space_in_direction(point_d) / 15,
                    

            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ]
        return np.array(state, dtype=float)
    

    def _count_free_space(self, point, max_depth=10):
        """Count accessible free spaces from a given point using flood fill"""
        if self._is_collision(point):
            return 0
        
        visited = set()
        queue = [point]
        visited.add((point.x, point.y))
        count = 0
        
        while queue and count < max_depth:
            current = queue.pop(0)
            count += 1
            
            # Check all 4 directions
            neighbors = [
                Point(current.x + BLOCK_SIZE, current.y),
                Point(current.x - BLOCK_SIZE, current.y),
                Point(current.x, current.y + BLOCK_SIZE),
                Point(current.x, current.y - BLOCK_SIZE)
            ]
            
            for neighbor in neighbors:
                pos = (neighbor.x, neighbor.y)
                if pos not in visited and not self._is_collision(neighbor):
                    visited.add(pos)
                    queue.append(neighbor)
        
        return count

    def _space_in_direction(self, point):
        """Check how much free space is available in a direction"""
        return self._count_free_space(point, max_depth=15)


    def _is_body(self, point):
        """Check if a point contains a body segment (not head)"""
        for segment in self.snake[1:]:  # Skip head
            if segment.x == point.x and segment.y == point.y:
                return True
        return False
       
    def play_step(self, action, view_type, game):
        self.moves += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        reward = 0

        if self._is_collision():
            game_over = True
            reward = -10
            return game_over, self.score, reward, self.get_state(view_type)
        
        if self.moves_since_last_food > 100 and self.snake.__len__() < 10 and self.test is False:
            game_over = True
            reward = -10
            return game_over, self.score, reward, self.get_state(view_type)

        if self.moves_since_last_food > 1000 and self.test:
            game_over = True
            reward = 0
            return game_over, self.score, reward, self.get_state(view_type)
        
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.moves_since_last_food = 0
            self._place_food()
        else:
            self.moves_since_last_food += 1
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui(game)
        self.clock.tick(self.speed)
        # 6. return game over and score
        return game_over, self.score, reward, self.get_state(view_type)
    
    def _is_collision(self,point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self, game):
        if not self.rendering:
            return
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score) + " Game: " + str(game), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): #straight
            new_dir = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]): #right turn
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        else: #left turn
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            
