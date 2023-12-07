# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.
import EasyGA

from src.kesslergame import KesslerController, TrainerEnvironment, GraphicsType, Scenario
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import random
import matplotlib as plt


class Group12Controller(KesslerController):

    def __init__(self):
        self.eval_frames = 0 # What is this?

        ga = EasyGA.GA()
        ga.gene_impl = lambda: self.generate_chromosome() # so each gene is a value from 0 to 1 (representing 0% to 100%)
        ga.chromosome_length = 11

        ga.population_size = 10 # this is chosen completely randomly lol
        ga.generation_goal = 10 # this is chosen completely randomly lol

        ga.target_fitness_type = 'max'
        ga.fitness_function_impl = self.fitness

        ga.evolve()
        ga.print_best_chromosome()
        # chromosome for thrust, turn_rate, fire
    #
    def generate_chromosome(self):
        a = round(random.uniform(-1, 0.98), 2)
        b = round(random.uniform(a+0.01, 1), 2)
        c = round(random.uniform(b+0.01, 1), 2)

        return [a, b, c]

    def fitness(self, chromosome):
        self.set_up_fuzzy_system(chromosome)

        my_test_scenario = Scenario(name='Train Scenario',
                            num_asteroids=5, # changed 10 to 5
                            ship_states=[
                                {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 1}
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)
        
        game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None}
        
        game = TrainerEnvironment(settings=game_settings)
        score, perf_data = game.run(scenario=my_test_scenario, controllers=[self])

        return score.teams[0].asteroids_hit

    def set_up_fuzzy_system(self, chromosome):
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        # Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)

        closest_asteroid_ship_dist = ctrl.Antecedent(np.arange(0, 1281, 1), 'closest_asteroid_ship_dist')
        closest_asteroid_ship_dist['C'] = fuzz.trimf(closest_asteroid_ship_dist.universe, [0, 0, 100])
        closest_asteroid_ship_dist['F'] = fuzz.trimf(closest_asteroid_ship_dist.universe, [100, 1281, 1281])

        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [180*chromosome[0].value[0], 180*chromosome[0].value[1], 180*chromosome[0].value[2]]) # [-180,-180,-30]
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [180*chromosome[1].value[0], 180*chromosome[1].value[1], 180*chromosome[1].value[2]]) # [-90,-30,0]
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [180*chromosome[2].value[0], 180*chromosome[2].value[1], 180*chromosome[2].value[2]]) # [-30,0,30]
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [180*chromosome[3].value[0], 180*chromosome[3].value[1], 180*chromosome[3].value[2]]) # [0,30,90]
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [180*chromosome[4].value[0], 180*chromosome[4].value[1], 180*chromosome[4].value[2]]) # [30,180,180]

        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        # Declare singleton fuzzy sets for the ship_fire consequent;
        # -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [chromosome[5].value[0],chromosome[5].value[0],chromosome[5].value[1]]) # [-1,-1,0.0]
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [chromosome[6].value[0],chromosome[6].value[1],chromosome[6].value[1]]) # [0.0,1,1]

        ship_thrust = ctrl.Consequent(np.arange(-500, 500, 10), 'ship_thrust')
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_thrust['RF'] = fuzz.trimf(ship_thrust.universe, [500*chromosome[7].value[0], 500*chromosome[7].value[1], 500*chromosome[7].value[2]]) # [-500, -500, -250]
        ship_thrust['RS'] = fuzz.trimf(ship_thrust.universe, [500*chromosome[8].value[0], 500*chromosome[8].value[1], 500*chromosome[8].value[2]]) # [-250, -100, 0]
        ship_thrust['FS'] = fuzz.trimf(ship_thrust.universe, [500*chromosome[9].value[0], 500*chromosome[9].value[1], 500*chromosome[9].value[2]]) # [0, 100, 250]
        ship_thrust['FF'] = fuzz.trimf(ship_thrust.universe, [500*chromosome[10].value[0], 500*chromosome[10].value[1], 500*chromosome[10].value[2]]) # [250, 500, 500]

        # Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        rule16 = ctrl.Rule(closest_asteroid_ship_dist['C'], ship_thrust['RF'])
        rule17 = ctrl.Rule(closest_asteroid_ship_dist['F'], ship_thrust['FF'])

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t
        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['closest_asteroid_ship_dist'] = math.sqrt(asteroid_ship_x**2 + asteroid_ship_y**2)

        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = shooting.output['ship_thrust']
        
        self.eval_frames +=1
        
        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "Group 12 Controller"
    
if __name__=="__main__":
    controller = Group12Controller()