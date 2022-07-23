import numpy as np
import math

# physical/external base state of all entites
class EntityState(object):

    def __init__(self):
        # physical position  2
        self.p_pos = None
        # physical velocity   2
        self.p_vel = None
        # physical roll       1
        self.p_roll = None
        # physical course    1
        self.course_angle = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):

    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # jam
        self.f = None



# action of the agent
class Action(object):

    def __init__(self):
        # physical action
        self.u = None
        # roll v  1
        self.r = None
        # jam times  1
        self.f = None
        # communication action  1
        self.c = None

# properties and state of physical world entity

class Entity(object):
    def __init__(self):

        self.name = ''
        # properties
        self.size = 0.050
        # entity can move / be pushed-

        self.movable = False
        # entity collides with others

        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        self.max_roll = 23.0
        self.max_course = 180.0
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.2

        self.death = False


    @property

    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):

     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities-
class Agent(Entity):

    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.scoutUAV = 1.0
        self.combatUAV = 0.5
        self.electronic_interference = 1.0


# multi-agent world
class World(object):
    # 定义初始化
    def __init__(self):
        # list of agents and entities (can change at execution-time!)

        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 1
        # position dimensionality
        self.dim_p = 2

        self.dim_f = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep dt=0.1
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world

    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]


    # update state of the world

    def step(self):
        # set actions for scripted agents-
        for agent in self.scripted_agents: # NPC
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities

        p_force = [None] * len(self.entities)
        # apply agent physical controls

        p_force  = self.apply_action_force(p_force)
        # apply environment forces

        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:

            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            u = np.zeros(2)
            epislon = 1e-6
            if agent.movable:
                # 弧度值计算
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                F = agent.action.u + noise
                agent.state.p_roll += agent.action.r * self.dt
                if agent.max_roll is not None:
                    if abs(agent.state.p_roll) > agent.max_roll * math.pi / 180:
                        agent.state.p_roll[0] =(agent.state.p_roll / abs(agent.state.p_roll)) * agent.max_roll * math.pi / 180
                agent_course_angular_velocity = 9.81 * 1.2 / ((agent.action.u + epislon) * self.dt) * math.tan(agent.state.p_roll)
                agent.state.course_angle += agent_course_angular_velocity * self.dt
                if agent.max_course is not None:
                    if abs(agent.state.course_angle) > agent.max_course * math.pi /180:
                        agent.state.course_angle[0] = (agent.state.course_angle / abs(agent.state.course_angle)) * agent.max_course * math.pi /180

                u[0] = F * math.sin(agent.state.course_angle)
                u[1] = F * math.cos(agent.state.course_angle)
                p_force[i] = u

        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response

        p = p_force
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue

            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                # dv =F/m·dt   V <— V + dv
                entity.state.p_vel += p_force[i] / entity.mass * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed

            entity.state.course_angle[0] = math.atan2(entity.state.p_vel[1], entity.state.p_vel[0])

            entity.state.p_pos += entity.state.p_vel * self.dt  # x(t) = x(t) + vx(t) * dt
            # ban the size of bound range
            if entity.death == False:
                if abs(entity.state.p_pos[0]) > 2.0:
                    entity.state.p_pos[0] = 2.0 * (entity.state.p_pos[0] /abs(entity.state.p_pos[0]))
                if abs(entity.state.p_pos[1]) > 2.0:
                    entity.state.p_pos[1] = 2.0 * (entity.state.p_pos[1] /abs(entity.state.p_pos[1]))

            # jam dynamic step
            if entity.state.f[0] < 0:
                entity.state.f = np.zeros(1)
            else:
                entity.state.f -= entity.action.f


    def update_agent_state(self, agent):
        # set communication state (directly for now)

        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):  #

        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        epsilon_dist = 1e-6  # NAN
        # minimum allowable distance-

        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        # np.logaddexp： log(exp(x1) + exp(x2))

        penetration = np.logaddexp(0, - (dist - dist_min) / k ) * k
        force = self.contact_force * delta_pos / (dist+ epsilon_dist) * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        return [force_a, force_b]