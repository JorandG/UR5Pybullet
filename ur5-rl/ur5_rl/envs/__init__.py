from gym.envs.registration import register

register(
    id='ur5reach-v0',
    entry_point='gym_pybullet.envs:Ur5ReachEnv',
)

