from setuptools import setup

setup(
    name='aj_marl',
    version='0.0.1',
    packages=['envs', 'envs.mpe_env', 'envs.mpe_env.multiagent', 'envs.mpe_env.multiagent.scenarios',
              'envs.mpe_env.interactive', 'envs.custom_env', 'tools', 'trainer', 'algorithms', 'algorithms.Config',
              'algorithms.Network', 'algorithms.Trainer', 'algorithms.Trainer.tf1', 'algorithms.Trainer.baseline',
              'algorithms.Trainer.baseline.replay_buffer', 'learning_curve'],
    package_dir={"requirements.txt": "main"},
    url='https://github.com/Gj-12222/ajing_marl',
    license='',
    author='Gj22222',
    author_email='',
    description='this is marl algorithms lab'
)
