import collections
import re
import warnings

import embodied
import numpy as np

'''
this file samples from existing trajectories and trains the world model. 
'''

def test(agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Every(args.log_every)
  should_expl = embodied.when.Until(args.expl_until)
  should_video = embodied.when.Every(args.eval_every)

  dataset = iter(agent.dataset(replay.dataset))
  state = [None]  # To be writable from train step function below.
  assert args.pretrain > 0  # At least one step to initialize variables.
  for _ in range(args.pretrain):
    _, state[0], met = agent.train(next(dataset), state[0])

  metrics = collections.defaultdict(list)
  batch = [None]
  for step in range(int(1e10)):
    for _ in range(args.train_steps):
      batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      [metrics[key].append(value) for key, value in mets.items()]
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
    if step % 100 == 0:
      with warnings.catch_warnings():  # Ignore empty slice warnings.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        print(step)
        for name, values in metrics.items():
          print('train/' + name, np.nanmean(values, dtype=np.float64))
          metrics[name].clear()

    if step % 10000 == 0:
      agent.save(args.logdir+'/'+str(step)+"_")