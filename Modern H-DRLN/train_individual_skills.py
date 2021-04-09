import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
from dqn import DQN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MALMOENV')
    parser.add_argument('--mission', type=str,
                        default='missions/mobchase_single_agent.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000,
                        help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1',
                        help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None,
                        help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None,
                        help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1,
                        help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0,
                        help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0,
                        help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int,
                        default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int,
                        default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str,
                        default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync)

    dqn = DQN()

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            print(obs)
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            steps += 1
            print("reward: " + str(reward))
            # print("done: " + str(done))
            print("obs: " + str(obs))
            # print("info" + info)
            if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                h, w, d = env.observation_space.shape
                img = Image.fromarray(obs.reshape(h, w, d))
                img.save('image' + str(args.role) + '_' + str(steps) + '.png')

            time.sleep(.05)

    env.close()
