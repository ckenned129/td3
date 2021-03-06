import gym
import torch
import time
from models import TD3


def train(env_name, warmup_iter=int(25e3), train_iter=int(1e6)):

    env = gym.make(env_name)
    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    u_max = float(env.action_space.high[0])
    td3 = TD3(x_dim, u_dim, u_max)
    if torch.cuda.is_available():
        td3.actor.cuda()
        td3.critic_1.cuda()
        td3.critic_2.cuda()

    print('pre_train: ', eval(env_name, td3.actor))
    x = env.reset()
    for i in range(warmup_iter):
        u_random = env.action_space.sample()
        x_next, reward, done, info = env.step(u_random)

        td3.replay_buffer.append([x, u_random, x_next, reward, done])

        #td3.add_replay_buffer_sample(x, u_random, x_next, reward, done)
        #print(x.shape, x_next.shape, x_dim)
        #step = torch.tensor([x, u_random, x_next, reward, float(done)]).view(1, 5)

        #td3.replay_buffer = torch.cat((td3.replay_buffer, step), dim=1)
        #td3.replay_buffer.append([x, u_random, x_next, reward, done])
        x = x_next

        if done:
            x = env.reset()
            done = False

            #ep_reward = 0
            #ep_steps = 0
            #ep_num += 1

    for i in range(train_iter):
        if i % 1e4 == 0:
            print('i: ', i,  eval(env_name, td3.actor))
        if i % 1e5 == 0:
            torch.save(td3.actor.state_dict(), f'td3_hop_actor_{time.strftime(r"%m_%d_%H:%M")}.pt')
            torch.save(td3.critic_1.state_dict(), f'td3_hop_critic_1_{time.strftime(r"%m_%d_%H:%M")}.pt')
            torch.save(td3.critic_2.state_dict(), f'td3_hop_critic_2_{time.strftime(r"%m_%d_%H:%M")}.pt')

        u = td3.actor(torch.tensor(x).float())
        u_noise = torch.normal(0, .1 * td3.u_max, size=u.shape)
        u = (u + u_noise).clip(-td3.u_max, td3.u_max).detach().cpu().numpy()

        x_next, reward, done, info = env.step(u)
        td3.replay_buffer.append([x, u, x_next, reward, done])
        x = x_next

        batch = td3.sample_replay_buffer()
        critic_loss = td3.critic_loss(batch)

        td3.critic_optim.zero_grad()
        critic_loss.backward()
        td3.critic_optim.step()

        #print(critic_loss.item())
        # Delayed model updates
        if i % td3.policy_update_period == 0:

            actor_loss = td3.actor_loss(batch)

            td3.actor_optim.zero_grad()
            actor_loss.backward()
            td3.actor_optim.step()

            for target_param, cr_param in zip(td3.critic_1_target.parameters(), td3.critic_1.parameters()):
                target_param.data.copy_(td3.tau * cr_param.data + (1 - td3.tau) * target_param.data)

            for target_param, cr_param in zip(td3.critic_2_target.parameters(), td3.critic_2.parameters()):
                target_param.data.copy_(td3.tau * cr_param.data + (1 - td3.tau) * target_param.data)

            for target_param, actor_param in zip(td3.actor_target.parameters(), td3.actor.parameters()):
                target_param.data.copy_(td3.tau * actor_param.data + (1 - td3.tau) * target_param.data)

        if done:
            x = env.reset()
            done = False


    print('train end: ', eval(env_name, td3.actor))
    return td3

def eval(env_name, actor, num_episodes=10):
    env = gym.make(env_name)
    tot_reward = 0

    for _ in range(num_episodes):
        x = env.reset()
        done = False
        while not done:
            u = actor(torch.tensor(x).float())
            x, reward, done, _ = env.step(u.detach().cpu().numpy())
            tot_reward += reward

    return tot_reward / num_episodes

def render(env_name, actor):
    env = gym.make(env_name)
    for i in range(10):
        x = env.reset()
        done = False
        while not done:
            u = actor(torch.tensor(x).float())
            env.render()
            x, reward, done, _ = env.step(u.detach().cpu().numpy())

if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    #env = gym.make('Hopper-v2')
    #x_dim = env.observation_space.shape[0]
    #u_dim = env.action_space.shape[0]
    #u_max = float(env.action_space.high[0])
    #td3 = TD3(x_dim, u_dim, u_max)

    #td3.actor.load_state_dict(torch.load('td3_hop_actor_05_07_08:47.pt', map_location=torch.device('cpu')))
    #render('Hopper-v2', td3.actor)
    #exit()

    torch.save(td3.actor.state_dict(), f'td3_hop_actor_{time.strftime(r"%m_%d_%H:%M")}.pt')
    torch.save(td3.critic_1.state_dict(), f'td3_hop_critic_1_{time.strftime(r"%m_%d_%H:%M")}.pt')
    torch.save(td3.critic_2.state_dict(), f'td3_hop_critic_2_{time.strftime(r"%m_%d_%H:%M")}.pt')
