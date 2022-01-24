#guild run qlearning:train epochs_training=[3000] steps_max_training=[1000] steps_max_testing=[1000] \
#  alpha=linspace[0.35:.99:4] gamma=linspace[0.001:0.99:4] epsilon=linspace[0.3:0.99:4]


#guild run GNN:train epochs=[2001,1000] learning_rate=[0.01] hidden_channels=[32,48,64] model=['shallow','deep']
#guild run GNN:train -y epochs=[2000] learning_rate=[0.001] hidden_channels=[8,16,32] model='deep'

# Low epochs high D embedding
#guild run GNN:train epochs=[200,2000] learning_rate=[0.001] \
#  hidden_channels=[32,64,128] model=['deep'] out_channels=[9,32,64] target=['rating_sleep']

guild run GNN:train_gcn epochs=[50000] learning_rate=[0.001] \
  hidden_channels=[128] model=['deep'] out_channels=[64] target=['rating_sleep']
