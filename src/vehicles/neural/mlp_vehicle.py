class MLPVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        instance_len = 20 # steps
        self.state_dim = 10
        self.obs_instance = np.zeros([self.samples_n, self.state_dim])
        # self.action_instance = [[0., 0.]]*20

        model_name = 'mlp_model'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core.mlp import MLP
        self.model = MLP()
        self.model.load_weights(exp_dir).expect_partial()

    def prep_obs_seq(self, obs_instance):
        obs_instance = np.float32(obs_instance)
        obs_instance[:, :-3] = self.scaler.transform(obs_instance[:, :-3])
        return obs_instance

    def update_obs_history(self, o_t):
        self.obs_instance[:, :] = o_t

    def act(self, obs):
        obs_instance = self.prep_obs_seq(self.obs_instance.copy())
        pred_dis = self.model(obs_instance)
        act_long = pred_dis.sample().numpy()[0][0]
        self.att = -1
        return act_long
