import numpy as np
from numpy.linalg import inv

class LinUCB:
    def __init__(self, num_actions, num_features, alpha):
        # Set number of arms
        self.num_actions = num_actions
        # Number of context features
        self.num_features = num_features
        # explore-exploit parameter
        self.alpha = alpha
        # Instantiate A as a ndims×ndims matrix for each action
        self.A = np.zeros((self.num_actions, self.num_features, self.num_features))
        # Instantiate b as a 0 vector of length num_features.
        self.b = np.zeros((num_actions, self.num_features, 1))
        # set each A per action as identity matrix of size num_features
        for action in range(self.num_actions):
            self.A[action] = np.eye(self.num_features)

    def choose_action(self, context):
        # gains per each action
        p_t = np.zeros(self.num_actions)

        for i in range(self.num_actions):
            self.theta = inv(self.A[i]).dot(self.b[i])
            
            cntx = context.reshape(-1, 1)  # Reshape the context to a column vector
            p_t[i] = self.theta.T.dot(cntx) + self.alpha * np.sqrt(
                cntx.T.dot(inv(self.A[i]).dot(cntx)))

        action = int(np.random.choice(np.where(p_t == max(p_t))[0]))
        return action

    def update_model(self, context, action, reward):
        self.A[action] = self.A[action] + np.outer(context, context)
        self.b[action] = np.add(self.b[action].T, context * reward).reshape(self.num_features, 1)
